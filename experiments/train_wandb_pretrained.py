############ USES A PRETRAINED (ON SVD EXTRACTED BASIS MODES) TRUNK NET ############

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

torch.set_default_dtype(torch.float32)

import pickle, yaml
from pathlib import Path

from flumen import CausalFlowModel, print_gpu_info, TrajectoryDataset, TrunkNet
from flumen.train import EarlyStopping, train_step, validate

from flumen.utils import trajectory,plot_space_time_flat_trajectory, plot_space_time_flat_trajectory_V2
from argparse import ArgumentParser
import time
import matplotlib.pyplot as plt

import wandb
import os

hyperparams = {
    'control_rnn_size': 128,
    'control_rnn_depth': 1,
    'encoder_size': 1,
    'encoder_depth': 1,
    'decoder_size': 1,
    'decoder_depth': 1,
    'batch_size': 32,
    'use_POD':False,
    'use_trunk':True,
    'use_petrov_galerkin':False, ## if False -> inputs will be projected using same basis functions of trunk and POD
    'unfreeze_epoch':10, ## From this epoch onwards, trunk will learn during online training
    'use_nonlinear':True, ## True: Nonlinearity at end, False: Inner product
    'use_fourier':False,
    'use_conv_encoder':False,
    'trunk_size':[100,100,100],
    'POD_modes':50,
    'trunk_modes':100,   
    'fourier_modes':50,
    'lr': 0.0005,
    'n_epochs': 1000,
    'es_patience': 30,
    'es_delta': 1e-7,
    'sched_patience': 5,
    'sched_factor': 2,
    'loss': "l1_loss_rejection",
}

def l1_relative_orthogonal_trunk(y_true,y_pred,basis_functions):
    ### orthogonal loss
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    loss_fn_orth = torch.nn.L1Loss().to(device)
    I = torch.eye(basis_functions.shape[1]).to(device)  # Identity matrix
    UTU = torch.matmul(basis_functions.T, basis_functions).to(device)  # Compute U^T 
    orthogonal_loss = loss_fn_orth(UTU, I)

    ### l1 relative loss
    abs_error = torch.abs(y_true - y_pred)
    variance = torch.mean((y_true - torch.mean(y_true))**2)    
    data_loss = torch.mean(abs_error) / torch.sqrt(variance)
    
    total_error = orthogonal_loss + data_loss
    return total_error, orthogonal_loss, data_loss 

def L1_relative(y_true, y_pred):

    abs_error = torch.abs(y_true - y_pred)
    variance = torch.mean((y_true - torch.mean(y_true))**2)    
    relative_error = torch.mean(abs_error) / torch.sqrt(variance)
    
    return relative_error

def L1_relative(y_true, y_pred):

    abs_error = torch.abs(y_true - y_pred)
    variance = torch.mean((y_true - torch.mean(y_true))**2)    
    relative_error = torch.mean(abs_error) / torch.sqrt(variance)
    return relative_error


def orthogonality_loss(U):
    loss_fn_orth = nn.L1Loss()
    I = torch.eye(U.shape[1], device=U.device)  # Identity matrix
    UTU = torch.matmul(U.T, U)  # Compute U^T 

    return loss_fn_orth(UTU, I)  # Minimize deviation from I

def unit_norm_loss(U):
    norms = torch.norm(U, dim=0)  # Compute column-wise norms
    loss_unit_norm = nn.L1Loss()
    return loss_unit_norm(norms, torch.ones_like(norms))  # Penalize deviations from 1

def total_loss(U_pred, U_true, alpha=15.0,beta=250.0):
    data_loss = L1_relative(U_pred, U_true)  # Reconstruction loss
    ortho_loss = orthogonality_loss(U_pred)  # Enforce U^T U = I
    norm_loss = unit_norm_loss(U_pred)  # Ensure unit norm
    total_loss = data_loss + alpha * ortho_loss + beta * norm_loss

    return total_loss, data_loss, alpha * ortho_loss, beta*norm_loss

def l1_loss_rejection(y_true,y_pred):
    '''samples points based on their magnitude, and then computes the L1 loss on the selected points'''
    Loss = nn.L1Loss()
    magnitudes = torch.abs(y_true) 
    probabilities = magnitudes / torch.sum(magnitudes)  # Normalize to get probabilities
    sampled_indices = torch.multinomial(probabilities, num_samples=20, replacement=False)
    y_true_sampled, y_pred_sampled = y_true[sampled_indices], y_pred[sampled_indices]
    return Loss(y_true_sampled, y_pred_sampled)

def get_loss(which):
    if which == "mse":
        return torch.nn.MSELoss()
    elif which == "l1":
        return torch.nn.L1Loss()
    elif which == "l1_relative":
        return L1_relative
    elif which == "l1_relative_orthogonal_trunk":
        return l1_relative_orthogonal_trunk
    elif which == "l1_loss_rejection":
        return l1_loss_rejection
    else:
        raise ValueError(f"Unknown loss {which}.")


def main():
    ap = ArgumentParser()

    ap.add_argument('load_path', type=str, help="Path to trajectory dataset")

    ap.add_argument('name', type=str, help="Name of the experiment.")

    ap.add_argument('--reset_noise',
                    action='store_true',
                    help="Regenerate the measurement noise.")

    ap.add_argument('--noise_std',
                    type=float,
                    default=None,
                    help="If reset_noise is set, set standard deviation ' \
                            'of the measurement noise to this value.")

    ap.add_argument('--pretrained_trunk',
                    type=str,
                    default=False,
                    help="Path to pretrained trunk model, if none is given the trunk will be trained before training the flow model.")
    
    sys_args = ap.parse_args()
    data_path = Path(sys_args.load_path)
    run = wandb.init(project='brian2', name=sys_args.name, config=hyperparams)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    with data_path.open('rb') as f:
        data = pickle.load(f)
    ### Pretrain trunk if no pretrained trunk is given
    if sys_args.pretrained_trunk == False:
        print("No pretrained trunk model given, training trunk model...")
        trunk_model = TrunkNet(in_size=256,out_size=wandb.config['trunk_modes'],hidden_size=wandb.config['trunk_size'],use_batch_norm=False)
        trunk_model.to(device)
        trunk_model.train()
        optimizer = torch.optim.Adam(trunk_model.parameters(), lr=1e-3)
        PHI = data['PHI'][:,:wandb.config['trunk_modes']].to(device)
        best_loss = 0.03
        locations = data['Locations'].view(-1,1).to(device)
        for epoch in range(0,200000):
        
            optimizer.zero_grad()
            PHI_pred = trunk_model(locations)
            total, rec_loss, ortho_loss, norm_loss = total_loss(PHI, PHI_pred, alpha=1.0, beta=0.1)  # Get all losses
            total.backward()
            optimizer.step()

            # save the model
            if total.item() < best_loss:
                best_loss = total.item()
                model_name = f"trunk_model-{data_path.stem}-{sys_args.name}"

                torch.save(trunk_model.state_dict(), f"{model_name}.pth")
                artifact = wandb.Artifact(name=f"{model_name}", type="model")
                artifact.add_file(f"{model_name}.pth")
                wandb.log_artifact(artifact)
                # print(f"Epoch {i+1}: Improved model saved! Total Loss: {total.item()}")

            if epoch % 5000 == 0: 
                print(f'epoch {epoch+1}, Total Loss {total.item()}, data Loss {rec_loss.item()}, ortho Loss {ortho_loss.item()}, norm Loss {norm_loss.item()}')
            wandb.log({
            'Trunk/epoch': epoch + 1,
            'Trunk/Total_Loss': total.item(),
            'Trunk/Data_Loss': rec_loss.item(),
            'Trunk/Orthogonal_Loss': ortho_loss.item(),
            'Trunk/Norm_Loss': norm_loss.item(),
        })
    else:
        print("Using pretrained trunk model...")
        trunk_path = Path(sys_args.pretrained_trunk)
        trunk_model = TrunkNet(in_size=256,out_size=wandb.config['trunk_modes'],hidden_size=wandb.config['trunk_size'],use_batch_norm=False)
        trunk_model.load_state_dict(torch.load(trunk_path))
        trunk_model.to(device)
        trunk_model.train()  

    train_data = TrajectoryDataset(data["train"])
    val_data = TrajectoryDataset(data["val"])
    test_data = TrajectoryDataset(data["test"])


    model_args = {
        'state_dim': int(train_data.state_dim),
        'control_dim': int(train_data.control_dim),
        'output_dim': int(train_data.output_dim),
        'control_rnn_size': wandb.config['control_rnn_size'],
        'control_rnn_depth': wandb.config['control_rnn_depth'],
        'encoder_size': wandb.config['encoder_size'],
        'encoder_depth': wandb.config['encoder_depth'],
        'decoder_size': wandb.config['decoder_size'],
        'decoder_depth': wandb.config['decoder_depth'],
        'use_POD': wandb.config['use_POD'],
        'use_trunk': wandb.config['use_trunk'],
        'use_petrov_galerkin': wandb.config['use_petrov_galerkin'],
        'use_nonlinear': wandb.config['use_nonlinear'],
        'use_fourier':wandb.config['use_fourier'],
        'use_conv_encoder':wandb.config['use_conv_encoder'],
        'trunk_size': wandb.config['trunk_size'],
        'POD_modes':wandb.config['POD_modes'],
        'trunk_modes':wandb.config['trunk_modes'],
        'fourier_modes':wandb.config['fourier_modes'],
        'use_batch_norm': False,
    }

    model_metadata = {
        'args': model_args,
        'data_path': data_path.absolute().as_posix(),
        'data_settings': data["settings"],
        'data_args': data["args"]
    }
    model_name = f"flow_model-{data_path.stem}-{sys_args.name}"

    # Prepare for saving the model
    model_save_dir = Path(
        f"./outputs/{sys_args.name}/{sys_args.name}")
    model_save_dir.mkdir(parents=True, exist_ok=True)

    # Save local copy of metadata
    with open(model_save_dir / "metadata.yaml", 'w') as f:
        yaml.dump(model_metadata, f)

    model = CausalFlowModel(**model_args,trunk_model=trunk_model)
    model.to(device)

    # Freeze the pretrained model 
    for param in trunk_model.parameters():
        param.requires_grad = False

    # optimiser = torch.optim.Adam(model.parameters(), lr=wandb.config['lr'])
    optimiser = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=wandb.config['lr'])

    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimiser,
        patience=wandb.config['sched_patience'],
        cooldown=0,
        factor=1. / wandb.config['sched_factor'])

    loss = get_loss(wandb.config["loss"])

    early_stop = EarlyStopping(es_patience=wandb.config['es_patience'],
                               es_delta=wandb.config['es_delta'])

    bs = wandb.config['batch_size']
    train_dl = DataLoader(train_data, batch_size=bs, shuffle=True)
    val_dl = DataLoader(val_data, batch_size=bs, shuffle=True)
    test_dl = DataLoader(test_data, batch_size=bs, shuffle=True)

    header_msg = f"{'Epoch':>5} :: {'Loss (Train)':>16} :: " \
        f"{'Loss (Val)':>16} :: {'Loss (Test)':>16} :: {'Best (Val)':>16} :: {'Loss orthogonal (Train)':>16}"

    print(header_msg)
    print('=' * len(header_msg))

    # Evaluate initial loss
    model.eval()
    train_loss = validate(train_dl, data['PHI'],data['Locations'],loss, model, device,epoch=0)
    val_loss = validate(val_dl, data['PHI'],data['Locations'],loss, model, device,epoch=0)
    test_loss = validate(test_dl,data['PHI'],data['Locations'],loss, model,device,epoch=0)

    early_stop.step(val_loss)
    print(
        f"{0:>5d} :: {train_loss:>16e} :: {val_loss:>16e} :: " \
        f"{test_loss:>16e} :: {early_stop.best_val_loss:>16e}"
    )

    start = time.time()

    for epoch in range(wandb.config['n_epochs']):
        model.train()
        if epoch == wandb.config['unfreeze_epoch']:
            print("Unfreezing the pretrained model's layers for fine-tuning...")
            for param in trunk_model.parameters():
                param.requires_grad = True
                optimiser = torch.optim.Adam(model.parameters(), lr=wandb.config['lr'])

        for example in train_dl:
            train_step(example, data['PHI'],data['Locations'],loss, model, optimiser, device,epoch)


        model.eval()
        train_loss = validate(train_dl,data['PHI'],data['Locations'], loss, model, device,epoch)
        val_loss = validate(val_dl, data['PHI'],data['Locations'],loss, model, device,epoch)
        test_loss = validate(test_dl, data['PHI'],data['Locations'],loss, model, device,epoch)

        sched.step(val_loss)
        early_stop.step(val_loss)

        print(
            f"{epoch + 1:>5d} :: {train_loss:>16e} :: {val_loss:>16e} :: " \
            f"{test_loss:>16e} :: {early_stop.best_val_loss:>16e}"
        )

        if early_stop.best_model:
            torch.save(model.state_dict(), model_save_dir / "state_dict.pth")
            run.log_model(model_save_dir.as_posix(), name=model_name)

            run.summary["Flownet/best_train"] = train_loss
            run.summary["Flownet/best_val"] = val_loss
            run.summary["Flownet/best_test"] = test_loss
            run.summary["Flownet/best_epoch"] = epoch + 1

            # visualize a test trajectory:
            y,x0_feed,t_feed,u_feed,deltas_feed = trajectory(data['test'],delta=1) # delta is hardcoded
            y_pred, basis_functions = model(x0_feed.to(device), u_feed.to(device), data['PHI'].to(device),data['Locations'].to(device),deltas_feed.to(device),epoch)
            test_loss_trajectory = torch.abs(y.to(device) - y_pred).sum(dim=1)  # Or .mean(dim=1) for mean L1
            # fig = plot_space_time_flat_trajectory(y,y_pred)
            fig = plot_space_time_flat_trajectory_V2(y,y_pred)
            wandb.log({"Flownet/Test trajectory": wandb.Image(fig),"Flownet/Best_epoch": epoch+1})

            # for example in test_dl:
            #         test_loss,y_pred, y, basis_functions = trajectory(example, data['PHI'],data['Locations'],loss, model, optimiser, device,epoch)
            #         y_pred_np = y_pred.cpu().numpy()
            #         y_np = y.cpu().cpu().numpy()
            #         fig, ax = plt.subplots()
            #         ax.plot(y_pred_np[0], label='Ground Truth')
            #         ax.plot(y_np[0], label='Prediction')
            #         ax.set_title(f'Epoch {epoch+1}, Test Loss: {test_loss:.4f}')
            #         ax.set_xlabel('Space')
            #         ax.set_ylabel('Output')
            #         ax.legend()
            #         wandb.log({"Flownet/Test trajectory": wandb.Image(fig),"Flownet/Best_epoch": epoch+1})
            #         plt.close(fig)

            #         break  # remove this break if you want to plot more samples

        wandb.log({
            'Flownet/time': time.time() - start,
            'Flownet/epoch': epoch + 1,
            'Flownet/lr': sched.get_last_lr()[0],
            'Flownet/train_loss': train_loss,
            'Flownet/val_loss': val_loss,
            'Flownet/test_loss': test_loss,
        })

        if early_stop.early_stop:
            print(f"{epoch + 1:>5d} :: --- Early stop ---")
            break

    train_time = time.time() - start

    print(f"Training took {train_time:.2f} seconds.")


if __name__ == '__main__':
    main()
   

