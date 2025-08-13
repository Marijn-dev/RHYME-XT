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
    'control_rnn_size': 250,
    'control_rnn_depth': 1,
    'encoder_size': 2,
    'encoder_depth': 2,
    'decoder_size': 1,
    'decoder_depth': 3,
    'batch_size': 64,
    'unfreeze_epoch':1000, ## From this epoch onwards, trunk will learn during online training
    'use_nonlinear':True, ## True: Nonlinearity at end, False: Inner product
    'IC_encoder_decoder':False, # True: encoder and decoder enforce initial condition
    'regular':False, # True: standard flow model
    'use_conv_encoder':False,
    'trunk_size_svd':[100,100,100,100], # hidden size of the trunk modeled as SVD
    'trunk_size_extra':[100,100,100], # hidden size of the trunk modeled as extra layers
    'NL_size':[50,50], # hidden size of nonlinearity at end, only used if use_nonlinear is True
    'trunk_modes':50,   # if bigger than state dim, second trunk_extra will be used
    'lr': 0.00011614090101177696,
    'max_seq_len': 20,  # Maximum sequence length for training dataset (-1 for full sequences)
    'n_samples': 4, # Number of samples to use for training dataset when max_seq_len is NOT set to -1
    'n_epochs': 1000,
    'es_patience': 30,
    'es_delta': 1e-7,
    'sched_patience': 5,
    'sched_factor': 2,
    'train_loss': "L1",
    'val_loss': "L1"
}

def L1_relative_orthogonal_trunk(y_true,y_pred,basis_functions):
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

def L1_orthogonal(y_true,y_pred,basis_functions,alfa=1,beta=0.1):
    '''returns data loss y_true and y_pred and orthogonal loss of trunk'''
    # data_loss = l1_loss_rejection
    data_loss_v, _ = L1(y_true,y_pred,basis_functions)  # Reconstruction loss
    ortho_loss = orthogonality_loss(basis_functions)  # Enforce U^T U = I
    norm_loss = unit_norm_loss(basis_functions)  # Ensure unit norm

    total_loss = data_loss_v + alfa * ortho_loss + beta * norm_loss
    return total_loss, data_loss_v

def MSE_orthogonal(y_true,y_pred,basis_functions,alfa=1,beta=0.1):
    '''returns data loss y_true and y_pred and orthogonal loss of trunk'''
    # data_loss = l1_loss_rejection
    data_loss_v, _ = MSE(y_true,y_pred,basis_functions)  # Reconstruction loss
    ortho_loss = orthogonality_loss(basis_functions)  # Enforce U^T U = I
    norm_loss = unit_norm_loss(basis_functions)  # Ensure unit norm

    total_loss = data_loss_v + alfa * ortho_loss + beta * norm_loss
    return total_loss, data_loss_v

def orthogonality_loss(U):
    loss_fn_orth = nn.L1Loss()
    I = torch.eye(U.shape[1], device=U.device)  # Identity matrix
    UTU = torch.matmul(U.T, U)  # Compute U^T 

    return loss_fn_orth(UTU, I)  # Minimize deviation from I

def unit_norm_loss(U):
    norms = torch.norm(U, dim=0)  # Compute column-wise norms
    loss_unit_norm = nn.L1Loss()
    return loss_unit_norm(norms, torch.ones_like(norms))  # Penalize deviations from 1

def total_loss(U_pred, U_true, alpha=1.0,beta=0.1):
    data_loss = L1_relative(U_pred, U_true)  # Reconstruction loss
    ortho_loss = orthogonality_loss(U_pred)  # Enforce U^T U = I
    norm_loss = unit_norm_loss(U_pred)  # Ensure unit norm
    total_loss = data_loss + alpha * ortho_loss + beta * norm_loss

    return total_loss, data_loss, alpha * ortho_loss, beta*norm_loss

def L1_loss_rejection(y_true,y_pred,basis_functions=0,num_samples=50):
    '''samples points based on their magnitude, and then computes the L1 loss on the selected points'''
    Loss = nn.L1Loss()
    
    magnitudes = torch.abs(y_true)
    probs = magnitudes / (torch.sum(magnitudes,dim=1,keepdim=True)+ 1e-10)  # Normalize to get probabilities
    probs = probs + 1e-5  # Avoid zero probabilities
    indices = torch.stack([
        torch.multinomial(probs[i], num_samples=num_samples, replacement=False)
        for i in range(y_true.shape[0])
    ])
    batch_indices = torch.arange(y_true.shape[0]).unsqueeze(1)
    y_true_sampled = y_true[batch_indices,indices]
    y_pred_sampled = y_pred[batch_indices,indices]
    Loss_v = Loss(y_true_sampled, y_pred_sampled)
    return Loss_v, Loss_v

def L1(y_true,y_pred,_):
    Loss = nn.L1Loss()
    Loss_v = Loss(y_true,y_pred)
    return Loss_v, Loss_v

def MSE(y_true,y_pred,_):
    Loss = nn.MSELoss()
    Loss_v = Loss(y_true,y_pred)
    return Loss_v, Loss_v

def get_loss(which):
    if which == "MSE":
        return MSE
    elif which == "L1":
        return L1
    elif which == "L1_relative":
        return L1_relative
    elif which == "L1_relative_orthogonal_trunk":
        return L1_relative_orthogonal_trunk
    elif which == "L1_loss_rejection":
        return L1_loss_rejection
    elif which == "L1_orthogonal":
        return L1_orthogonal
    elif which == "MSE_orthogonal":
        return MSE_orthogonal
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
    run = wandb.init(project='Noise', name=sys_args.name, config=hyperparams)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    with data_path.open('rb') as f:
        data = pickle.load(f)

    ### add noise to clean dataset ###
    if sys_args.reset_noise == True:
        print("add noise to IC and output with STD:",sys_args.noise_std)
        train_data = TrajectoryDataset(data["train"],max_seq_len=wandb.config['max_seq_len'],n_samples=wandb.config['n_samples'],noise_std=sys_args.noise_std)
        val_data = TrajectoryDataset(data["val"],noise_std=sys_args.noise_std)
    else:   
        print("No noise")
        train_data = TrajectoryDataset(data["train"],max_seq_len=wandb.config['max_seq_len'],n_samples=wandb.config['n_samples'])
        val_data = TrajectoryDataset(data["val"])

    test_data = TrajectoryDataset(data["test"])

    ### Pretrain trunk if no pretrained trunk is given ###
    if sys_args.pretrained_trunk == False:
        print("No pretrained trunk model given, training trunk model...")
        modes = wandb.config['trunk_modes'] if wandb.config['trunk_modes']<int(train_data.state_dim) else int(train_data.state_dim)
        trunk_model = TrunkNet(in_size=256,out_size=modes,hidden_size=wandb.config['trunk_size_svd'],use_batch_norm=False)
        trunk_model.to(device)
        trunk_model.train()
        optimizer = torch.optim.Adam(trunk_model.parameters(), lr=1e-3)
        PHI = data['PHI_01'][:,:wandb.config['trunk_modes']].to(device)
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
    
    ### Use pretrained trunk model ###
    else:
        print("Using pretrained trunk model...")
        modes = wandb.config['trunk_modes'] if wandb.config['trunk_modes']<int(train_data.state_dim) else int(train_data.state_dim)
        trunk_path = Path(sys_args.pretrained_trunk)
        trunk_model = TrunkNet(in_size=256,out_size=100,hidden_size=wandb.config['trunk_size_svd'],use_batch_norm=False)
        trunk_model.load_state_dict(torch.load(trunk_path))
        trunk_model.to(device)
        trunk_model.train()  

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
        'use_nonlinear': wandb.config['use_nonlinear'],
        'IC_encoder_decoder':wandb.config['IC_encoder_decoder'],
        'regular': wandb.config['regular'],
        'use_conv_encoder':wandb.config['use_conv_encoder'],
        'trunk_size_svd': wandb.config['trunk_size_svd'],
        'trunk_size_extra': wandb.config['trunk_size_extra'],
        'trunk_modes':wandb.config['trunk_modes'],
        'NL_size':wandb.config['NL_size'],
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
    for param in model.trunk_svd.parameters():
        param.requires_grad = False

    # optimiser = torch.optim.Adam(model.parameters(), lr=wandb.config['lr'])
    optimiser = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=wandb.config['lr'])

    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimiser,
        patience=wandb.config['sched_patience'],
        cooldown=0,
        factor=1. / wandb.config['sched_factor'])

    train_loss_fn = get_loss(wandb.config["train_loss"])
    val_loss_fn = get_loss(wandb.config["val_loss"])

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
    train_loss, train_loss_data = validate(train_dl,data['Locations'],train_loss_fn, model, device)
    _, val_loss = validate(val_dl,data['Locations'],val_loss_fn, model, device)
    _, test_loss = validate(test_dl,data['Locations'],val_loss_fn, model,device)

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
            for param in model.trunk_svd.parameters():
                param.requires_grad = True
            optimiser = torch.optim.Adam(model.parameters(), lr=wandb.config['lr'])

        for example in train_dl:
            train_step(example,data['Locations'],train_loss_fn, model, optimiser, device)


        model.eval()
        train_loss, train_loss_data = validate(train_dl,data['Locations'], train_loss_fn, model, device)
        _, val_loss = validate(val_dl, data['Locations'],val_loss_fn, model, device)
        _, test_loss = validate(test_dl,data['Locations'],val_loss_fn, model, device)

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

            ### Visualize trajectory in WB ###
            y,x0_feed,t_feed,u_feed,deltas_feed = trajectory(data['test'],0,delta=1) # delta is hardcoded
            y_pred, basis_functions = model(x0_feed.to(device), u_feed.to(device),data['Locations'].to(device),deltas_feed.to(device))
            test_loss_trajectory = torch.abs(y.to(device) - y_pred).sum(dim=1)  # Or .mean(dim=1) for mean L1
            fig = plot_space_time_flat_trajectory_V2(y,y_pred)
            wandb.log({"Flownet/Test trajectory": wandb.Image(fig),"Flownet/Best_epoch": epoch+1})

        wandb.log({
            'Flownet/time': time.time() - start,
            'Flownet/epoch': epoch + 1,
            'Flownet/lr': sched.get_last_lr()[0],
            'Flownet/train_loss': train_loss,
            'Flownet/train_loss_data': train_loss_data,
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
   

