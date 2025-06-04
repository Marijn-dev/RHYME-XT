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
    elif which == "L1_orthogonal":
        return L1_orthogonal
    elif which == "MSE_orthogonal":
        return MSE_orthogonal
    else:
        raise ValueError(f"Unknown loss {which}.")

def main():

    run = wandb.init(project='LIF_L1_sweep')
    config = wandb.config
    
    data_path = '../data/brian_T2500_S100_LOOP.pkl' # hardcoded for now
    data_path = Path(data_path)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    with data_path.open('rb') as f:
        data = pickle.load(f)

    train_data = TrajectoryDataset(data["train"],max_seq_len=wandb.config['max_seq_len'],n_samples=wandb.config['n_samples'])
    val_data = TrajectoryDataset(data["val"])
    test_data = TrajectoryDataset(data["test"])

    trunk_path = '../models_trunk/brian2/trunk_model-brian_T2500_S100_LOOP.pth'
    modes = wandb.config['trunk_modes'] if wandb.config['trunk_modes']<int(train_data.state_dim) else int(train_data.state_dim)
    trunk_path = Path(trunk_path)
    trunk_model = TrunkNet(in_size=256,out_size=modes,hidden_size=wandb.config['trunk_size_svd'],use_batch_norm=False)
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
    model_name = f"{run.id}"

    # Prepare for saving the model
    model_save_dir = Path(
        f"./outputs/models/{model_name}")
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
            for param in trunk_model.parameters():
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



if __name__ == "__main__":
    main()
   

