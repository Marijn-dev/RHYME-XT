import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pickle, yaml
from pathlib import Path
from RHYME_XT import print_gpu_info, TrajectoryDataset,TrajectoryDataset_DeepONet, TrunkNet,RHYME_XT_Model, DeepONet_Model
from RHYME_XT.train import EarlyStopping, train_step, validate_DeepONet, train_step_DeepONet
from RHYME_XT.utils import trajectory,plot_space_time_trajectory
from argparse import ArgumentParser
import time
import matplotlib.pyplot as plt
import wandb
import os

torch.set_default_dtype(torch.float32)

hyperparams = {
    'branch_size_ic': 120,
    'branch_depth_ic': 4,
    'branch_size_f': 120,
    'branch_depth_f': 4,
    'trunk_size': 120,
    'trunk_depth':8,
    'modes':250,
    'batch_size': 128,
    'location_scaling':1,                 # Scale the location inputs to the trunk net by this factor (normalization) 
    'lr': 0.00011614090101177696,
    'n_epochs': 1000,                        # Number of epochs to train complete model for
    'es_patience': 30,
    'es_delta': 1e-7,
    'sched_patience': 5,
    'sched_factor': 2,
    'train_loss': "MSE",
    'val_loss': "MSE"
}


def L1(y_true,y_pred):
    Loss = nn.L1Loss()
    data_loss = Loss(y_true,y_pred)
    return data_loss

def MSE(y_true,y_pred):
    Loss = nn.MSELoss()
    data_loss = Loss(y_true,y_pred)
    return data_loss

def get_loss(which):
    if which == "MSE":
        return MSE
    elif which == "L1":
        return L1


def main():
    ap = ArgumentParser()

    ap.add_argument('load_path', type=str, help="Path to trajectory dataset")

    ap.add_argument('name', type=str, help="Name of the experiment.")


    sys_args = ap.parse_args()
    data_path = Path(sys_args.load_path)
    run = wandb.init(project='DeepONet', name=sys_args.name, config=hyperparams)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    with data_path.open('rb') as f:
        data = pickle.load(f)

    train_data = TrajectoryDataset_DeepONet(data["train"])
    val_data = TrajectoryDataset_DeepONet(data["val"])
    test_data = TrajectoryDataset_DeepONet(data["test"])
    locations = data['Locations_online']*wandb.config['location_scaling'] 
     
    DeepONet_model_args = {
        'state_dim': int(data["train"].state_dim),
        'control_dim': int(data["train"].control_dim),
        'output_dim': int(data["train"].output_dim),
        'modes': wandb.config["modes"],
        'branch_size_ic': wandb.config["branch_size_ic"],
        'branch_depth_ic': wandb.config["branch_depth_ic"],
        'branch_size_f': wandb.config["branch_size_f"],
        'branch_depth_f': wandb.config["branch_depth_f"],
        'trunk_size': wandb.config["trunk_size"],
        'trunk_depth': wandb.config["trunk_depth"],
        'use_batch_norm': False,
    }
  

    model_metadata = {
        'args': DeepONet_model_args,
        'data_path': data_path.absolute().as_posix(),
        'location_scaling':wandb.config['location_scaling'],
        'data_settings': data["settings"],
        'data_args': data["args"]
    }

    model_name = f"DeepONet_model-{data_path.stem}-{sys_args.name}"

    # Prepare for saving the model 
    model_save_dir = Path(
        f"./outputs/{sys_args.name}/{sys_args.name}")
    model_save_dir.mkdir(parents=True, exist_ok=True)

    # Save local copy of metadata 
    with open(model_save_dir / "metadata.yaml", 'w') as f:
        yaml.dump(model_metadata, f)

    
    model = DeepONet_Model(**DeepONet_model_args)
    model.to(device)

    optimiser = torch.optim.Adam(model.parameters(), lr=wandb.config['lr'])

    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimiser,
        patience=wandb.config['sched_patience'],
        cooldown=0,
        factor=1. / wandb.config['sched_factor'])

    train_loss_fn = get_loss(wandb.config["train_loss"])
    val_loss_fn = get_loss(wandb.config["val_loss"]) # test loss uses val loss

    early_stop = EarlyStopping(es_patience=wandb.config['es_patience'],
                               es_delta=wandb.config['es_delta'])

    bs = wandb.config['batch_size']
    train_dl = DataLoader(train_data, batch_size=bs, shuffle=True)
    val_dl = DataLoader(val_data, batch_size=bs, shuffle=True)
    test_dl = DataLoader(test_data, batch_size=bs, shuffle=True)

    header_msg = f"{'Epoch':>5} :: {'Loss (Train)':>16} :: {'Data Loss (Val)':>16} :: {'Data Loss (Test)':>16} :: {'Best (Val)':>16}"

    print(header_msg)
    print('=' * len(header_msg))

    # Evaluate initial loss
    model.eval()
    train_loss = validate_DeepONet(train_dl,locations,train_loss_fn, model, device)
    val_loss = validate_DeepONet(val_dl,locations,train_loss_fn, model, device)
    test_loss = validate_DeepONet(test_dl,locations,train_loss_fn, model, device)

    early_stop.step(val_loss)

    print(
            f"{0:>5d} :: {train_loss:>16e} :: {val_loss:>16e} :: {test_loss:>16e} :: " 
            f"{test_loss:>16e} :: {early_stop.best_val_loss:>16e}"
    )
    start = time.time()

    ### Main training loop ###
    for epoch in range(wandb.config['n_epochs']):
        model.train()
        for example in train_dl:
            train_step_DeepONet(example,locations,train_loss_fn,model,optimiser, device)

        model.eval()
        train_loss = validate_DeepONet(train_dl,locations,train_loss_fn, model, device)
        val_loss = validate_DeepONet(val_dl,locations,train_loss_fn, model, device)
        test_loss = validate_DeepONet(test_dl,locations,train_loss_fn, model, device)

        sched.step(val_loss)
        early_stop.step(val_loss)
       
        print(
            f"{0:>5d} :: {train_loss:>16e} :: {val_loss:>16e} :: {test_loss:>16e} :: " 
            f"{test_loss:>16e} :: {early_stop.best_val_loss:>16e}"
        )

        if early_stop.best_model:
            torch.save(model.state_dict(), model_save_dir / "state_dict.pth")
            run.log_model(model_save_dir.as_posix(), name=model_name)

            run.summary["DeepONet/best_train"] = train_loss
            run.summary["DeepONet/best_val"] = val_loss
            run.summary["DeepONet/best_test"] = test_loss
            run.summary["DeepONet/best_epoch"] = epoch + 1

            # ### Visualize trajectory in WB ###
            # y,x0_feed,t_feed,u_feed,deltas_feed = trajectory(data['test'],trajectory_index=0,delta=test_data.delta) 
            # y_pred, basis_functions = model(x0_feed.to(device), u_feed.to(device),x_out_test.view(-1,1).to(device),deltas_feed.to(device),x_in.view(-1,1).to(device))
            # time_dim, space_dim = y.shape
            # time_indices=[0, int(time_dim*0.25), int(time_dim*0.5),  int(time_dim*0.75), int(time_dim*0.95)]    # depends on n_samples
            # space_indices=[0, int(space_dim*0.25), int(space_dim*0.5), int(space_dim*0.75),int(space_dim*0.95)] # depends on n_neurons
            # fig = plot_space_time_trajectory(y,y_pred,time_indices=time_indices,space_indices=space_indices)
            # wandb.log({"RHYME-XT/Test trajectory": wandb.Image(fig),"RHYME-XT/Best_epoch": epoch+1})

        wandb.log({
            'DeepONet/time': time.time() - start,
            'DeepONet/epoch': epoch + 1,
            'DeepONet/lr': optimiser.param_groups[0]["lr"],
            'DeepONet/train_loss_data': train_loss,
            'DeepONet/val_loss': val_loss,
            'DeepONet/test_loss': test_loss,
        })

        if early_stop.early_stop:
            print(f"{epoch + 1:>5d} :: --- Early stop ---")
            break

    train_time = time.time() - start

    print(f"Training took {train_time:.2f} seconds.")


if __name__ == '__main__':
    main()
   

