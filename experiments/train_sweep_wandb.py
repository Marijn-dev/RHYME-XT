import torch
from torch.utils.data import DataLoader

torch.set_default_dtype(torch.float32)

import pickle, yaml
from pathlib import Path

from flumen import CausalFlowModel, print_gpu_info, TrajectoryDataset
from flumen.train import EarlyStopping, train_step, validate

from argparse import ArgumentParser
import time

import wandb

def get_loss(which):
    if which == "mse":
        return torch.nn.MSELoss()
    elif which == "l1":
        return torch.nn.L1Loss()
    else:
        raise ValueError(f"Unknown loss {which}.")


def train_main():
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

    sys_args = ap.parse_args()
    data_path = Path(sys_args.load_path)
    run = wandb.init(project='flumen_spatial_sweep_regular', name=sys_args.name, config=wandb.config)

    ## if conv is on, POD and fourier cant be on
    # if wandb.config['use_conv_encoder'] == True and (wandb.config['use_POD'] == True or wandb.config['use_fourier'] == True):
    #     print("invalid combination, skip run")
    #     return
    
    with data_path.open('rb') as f:
        data = pickle.load(f)

    train_data = TrajectoryDataset(data["train"])
    val_data = TrajectoryDataset(data["val"])
    test_data = TrajectoryDataset(data["test"])
    model_args = {
        'state_dim': train_data.state_dim,
        'control_dim': train_data.control_dim,
        'output_dim': train_data.output_dim,
        'control_rnn_size': wandb.config['control_rnn_size'],
        'control_rnn_depth': wandb.config['control_rnn_depth'],
        'encoder_size': wandb.config['encoder_size'],
        'encoder_depth': wandb.config['encoder_depth'],
        'decoder_size': wandb.config['decoder_size'],
        'decoder_depth': wandb.config['decoder_depth'],
        'use_POD': wandb.config['use_POD'],
        'use_trunk': wandb.config['use_trunk'],
        'use_petrov_galerkin': wandb.config['use_petrov_galerkin'],
        'use_fourier':wandb.config['use_fourier'],
        'use_conv_encoder':wandb.config['use_conv_encoder'],
        'trunk_size': wandb.config['trunk_size'],
        'POD_modes':wandb.config['POD_modes'],
        'trunk_modes':wandb.config['trunk_modes'],
        'fourier_modes':wandb.config['fourier_modes'],
        'use_batch_norm': True,
    }
    run_id = ""
    run_id += f"Petrov_galerkin_{model_args['use_petrov_galerkin']}_"

    if model_args['use_fourier']:
        run_id += f"Fourier_Modes_{model_args['fourier_modes']}_" 
    if model_args['use_trunk']: 
        run_id += f"Trunk_Modes_{model_args['trunk_modes']}_" 
    if model_args['use_POD']: 
        run_id += f"POD_Modes_{model_args['POD_modes']}_" 
    if model_args['use_conv_encoder']: 
        run_id += f"CONV"     
    if model_args['use_POD'] == False and model_args['use_fourier'] == False and model_args['use_trunk'] == False and model_args['use_conv_encoder']:  
        run_id += "Regular"  

    model_metadata = {
        'args': model_args,
        'data_path': data_path.absolute().as_posix(),
        'data_settings': data["settings"],
        'data_args': data["args"]
    }
    model_name = f"flow_model-{data_path.stem}-{sys_args.name}-{run_id}"

    # Prepare for saving the model
    model_save_dir = Path(
        f"./outputs/{sys_args.name}/{sys_args.name}_{run_id}")
    model_save_dir.mkdir(parents=True, exist_ok=True)

    # Save local copy of metadata
    with open(model_save_dir / "metadata.yaml", 'w') as f:
        yaml.dump(model_metadata, f)

    model = CausalFlowModel(**model_args)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    optimiser = torch.optim.Adam(model.parameters(), lr=wandb.config['lr'])
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimiser,
        patience=wandb.config['sched_patience'],
        cooldown=0,
        factor=1. / wandb.config['sched_factor'])

    loss = get_loss(wandb.config["loss"]).to(device)

    early_stop = EarlyStopping(es_patience=wandb.config['es_patience'],
                               es_delta=wandb.config['es_delta'])

    bs = wandb.config['batch_size']
    train_dl = DataLoader(train_data, batch_size=bs, shuffle=True)
    val_dl = DataLoader(val_data, batch_size=bs, shuffle=True)
    test_dl = DataLoader(test_data, batch_size=bs, shuffle=True)

    header_msg = f"{'Epoch':>5} :: {'Loss (Train)':>16} :: " \
        f"{'Loss (Val)':>16} :: {'Loss (Test)':>16} :: {'Best (Val)':>16}"

    print(header_msg)
    print('=' * len(header_msg))

    # Evaluate initial loss
    model.eval()
    train_loss = validate(train_dl, data['PHI'],data['Locations'],loss, model, device)
    val_loss = validate(val_dl, data['PHI'],data['Locations'],loss, model, device)
    test_loss = validate(test_dl,data['PHI'],data['Locations'],loss, model,device)

    early_stop.step(val_loss)
    print(
        f"{0:>5d} :: {train_loss:>16e} :: {val_loss:>16e} :: " \
        f"{test_loss:>16e} :: {early_stop.best_val_loss:>16e}"
    )

    start = time.time()

    for epoch in range(wandb.config['n_epochs']):
        model.train()
        for example in train_dl:
            train_step(example, data['PHI'],data['Locations'],loss, model, optimiser, device)

        model.eval()
        train_loss = validate(train_dl,data['PHI'],data['Locations'], loss, model, device)
        val_loss = validate(val_dl, data['PHI'],data['Locations'],loss, model, device)
        test_loss = validate(test_dl, data['PHI'],data['Locations'],loss, model, device)

        sched.step(val_loss)
        early_stop.step(val_loss)

        print(
            f"{epoch + 1:>5d} :: {train_loss:>16e} :: {val_loss:>16e} :: " \
            f"{test_loss:>16e} :: {early_stop.best_val_loss:>16e}"
        )

        if early_stop.best_model:
            torch.save(model.state_dict(), model_save_dir / "state_dict.pth")
            run.log_model(model_save_dir.as_posix(), name=model_name)

            run.summary["best_train"] = train_loss
            run.summary["best_val"] = val_loss
            run.summary["best_test"] = test_loss
            run.summary["best_epoch"] = epoch + 1

        wandb.log({
            'time': time.time() - start,
            'epoch': epoch + 1,
            'lr': sched.get_last_lr()[0],
            'train_loss': train_loss,
            'val_loss': val_loss,
            'test_loss': test_loss,
        })

        if early_stop.early_stop:
            print(f"{epoch + 1:>5d} :: --- Early stop ---")
            break

    train_time = time.time() - start

    print(f"Training took {train_time:.2f} seconds.")


if __name__ == '__main__':
    # Define the sweep configuration with your hyperparameters
    sweep_configuration = {
    'method': 'random',  # Random search for hyperparameter combinations
    'name': 'rnn_sweep_example',
    'metric': {'name': 'val_loss', 'goal': 'minimize'},
    'parameters': {
        'control_rnn_size': {'values': [64, 128,256]}, 
        'control_rnn_depth': {'values': [1,2, 3,4]}, 
        'encoder_size': {'values': [1, 2,3,4]},  
        'encoder_depth': {'values': [1, 2,3,4]}, 
        'decoder_size': {'values': [1, 2,3,4]},  
        'decoder_depth': {'values': [1, 2, 3,4]}, 
        'batch_size': {'values': [64, 128, 256]}, 
        'use_POD': {'values': [True,False]},  
        'use_trunk': {'values': [True,False]},  
        'use_petrov_galerkin': {'values': [True,False]}, 
        'use_fourier': {'values': [True,False]}, 
        'use_conv_encoder': {'values': [True,False]}, 
        'trunk_size': {
            'values': [
                [100, 100, 100],  
                [100, 100, 100,100],   
                [150, 150,150],      
                [250, 250,250],      
                [100, 100]         
            ]
        },  
        'POD_modes': {'values': [50]},  
        'trunk_modes': {'values': [5,50,100,250,500]},  
        'fourier_modes': {'values': [8, 24,100,200]}, 
        'lr': {'max': 0.01, 'min': 1e-5},  
        'n_epochs': {'values': [500, 1000]},  
        'es_patience': {'values': [10, 20]},  
        'es_delta': {'values': [1e-7, 1e-6]},  
        'sched_patience': {'values': [5, 10]},  
        'sched_factor': {'values': [1.5, 2]}, 
        'loss': {'values': ['mse','l1']}  
        }
        }
    # Initialize sweep
    # Set the environment variable to disable flapping
    import os
    os.environ["WANDB_AGENT_DISABLE_FLAPPING"] = "True"
    sweep_id = wandb.sweep(sweep_configuration, project="flumen_spatial_sweep_regular")
    # Step 5: Start the Sweep Agent
    wandb.agent(sweep_id, function=train_main)