############ USES A PRETRAINED (ON SVD EXTRACTED BASIS MODES) TRUNK NET ############

import torch
from torch.utils.data import DataLoader

torch.set_default_dtype(torch.float32)

import pickle, yaml
from pathlib import Path

from flumen import CausalFlowModel, print_gpu_info, TrajectoryDataset, TrunkNet, DeepONet
from flumen.train import EarlyStopping, train_step, validate

from argparse import ArgumentParser
import time

import wandb
import os

hyperparams = {
    'batch_size': 256,
    'modes':12,   
    'lr': 0.0005,
    'n_epochs': 1000,
    'es_patience': 30,
    'es_delta': 1e-7,
    'sched_patience': 5,
    'sched_factor': 2,
    'loss': "l1_relative",
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

def get_loss(which):
    if which == "mse":
        return torch.nn.MSELoss()
    elif which == "l1":
        return torch.nn.L1Loss()
    elif which == "l1_relative":
        return L1_relative
    elif which == "l1_relative_orthogonal_trunk":
        return l1_relative_orthogonal_trunk
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

    sys_args = ap.parse_args()
    data_path = Path(sys_args.load_path)
    run = wandb.init(project='flumen_spatial_galerkin', name=sys_args.name, config=hyperparams)

    
    with data_path.open('rb') as f:
        data = pickle.load(f)

    train_data = TrajectoryDataset(data["train"])
    val_data = TrajectoryDataset(data["val"])
    test_data = TrajectoryDataset(data["test"])  

    model_args = {
        'state_dim': train_data.state_dim,
        'control_dim': train_data.control_dim,
        'output_dim': train_data.output_dim,
        'modes':wandb.config['modes']}

    run_id = "DeepONet"
 

    model_metadata = {
        'args': model_args,
        'data_path': data_path.absolute().as_posix(),
        'data_settings': data["settings"],
        'data_args': data["args"]
    }
    model_name = f"DeepONet-{data_path.stem}-{sys_args.name}-{run_id}"

    # Prepare for saving the model
    model_save_dir = Path(
        f"./outputs/{sys_args.name}/{sys_args.name}_{run_id}")
    model_save_dir.mkdir(parents=True, exist_ok=True)

    # Save local copy of metadata
    with open(model_save_dir / "metadata.yaml", 'w') as f:
        yaml.dump(model_metadata, f)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = DeepONet(**model_args)
    model.to(device)

    optimiser = torch.optim.Adam(model.parameters(), lr=wandb.config['lr'])
    # optimiser = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=wandb.config['lr'])

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
        f"{'Loss (Val)':>16} :: {'Loss (Test)':>16} :: {'Best (Val)':>16}"

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
    main()
   

