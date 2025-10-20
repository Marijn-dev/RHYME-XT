import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pickle, yaml
from pathlib import Path
from RHYME_XT import print_gpu_info, TrajectoryDataset, TrunkNet,RHYME_XT_Model
from RHYME_XT.train import EarlyStopping, train_step, validate
from RHYME_XT.utils import trajectory,plot_space_time_trajectory
from argparse import ArgumentParser
import time
import matplotlib.pyplot as plt
import wandb
import os

torch.set_default_dtype(torch.float32)

hyperparams = {
    'control_rnn_size': 250,
    'control_rnn_depth': 1,
    'encoder_size': 2,
    'encoder_depth': 2,
    'decoder_size': 1,
    'decoder_depth': 3,
    'batch_size': 64,
    'unfreeze_epoch':1000,                  # From this epoch onwards, trunk svd net will learn during online training
    'use_nonlinear':True,                   # True: Nonlinearity, False: Inner product
    'NL_size':[50,50],                      # Hidden size of nonlinearity at end
    'IC_encoder_decoder':False,             # True: Initial Condition encoder decoder, False: Regular
    'trunk_modes_svd':100,                  # Number of modes used from SVD, max value = min(Nx,trunk_modes_svd)   
    'trunk_size_svd':[100,100,100,100],     # Hidden size of the trunk net modelled after the SVD
    'fourier_features': 128,                # Number of fourier features to use in the svd trunk net 
    'trunk_modes_extra':0,                  # Number of extra modes added to the trunk net       
    'trunk_size_extra':[100,100,100],       # Hidden size of the added trunk net
    'location_scaling':100,                 # Scale the location inputs to the trunk net by this factor (normalization)
    'lr': 0.00011614090101177696,
    'max_seq_len': 20,                      # Maximum sequence length used for training, -1 for full sequences
    'n_samples': 4,                         # Number of samples to use for training when max_seq_len is not -1
    'n_epochs': 100,                        # Number of epochs to train complete model for
    'n_epochs_trunk': 100000,               # Number of epochs to train trunk model for (only used if no pretrained trunk is given)
    'es_patience': 10,
    'es_delta': 1e-7,
    'sched_patience': 5,
    'sched_factor': 2,
    'num_locations': 1.0,                   # Ratio of equispaced output sensor locations used for training, 1: all locations, 0.5: half the locations etc
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
    data_loss, _ = L1(y_true,y_pred,basis_functions)  # Reconstruction loss
    ortho_loss = orthogonality_loss(basis_functions)  # Enforce U^T U = I
    norm_loss = unit_norm_loss(basis_functions)  # Ensure unit norm

    total_loss = data_loss + alfa * ortho_loss + beta * norm_loss
    return total_loss, data_loss

def MSE_orthogonal(y_true,y_pred,basis_functions,alfa=1,beta=0.1):
    '''returns data loss y_true and y_pred and orthogonal loss of trunk'''
    # data_loss = l1_loss_rejection
    data_loss, _ = MSE(y_true,y_pred,basis_functions)  # Reconstruction loss
    ortho_loss = orthogonality_loss(basis_functions)  # Enforce U^T U = I
    norm_loss = unit_norm_loss(basis_functions)  # Ensure unit norm

    total_loss = data_loss + alfa * ortho_loss + beta * norm_loss
    return total_loss, data_loss

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
    data_loss = Loss(y_true,y_pred)
    return data_loss, data_loss # returns data loss twice for compatibility with other loss functions

def MSE(y_true,y_pred,_):
    Loss = nn.MSELoss()
    data_loss = Loss(y_true,y_pred)
    return data_loss, data_loss # returns data loss twice for compatibility with other loss functions

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
                    help="Path to pretrained trunk model, if none is given the trunk will be trained before training the branch model.")


    ap.add_argument('--pretrained_branch',
                    type=str,
                    default=False,
                    help="Path to pretrained model, if a model is given it will used to initialize the new model with. Hyperparameters must match.")
    
    sys_args = ap.parse_args()
    data_path = Path(sys_args.load_path)
    run = wandb.init(project='RHYME-XT', name=sys_args.name, config=hyperparams)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    with data_path.open('rb') as f:
        data = pickle.load(f)

    ### Noise settings ###
    if sys_args.reset_noise == True:
        print("add noise to IC and output with STD:",sys_args.noise_std)
        train_data = TrajectoryDataset(data["train"],max_seq_len=wandb.config['max_seq_len'],n_samples=wandb.config['n_samples'],noise_std=sys_args.noise_std)
        val_data = TrajectoryDataset(data["val"],noise_std=sys_args.noise_std)
    else:   
        print("No noise")
        train_data = TrajectoryDataset(data["train"],max_seq_len=wandb.config['max_seq_len'],n_samples=wandb.config['n_samples'])
        val_data = TrajectoryDataset(data["val"])

    # Don't add noise to test data
    test_data = TrajectoryDataset(data["test"])

    ### Select locations for training ###
    scaling = wandb.config['location_scaling'] # scale locations values
    locations = data['Locations']*scaling
    steps = int(int(train_data.state_dim) * wandb.config['num_locations'])  # Number of locations to use for training
    selected_indices = torch.linspace(0, int(train_data.state_dim)-1, steps=steps).long() # Pick locations for training
    x_out_train = locations[selected_indices]
    x_out_test = locations
    x_in = locations

    ### Pretrain Trunknet if no pretrained trunk is given ###
    if sys_args.pretrained_trunk == False:
        print("No pretrained trunk model given, training trunk model...")
        trunk_model = TrunkNet(in_size=2,fourier_features=wandb.config['fourier_features'],out_size=wandb.config['trunk_modes_svd'],hidden_size=wandb.config['trunk_size_svd'],use_batch_norm=False)
        trunk_model.to(device)
        trunk_model.train()
        optimizer = torch.optim.Adam(trunk_model.parameters(), lr=1e-3)
        PHI = data["PHI"][:,:wandb.config['trunk_modes_svd']].to(device)
        print('Training on ground truth PHI with shape:', PHI.shape)
        locations = x_out_train.view(-1,1).to(device)
        for epoch in range(0,wandb.config['n_epochs_trunk']):
        
            optimizer.zero_grad()
            PHI_pred = trunk_model(locations)
            total, rec_loss, ortho_loss, norm_loss = total_loss(PHI, PHI_pred, alpha=1.0, beta=0.1)  # Get all losses
            total.backward()
            optimizer.step()

            # Log the loss
            if epoch % 5000 == 0: 
                print(f'epoch {epoch+1}, Total Loss {total.item()}, data Loss {rec_loss.item()}, ortho Loss {ortho_loss.item()}, norm Loss {norm_loss.item()}')

            # Wandb logging
            wandb.log({
            'Trunk/epoch': epoch + 1,
            'Trunk/Total_Loss': total.item(),
            'Trunk/Data_Loss': rec_loss.item(),
            'Trunk/Orthogonal_Loss': ortho_loss.item(),
            'Trunk/Norm_Loss': norm_loss.item(),
        })
            
        # Save the trunk model
        model_name = f"trunk_model-{data_path.stem}-{sys_args.name}"
        torch.save(trunk_model.state_dict(), f"{model_name}.pth")
        artifact = wandb.Artifact(name=f"{model_name}", type="model")
        artifact.add_file(f"{model_name}.pth")
        wandb.log_artifact(artifact)

    ### Use pretrained trunk model ###
    else:
        print("Using pretrained trunk model...")
        trunk_path = Path(sys_args.pretrained_trunk)
        trunk_model = TrunkNet(in_size=256,out_size=wandb.config['trunk_modes_svd'],hidden_size=wandb.config['trunk_size_svd'],use_batch_norm=False)
        trunk_model.load_state_dict(torch.load(trunk_path))
        trunk_model.to(device)
        trunk_model.train()  

    RHYME_XT_model_args = {
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
        'trunk_size_svd': wandb.config['trunk_size_svd'],
        'trunk_size_extra': wandb.config['trunk_size_extra'],
        'trunk_modes_svd':wandb.config['trunk_modes_svd'],
        'trunk_modes_extra':wandb.config['trunk_modes_extra'],
        'NL_size':wandb.config['NL_size'],
        'use_batch_norm': False,
    }

    trunk_args = {
        'in_size': 2,
        'fourier_features': wandb.config['fourier_features'],
        'out_size': wandb.config['trunk_modes_svd'],
        'hidden_size': wandb.config['trunk_size_svd'],
        'use_batch_norm': False
    }

    model_metadata = {
        'args': RHYME_XT_model_args,
        'trunk_args': trunk_args,
        'location_scaling': scaling,
        'data_path': data_path.absolute().as_posix(),
        'data_settings': data["settings"],
        'data_args': data["args"]
    }

    model_name = f"RHYME_XT_model-{data_path.stem}-{sys_args.name}"

    # Prepare for saving the model 
    model_save_dir = Path(
        f"./outputs/{sys_args.name}/{sys_args.name}")
    model_save_dir.mkdir(parents=True, exist_ok=True)

    # Save local copy of metadata 
    with open(model_save_dir / "metadata.yaml", 'w') as f:
        yaml.dump(model_metadata, f)

    # No pretrained branch model is given, train from scratch
    if sys_args.pretrained_branch == False:
        print("No pretrained branch model given, training from scratch...")
        model = RHYME_XT_Model(**RHYME_XT_model_args,trunk_model=trunk_model)
        model.to(device)

    else: # pretrained branch model is given (pre-loading strategy)
        print("Pretrained branch model given...")
        RHYME_XT_path = Path(sys_args.pretrained_branch)
        model = RHYME_XT_Model(**RHYME_XT_model_args,trunk_model=trunk_model)
        model.load_state_dict(torch.load(RHYME_XT_path))
        model.trunk_svd = trunk_model  # Replace the loaded in trunk model with correct one
        model.to(device)
        model.train() 
    

    ### Optimization settings ###

    # Uncomment parts to freeze certain parts of the model during training
    # # Freeze encoder in branch model
    # print("Freezing encoder in branch model...")
    # for param in model.x_dnn.parameters():
    #     param.requires_grad = False

    # # Freeze RNN in branch model 
    # print("Freezing rnn in branch model...")
    # for param in model.u_rnn.parameters():
    #     param.requires_grad = False

    # # Freeze decoder in branch model 
    # print("Freezing decoder in branch model...")
    # for param in model.u_dnn.parameters():
    #     param.requires_grad = False

    # # Freeze decoder in branch model 
    # print("Freezing nonlinearity in branch model...")
    # for param in model.output_NN.parameters():
    #     param.requires_grad = False

    # Freeze the pretrained trunk model
    print("Freezing trunk model...")
    for param in model.trunk_svd.parameters():
        param.requires_grad = False

    optimiser = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=wandb.config['lr'])

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

    header_msg = f"{'Epoch':>5} :: {'Total Loss (Train)':>16} :: {'Data Loss (Train)':>16} :: {'Orthogonal Loss (Train)':>16} :: " \
        f"{'Data Loss (Val)':>16} :: {'Data Loss (Test)':>16} :: {'Best (Val)':>16}"

    print(header_msg)
    print('=' * len(header_msg))

    # Evaluate initial loss
    model.eval()
    train_loss_total, train_loss_data = validate(train_dl,x_out_train.view(-1,1).to(device),x_in.view(-1,1).to(device),train_loss_fn, model, device,selected_indices.to(device))
    _, val_loss_data = validate(val_dl,x_out_train.view(-1,1).to(device),x_in.view(-1,1).to(device),val_loss_fn, model, device,selected_indices.to(device))
    _, test_loss_data = validate(test_dl,x_out_test.view(-1,1).to(device),x_in.view(-1,1).to(device),val_loss_fn, model,device)

    early_stop.step(val_loss_data)

    if get_loss(wandb.config["train_loss"]).__name__.endswith("orthogonal"):
        train_loss_ortho = train_loss_total-train_loss_data
    else:
        train_loss_ortho = float("nan")
    print(
            f"{0:>5d} :: {train_loss_total:>16e} :: {train_loss_data:>16e} :: {train_loss_ortho:>16e} :: " \
            f"{val_loss_data:>16e} :: {test_loss_data:>16e}  :: {early_stop.best_val_loss:>16e}"
    )
    start = time.time()

    ### Main training loop ###
    for epoch in range(wandb.config['n_epochs']):
        model.train()
        if epoch == wandb.config['unfreeze_epoch']:
            print("Unfreezing the pretrained model's layers for fine-tuning...")
            # Unfreeze the trunk model
            for param in model.trunk_svd.parameters():
                param.requires_grad = True
            optimiser = torch.optim.Adam(model.parameters(), lr=wandb.config['lr'])

        for example in train_dl:
            train_step(example,x_out_train.view(-1,1).to(device),x_in.view(-1,1).to(device),train_loss_fn, model, optimiser, device,selected_indices.to(device))

        model.eval()
        train_loss_total, train_loss_data = validate(train_dl,x_out_train.view(-1,1).to(device),x_in.view(-1,1).to(device),train_loss_fn, model, device,selected_indices.to(device))
        _, val_loss_data = validate(val_dl,x_out_train.view(-1,1).to(device),x_in.view(-1,1).to(device),val_loss_fn, model, device,selected_indices.to(device))
        _, test_loss_data = validate(test_dl,x_out_test.view(-1,1).to(device),x_in.view(-1,1).to(device),val_loss_fn, model, device)

        sched.step(val_loss_data)
        early_stop.step(val_loss_data)
        if get_loss(wandb.config["train_loss"]).__name__.endswith("orthogonal"):
            train_loss_ortho = train_loss_total-train_loss_data
        else:
            train_loss_ortho = float("nan")
        print(
            f"{epoch+1:>5d} :: {train_loss_total:>16e} :: {train_loss_data:>16e} :: {train_loss_ortho:>16e} :: " \
            f"{val_loss_data:>16e} :: {test_loss_data:>16e}  :: {early_stop.best_val_loss:>16e}"
        )
        if early_stop.best_model:
            torch.save(model.state_dict(), model_save_dir / "state_dict.pth")
            run.log_model(model_save_dir.as_posix(), name=model_name)

            run.summary["RHYME-XT/best_train"] = train_loss_total
            run.summary["RHYME-XT/best_val"] = val_loss_data
            run.summary["RHYME-XT/best_test"] = test_loss_data
            run.summary["RHYME-XT/best_epoch"] = epoch + 1

            ### Visualize trajectory in WB ###
            y,x0_feed,t_feed,u_feed,deltas_feed = trajectory(data['test'],trajectory_index=0,delta=test_data.delta) 
            y_pred, basis_functions = model(x0_feed.to(device), u_feed.to(device),x_out_test.view(-1,1).to(device),deltas_feed.to(device),x_in.view(-1,1).to(device))
            time_dim, space_dim = y.shape
            time_indices=[0, int(time_dim*0.25), int(time_dim*0.5),  int(time_dim*0.75), int(time_dim*0.95)]    # depends on n_samples
            space_indices=[0, int(space_dim*0.25), int(space_dim*0.5), int(space_dim*0.75),int(space_dim*0.95)] # depends on n_neurons
            fig = plot_space_time_trajectory(y,y_pred,time_indices=time_indices,space_indices=space_indices)
            wandb.log({"RHYME-XT/Test trajectory": wandb.Image(fig),"RHYME-XT/Best_epoch": epoch+1})

        wandb.log({
            'RHYME-XT/time': time.time() - start,
            'RHYME-XT/epoch': epoch + 1,
            'RHYME-XT/lr': sched.get_last_lr()[0],
            'RHYME-XT/train_loss_total': train_loss_total,
            'RHYME-XT/train_loss_data': train_loss_data,
            'RHYME-XT/val_loss': val_loss_data,
            'RHYME-XT/test_loss': test_loss_data,
        })

        if early_stop.early_stop:
            print(f"{epoch + 1:>5d} :: --- Early stop ---")
            break

    train_time = time.time() - start

    print(f"Training took {train_time:.2f} seconds.")


if __name__ == '__main__':
    main()
   

