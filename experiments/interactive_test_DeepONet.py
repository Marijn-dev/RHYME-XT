from distutils.command import config
import torch
import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl

from RHYME_XT import RHYME_XT_Model, TrunkNet, DeepONet_Model
from RHYME_XT.utils import pack_model_inputs, plot_2D_trajectories, plot_heatmap, plot_slider, save_GIF, plot_2D_fixedspace, prepare_model_inputs_DeepONet
from generate_data import make_trajectory_sampler

from argparse import ArgumentParser

import yaml
from pathlib import Path
import sys
from pprint import pprint
from time import time
from matplotlib.ticker import ScalarFormatter

def parse_args():
    ap = ArgumentParser()
    ap.add_argument(
        'path',
        type=str,
        help="Path to .pth file "
        "(or, if run with --wandb, path to a Weights & Biases artifact)")
    ap.add_argument('--print_info',
                    action='store_true',
                    help="Print training metadata and quit")
    ap.add_argument('--continuous_state', action='store_true')
    ap.add_argument('--wandb', action='store_true')

    return ap.parse_args()


def main():
    args = parse_args()

    if args.wandb:
        import wandb
        api = wandb.Api()
        model_artifact = api.artifact(args.path)
        model_path = Path(model_artifact.download())

        model_run = model_artifact.logged_by()
        # print(model_run.summary)
    else:
        model_path = Path(args.path)

    with open(model_path / "state_dict.pth", 'rb') as f:
        state_dict = torch.load(f, weights_only=True)
    with open(model_path / "metadata.yaml", 'r') as f:
        metadata: dict = yaml.load(f, Loader=yaml.FullLoader)

   
    model = DeepONet_Model(**metadata["args"])
    model.load_state_dict(state_dict)
    model.eval()

    sampler = make_trajectory_sampler(metadata["data_settings"])
    sampler.reset_rngs()
    delta = sampler._delta

    
    time_horizon = metadata["data_args"]["time_horizon"]
    n_samples = 200

    save = False
    if save == True:
        x0, t, y, u,y_full = sampler.get_example   (time_horizon=time_horizon, n_samples=n_samples)
        # Save to a single compressed file
        np.savez('test_data.npz', 
                x0=x0, 
                t=t, 
                y=y, 
                u=u)
    else: 
        data = np.load('test_data.npz')
        # Extract the variables back to their names
        x0 = data['x0']
        t = data['t']
        y = data['y']
        u = data['u']
    
    # time_integrate = time() - time_integrate
    idx = np.linspace(0, y.shape[1] - 1, 100, dtype=int)
    x0 = x0[idx]
    y = y[:,idx]
    u = u[:,idx]
    locations_output = torch.tensor(sampler._dyn.locations,dtype=torch.get_default_dtype()) * metadata["location_scaling"]
    locations_output = locations_output[idx]

    x0_feed, t_feed, u_feed = prepare_model_inputs_DeepONet(x0,t,u,delta,time_horizon,n_samples)
    x0 = x0_feed
    y_pred_list = []
    with torch.no_grad():
        for i in range(0,u_feed.shape[0]):
            y_pred = model(x0_feed, u_feed[i], t_feed[i],locations_output)
            x0_feed = y_pred[:,-1,:]
            y_pred_list.append(y_pred)
            # print(y_pred.shape)

    y_pred = torch.stack(y_pred_list) 
    y_pred = y_pred.transpose(0,1)
    y_pred = y_pred.view(n_samples,100)
    y_pred = torch.cat([x0, y_pred], dim=0)
    y_pred = y_pred.cpu().numpy()

    def relative_l2_error(y_pred, y_true):
        # L2 norm of the difference / L2 norm of the truth
        return np.linalg.norm(y_true - y_pred) / np.linalg.norm(y_true)

    L2_error = np.square(y - y_pred)
    print("L2 error",np.mean(L2_error))

    error = relative_l2_error(y_pred, y)
    print(f"Relative L2 Error: {error * 100:.2f}%")


    save_results = True
    if save_results == True:
        np.savez('results_don_short.npz', 
                    y_pred=y_pred, 
                    t=t_feed, 
                    y=y) 
        
    # _, t_feed, _, _ = pack_model_inputs(
    #     x0, t, u, delta)
    # # t_feed = torch.flip(t_feed,dims=[0])
    # # 2D Plot of slices in the trajectory
    # plot_2D_trajectories(
    # y, [y_pred], torch.flip(t_feed,dims=[0]),
    # labels=['Ground-truth', 'RHYME-XT'],
    # time_indices=[int(y.shape[0]*0.25), int(y.shape[0]*0.5), int(y.shape[0]*0.95)],
    # space_indices=[int(y.shape[1]*0.25), int(y.shape[1]*0.5), int(y.shape[1]*0.95)])

    # # Heatmap plot
    # plot_heatmap(
    # y, [y_pred], t_feed,
    # labels=['Ground-truth', 'RHYME-XT'])

    # # Slider plot
    # plot_slider(y, [y_pred], t_feed, labels=['Ground-truth', 'RHYME-XT'])

    # Save GIF
    # save_GIF(y,[y_pred],t_feed,labels=['Ground-truth', 'RHYME-XT'])




if __name__ == '__main__':
    main()