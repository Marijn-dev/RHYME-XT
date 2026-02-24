from distutils.command import config
import torch
import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl

from RHYME_XT import RHYME_XT_Model, TrunkNet
from RHYME_XT.utils import pack_model_inputs, plot_2D_trajectories, plot_heatmap, plot_slider, save_GIF, plot_2D_fixedspace
from generate_data import make_trajectory_sampler

from argparse import ArgumentParser

import yaml
from pathlib import Path
import sys
from pprint import pprint
from time import time
from matplotlib.ticker import ScalarFormatter
import seaborn as sns

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

def relative_l2_error(y_pred, y_true):
    # L2 norm of the difference / L2 norm of the truth
    return np.linalg.norm(y_true - y_pred) / np.linalg.norm(y_true)

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

    trunk_model = TrunkNet(**metadata["trunk_args"])
    model = RHYME_XT_Model(**metadata["args"],trunk_model=trunk_model)
    model.load_state_dict(state_dict)
    model.eval()

    sampler = make_trajectory_sampler(metadata["data_settings"])
    sampler.reset_rngs()
    delta = sampler._delta

    
    time_horizon = metadata["data_args"]["time_horizon"]
    time_horizon = 250
    n_samples = int(200 * time_horizon / 50)

   
    save = False
    if save == True:
        x0_data = []
        t_data = []
        y_data = []
        u_data = []
        for i in range(50):
            print(i)
            x0, t, y, u,y_full = sampler.get_example   (time_horizon=time_horizon, n_samples=n_samples)
            idx = np.linspace(0, y.shape[1] - 1, 100, dtype=int)
            x0_data.append(x0)
            t_data.append(t)
            y_data.append(y)
            u_data.append(u)

        x0_array = np.array(x0_data)
        t_array  = np.array(t_data)
        y_array  = np.array(y_data)
        u_array  = np.array(u_data)

        # # Save to a compressed npz file
        # np.savez_compressed(
        #     'amari_T250.npz', 
        #     x0=x0_array, 
        #     t=t_array, 
        #     y=y_array, 
        #     u=u_array
        # )

    
    # locations_input = torch.tensor(sampler._dyn.locations,dtype=torch.get_default_dtype()) * metadata["location_scaling"]
    save_results = False
    if save_results == True:
        data = np.load('amari_T50.npz')
        y_pred_results = []
        y_results = []
        t_results = []
        mse_error = []
        relative_error = []
        t_data = []
        for i in range(data['x0'].shape[0]):
            print(i)
            t = data['t'][i]
            x0 = data['x0'][i]
            y = data['y'][i]
            u = data['u'][i]
            idx = np.linspace(0, y.shape[1] - 1, 100, dtype=int)
            locations_output = torch.tensor(sampler._dyn.locations,dtype=torch.get_default_dtype()) * metadata["location_scaling"]
            locations_output = locations_output[idx]
            locations_input = locations_output
            x0 = x0[idx]
            y = y[:,idx]
            u = u[:,idx]

            x0_feed, t_feed, u_feed, deltas_feed = pack_model_inputs(
                x0, t, u, delta)

            with torch.no_grad():
                y_pred, basis_functions = model(x0_feed, u_feed, locations_output,deltas_feed,locations_input)
            y_pred = y_pred.cpu().numpy()
            y_pred = np.flip(y_pred, 0)
            y_pred_results.append(y_pred)
            y_results.append(y)
            t_data.append(t_feed)

            L2_error = np.square(y - y_pred)
            print("L2 error",np.mean(L2_error))
            mse_error.append(np.mean(L2_error))
            error = relative_l2_error(y_pred, y)
            print(f"Relative L2 Error: {error * 100:.2f}%")
            relative_error.append(error*100)

        y_pred_results = np.array(y_pred_results)
        y_results  = np.array(y_results)
        mse_error  = np.array(mse_error)
        relative_error  = np.array(relative_error)
        t_data = np.array(t_data)

        np.savez_compressed(
            'amari_T50_results.npz', 
            y_pred=y_pred_results, 
            y=y_results, 
            mse=mse_error, 
            relative_error=relative_error,
            t = t_data
        )
    analyze_results = True
    if analyze_results == True:
        results_T250 = np.load("amari_T250_results.npz")
        error_T250 = results_T250['relative_error']
        results_T100 = np.load("amari_T100_results.npz")
        error_T100 = results_T100['relative_error']
        results_T50 = np.load("amari_T50_results.npz")
        error_T50 = results_T50['relative_error']
        data = [error_T50, error_T100, error_T250]
        labels = ["50", "100", "250"]
        print(np.mean(error_T50))
        print(np.argmax(error_T50))
        print(error_T50)
        print(np.mean(error_T100))
        print(np.mean(error_T250))
        # # Create the box plot
        sns.boxplot(data=data)

        # Set the labels on the x-axis
        plt.xticks(ticks=[0, 1, 2], labels=labels)

        # Optional styling
        plt.xlabel("Time horizon T")
        plt.ylabel("Relative $L^2$ Error [%]")
        plt.grid(True)
        plt.savefig('amari_timehorizons_boxplot.pdf',dpi=300)

        y = results_T50['y'][49]
        y_pred = torch.tensor(results_T50['y_pred'][49])
        t_feed = torch.tensor(results_T50['t'][49])
        # plt.show()
        # Save the figure
    # Save to a compressed npz file
    # save_results = True
    # if save_results == True:
    #     np.savez('results_rhyme-xt.npz', 
    #                 y_pred=y_pred, 
    #                 t=t_feed, 
    #                 y=y) 
        
    t_feed = torch.flip(t_feed,dims=[0])
    # 2D Plot of slices in the trajectory
    plot_2D_trajectories(
    y, [y_pred], torch.flip(t_feed,dims=[0]),
    labels=['Ground-truth', 'RHYME-XT'],
    time_indices=[int(y.shape[0]*0.25), int(y.shape[0]*0.5), int(y.shape[0]*0.75)],
    space_indices=[int(y.shape[1]*0.25), int(y.shape[1]*0.5), int(y.shape[1]*0.75)])

    # Heatmap plot
    plot_heatmap(
    y, [y_pred], None,t_feed,
    labels=['Ground-truth', 'RHYME-XT'])

    # Slider plot
    plot_slider(y, [y_pred], t_feed, labels=['Ground-truth', 'RHYME-XT'])

    # # Save GIF
    # save_GIF(y,[y_pred],t_feed,labels=['Ground-truth', 'RHYME-XT'])




if __name__ == '__main__':
    main()