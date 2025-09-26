import torch
import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl

from flumen import RHYME_XT, TrunkNet
from flumen.utils import pack_model_inputs, plot_2D_trajectories, plot_heatmap, plot_slider, save_GIF
from generate_data import make_trajectory_sampler

from argparse import ArgumentParser

import yaml
from pathlib import Path
import sys
from pprint import pprint
from time import time


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
        print(model_run.summary)
    else:
        model_path = Path(args.path)

    with open(model_path / "state_dict.pth", 'rb') as f:
        state_dict = torch.load(f, weights_only=True)
    with open(model_path / "metadata.yaml", 'r') as f:
        metadata: dict = yaml.load(f, Loader=yaml.FullLoader)

    pprint(metadata)

    args = metadata["args"].copy()
    args.pop("regular", None)
    args.pop("trunk_modes", None)
    args.pop("use_conv_encoder", None)
    args["trunk_modes_svd"] = 100
    args["trunk_modes_extra"] = 0
    trunk_model = TrunkNet(in_size=256,out_size=100,hidden_size=[100,100,100,100],use_batch_norm=False)
    args["trunk_model"] = trunk_model
    model = RHYME_XT(**args)
    # model = RHYME_XT(**metadata["args"])

    model.load_state_dict(state_dict)
    model.eval()

    sampler = make_trajectory_sampler(metadata["data_settings"])
    sampler.reset_rngs()
    delta = sampler._delta

    time_horizon = metadata["data_args"]["time_horizon"]

    time_integrate = time()
    x0, t, y, u,y_full = sampler.get_example(time_horizon=time_horizon,
                                    n_samples=int(1 +
                                                    1000))
    time_integrate = time() - time_integrate
    locations_output = torch.tensor(sampler._dyn.locations,dtype=torch.get_default_dtype())
    locations_input = locations_output.clone()

    time_predict = time()

    x0_feed, t_feed, u_feed, deltas_feed = pack_model_inputs(
        x0, t, u, delta)

    with torch.no_grad():
        y_pred, basis_functions = model(x0_feed, u_feed, locations_output,deltas_feed,locations_input)
    y_pred = y_pred.cpu().numpy()
    y_pred = np.flip(y_pred, 0)
    time_predict = time() - time_predict
    
    print(f"Timings: {time_integrate}, {time_predict}")

    y = y[:, tuple(bool(v) for v in sampler._dyn.mask)]
    L1_error = np.abs(y - y_pred)
    L2_error = np.square(y - y_pred)
    print("MAE error:",np.mean(L1_error))
    print("MSE error:",np.mean(L2_error))

    # 2D Plot of slices in the trajectory
    plot_2D_trajectories(
    y, [y_pred], t_feed,
    labels=['Ground-truth', 'RHYME-XT'],
    time_indices=[int(y.shape[0]*0.25), int(y.shape[0]*0.5), int(y.shape[0]*0.95)],
    space_indices=[int(y.shape[1]*0.25), int(y.shape[1]*0.5), int(y.shape[1]*0.95)])

    # Heatmap plot
    plot_heatmap(
    y, [y_pred], t_feed,
    labels=['Ground-truth', 'RHYME-XT'])

    # Slider plot
    plot_slider(y, [y_pred], t_feed, labels=['Ground-truth', 'RHYME-XT'])

    # Save GIF
    save_GIF(y,[y_pred],t_feed,labels=['Ground-truth', 'RHYME-XT'])


if __name__ == '__main__':
    main()
