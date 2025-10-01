from semble import TrajectorySampler
from semble.dynamics import get_dynamics
from semble.sequence_generators import get_sequence_generator
from semble.initial_state import get_initial_state_generator

from argparse import ArgumentParser, ArgumentTypeError
from RHYME_XT import RawTrajectoryDataset
import torch
import numpy as np

def percentage(value):
    value = int(value)

    if not (0 <= value <= 100):
        raise ArgumentTypeError(f"{value} is not a valid percentage")

    return value


def parse_args():
    ap = ArgumentParser()

    ap.add_argument(
        'settings',
        type=str,
        help=
        "Path to a YAML file containing the parameters defining the trajectory sampler."
    )

    ap.add_argument('output_name',
                    type=str,
                    help="File name for writing the data to disk.")

    ap.add_argument('--time_horizon',
                    type=float,
                    help="Time horizon",
                    default=50.)

    ap.add_argument('--n_trajectories',
                    type=int,
                    help="Number of trajectories to sample",
                    default=100)

    ap.add_argument('--n_samples',
                    type=int,
                    help="Number of state samples per trajectory",
                    default=50)

    ap.add_argument('--noise_std',
                    type=float,
                    help="Standard deviation of measurement noise",
                    default=0.0)

    ap.add_argument('--noise_seed',
                    type=int,
                    help="Measurement noise seed",
                    default=None)

    ap.add_argument(
        '--data_split',
        nargs=2,
        type=percentage,
        help="Percentage of data used for validation and test sets",
        default=[20, 10])

    ap.add_argument(
        '--noise_std_svd',
        type=float,
        help="Standard deviation to add in SVD calculation (noisy training experiment)",
        default=0.0)
    
    ap.add_argument(
        '--num_locations_svd',
        type=float,
        help="Ratio of number of spatial locations to use in SVD calculation (spatial interpolation experiment)",
        default=1.0)

    return ap.parse_args()

def generate(args, trajectory_sampler: TrajectorySampler, postprocess=[]):
    if args.data_split[0] + args.data_split[1] >= 100:
        raise Exception("Invalid data split.")
    
    n_val = int(args.n_trajectories * (args.data_split[0] / 100.))
    n_test = int(args.n_trajectories * (args.data_split[1] / 100.))
    n_train = args.n_trajectories - n_val - n_test

    def get_example():
        x0, t, y, u, y_full = trajectory_sampler.get_example(args.time_horizon,
                                                     args.n_samples)
        
        return {
            "init_state": x0,
            "time": t,
            "state": y,
            "control": u,
            "full_state": y_full
        }
    train_data = []
    for i in range(n_train):
        if i % 100 == 0:
            print(f"Generating training dataset example {i+1}/{n_train}")
        example = get_example()
        train_data.append(example)
    trajectory_sampler.reset_rngs()

    val_data = [get_example() for _ in range(n_val)]
    trajectory_sampler.reset_rngs()

    test_data = [get_example() for _ in range(n_test)]

    states_combined = torch.cat([
    torch.tensor(d["full_state"], dtype=torch.get_default_dtype())
    for d in train_data
    ], dim=0)
    
    selected_indices = torch.linspace(0, states_combined.shape[1]-1, steps=int(states_combined.shape[1]*args.num_locations_svd)).long()
    PHI, _, _ = torch.linalg.svd(states_combined[:, selected_indices].T + args.noise_std_svd * torch.randn_like(states_combined[:, selected_indices].T),full_matrices=False)
    
    train_data = RawTrajectoryDataset(train_data,
                                      *trajectory_sampler.dims(),
                                      delta=trajectory_sampler._delta,
                                      output_mask=trajectory_sampler._dyn.mask,
                                      input_mask=trajectory_sampler._dyn.input_mask,
                                      noise_std=args.noise_std)

    val_data = RawTrajectoryDataset(val_data,
                                    *trajectory_sampler.dims(),
                                    delta=trajectory_sampler._delta,
                                    output_mask=trajectory_sampler._dyn.mask,
                                    input_mask=trajectory_sampler._dyn.input_mask,
                                    noise_std=args.noise_std)

    test_data = RawTrajectoryDataset(test_data,
                                     *trajectory_sampler.dims(),
                                     delta=trajectory_sampler._delta,
                                     output_mask=trajectory_sampler._dyn.mask,
                                     input_mask=trajectory_sampler._dyn.input_mask,
                                     noise_std=args.noise_std)

    for d in (train_data, val_data, test_data):
        for p in postprocess:
            p(d)

    return train_data, val_data, test_data, PHI


def make_trajectory_sampler(settings):
    dynamics = get_dynamics(settings["dynamics"]["name"],
                            settings["dynamics"]["args"])


    sequence_generator = get_sequence_generator(
        settings["sequence_generator"]["name"],
        settings["sequence_generator"]["args"])

    if "initial_state_generator" in settings:
        init_state_gen = get_initial_state_generator(
            settings["initial_state_generator"]["name"],
            settings["initial_state_generator"]["args"])
    else:
        init_state_gen = None

    sampler = TrajectorySampler(dynamics=dynamics,
                                control_delta=settings["control_delta"],
                                control_generator=sequence_generator,
                                method=settings.get("method"),
                                initial_state_generator=init_state_gen)

    return sampler
