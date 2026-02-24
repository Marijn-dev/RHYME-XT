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


from matplotlib.ticker import ScalarFormatter

def relative_l2_error(y_pred, y_true):
        # L2 norm of the difference / L2 norm of the truth
        return np.linalg.norm(y_true - y_pred) / np.linalg.norm(y_true)


def main():

    data_don = np.load('results_don_short.npz')
    data_rhyme = np.load('results_rhyme-xt.npz')
    # Extract the variables back to their names
    y = data_don['y']
    y_rhyme = data_rhyme['y']
    y_pred_don = data_don['y_pred']
    y_pred_rhyme = data_rhyme['y_pred']
    t_don = data_don['t']
    t_feed = data_rhyme['t_feed']

    L2_error_don = np.square(y - y_pred_don)
    print("L2 error",np.mean(L2_error_don))

    L2_error_don_relative = 100*relative_l2_error(y_pred_don, y)
    # print(f"Relative L2 Error: {error * 100:.2f}%")
    
    L2_error_rhyme = np.square(y - y_pred_rhyme)

    L2_error_rhyme_relative = 100*relative_l2_error(y_pred_rhyme, y)

    # Heatmap plot
    plot_heatmap(
    y_rhyme, [y_pred_rhyme, y_pred_don], [L2_error_rhyme_relative, L2_error_don_relative], t_don,
    labels=['Ground truth', 'RHYME-XT', "DeepONet"])

    # Slider plot
    # plot_slider(y, [y_pred_rhyme,y_pred_don], t_feed, labels=['Ground-truth', 'RHYME-XT', 'DeepONet'])

    # # Save GIF
    # save_GIF(y,[y_pred_rhyme, y_pred_don],t_feed,labels=['Ground-truth', 'RHYME-XT', 'DeepONet'])

if __name__ == '__main__':
    main()