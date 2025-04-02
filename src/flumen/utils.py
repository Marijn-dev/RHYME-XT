import torch
import numpy as np
from argparse import ArgumentParser, ArgumentTypeError
import pickle 
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button

def print_gpu_info():
    if torch.cuda.is_available():
        n_gpus = torch.cuda.device_count()
        print(f"CUDA is available, {n_gpus} devices can be used.")
        current_dev = torch.cuda.current_device()

        for id in range(n_gpus):
            msg = f"Device {id}: {torch.cuda.get_device_name(id)}"

            if id == current_dev:
                msg += " [Current]"

            print(msg)


def get_arg_parser():
    ap = ArgumentParser()

    model_args = ap.add_argument_group("Model hyperparameters")
    opt_args = ap.add_argument_group("Optimisation hyperparameters")

    model_args.add_argument('--control_rnn_size',
                            type=positive_int,
                            help="Size of the RNN hidden state",
                            required=True)

    model_args.add_argument('--control_rnn_depth',
                            type=positive_int,
                            help="Depth of the RNN",
                            default=1)

    model_args.add_argument('--encoder_size',
                            type=positive_int,
                            help="Size (multiplier) of the encoder layers",
                            required=True)

    model_args.add_argument('--encoder_depth',
                            type=positive_int,
                            help="Depth of the encoder",
                            required=True)

    model_args.add_argument('--decoder_size',
                            type=positive_int,
                            help="Size (multiplier) of the decoder layers",
                            required=True)

    model_args.add_argument('--decoder_depth',
                            type=positive_int,
                            help="Depth of the decoder",
                            required=True)

    opt_args.add_argument('--batch_size',
                          type=positive_int,
                          help="Batch size for training and validation",
                          required=True)

    opt_args.add_argument('--lr',
                          type=positive_float,
                          help="Initial learning rate",
                          required=True)

    opt_args.add_argument('--n_epochs',
                          type=positive_int,
                          help="Max number of epochs",
                          required=True)

    opt_args.add_argument('--es_patience',
                          type=positive_int,
                          help="Early stopping -- patience (epochs)",
                          required=True)

    opt_args.add_argument('--es_delta',
                          type=nonnegative_float,
                          help="Early stopping -- minimum loss change",
                          required=True)

    opt_args.add_argument('--sched_patience',
                          type=positive_int,
                          help="LR Scheduler -- Patience epochs",
                          required=True)

    opt_args.add_argument('--sched_cooldown',
                          type=positive_int,
                          help="LR scheduler -- Cooldown epochs",
                          default=0)

    opt_args.add_argument('--sched_factor',
                          type=positive_int,
                          help="LR Scheduler -- Reduction factor",
                          required=True)

    ap.add_argument('--use_batch_norm',
                    action='store_true',
                    help="Use batch normalisation in encoder and decoder.")

    ap.add_argument(
        '--max_seq_len',
        type=max_seq_len,
        help="Maximum length of the RNN sequences "
        "(for semigroup augmentation). No augmentation if equal to -1.",
        default=-1)

    ap.add_argument('--samples_per_state',
                    type=positive_int,
                    help="Number of samples per state measurement "
                    "(if using semigroup augmentation)",
                    default=1)

    ap.add_argument(
        '--whiten_data',
        action='store_true',
        help='Apply whitening normalization to the data before training.')

    ap.add_argument('--experiment_id',
                    type=str,
                    help="Human-readable experiment identifier. "
                    "Nothing is written to disk if this is not provided.",
                    default=None)

    ap.add_argument('--write_dir',
                    type=str,
                    help="Directory to which the model will be written.",
                    default='./outputs')

    return ap


def positive_int(value):
    value = int(value)

    if value <= 0:
        raise ArgumentTypeError(f"{value} is not a positive integer")

    return value


def positive_float(value):
    value = float(value)

    if value <= 0:
        raise ArgumentTypeError(f"{value} is not a positive float")

    return value


def nonnegative_float(value):
    value = float(value)

    if value < 0:
        raise ArgumentTypeError(f"{value} is not a nonnegative float")

    return value


def max_seq_len(value):
    value = int(value)
    if value <= 0 and value != -1:
        raise ArgumentTypeError("max_seq_len must be a positive integer or -1")

    return value


def pack_model_inputs(x0, t, u, delta):
    t = torch.Tensor(t.reshape((-1, 1))).flip(0)
    x0 = torch.Tensor(x0.reshape((1, -1))).repeat(t.shape[0], 1)
    rnn_inputs = torch.empty((t.shape[0], u.shape[0], u.shape[1] + 1))
    lengths = torch.empty((t.shape[0], ), dtype=torch.long)

    for idx, (t_, u_) in enumerate(zip(t, rnn_inputs)):
        control_seq = torch.from_numpy(u)
        deltas = torch.ones((u.shape[0], 1))

        seq_len = 1 + int(np.floor(t_ / delta))
        lengths[idx] = seq_len
        deltas[seq_len - 1] = ((t_ - delta * (seq_len - 1)) / delta).item()
        deltas[seq_len:] = 0.

        u_[:] = torch.hstack((control_seq, deltas))

    u_packed = torch.nn.utils.rnn.pack_padded_sequence(rnn_inputs,
                                                       lengths,
                                                       batch_first=True,
                                                       enforce_sorted=True)

    return x0, t, u_packed, rnn_inputs[:, :lengths[0], -1].unsqueeze(-1)

def plot_slider_1d(data):
    """
     Creates an interactive plot with a slider to visualize how activity and inputs change in time.
    """
    train = data['train']
    x = data['Locations'].numpy()
    x_lim = x[-1]
    
    trajectory = 1
    current_trajectory = 0

    ## assign data corresponding to trajectory
    for (x0, x0_n, t, y, y_n, u) in train:
        current_trajectory += 1
        dt = t[1][0].numpy()
        activity = y.numpy()
        inputs = u.numpy()
        if current_trajectory == trajectory:
            break

    upper_lim_y = max([activity.max(), inputs.max()])
    lower_lim_y = min([activity.min(), inputs.min()])

    fig, ax = plt.subplots(figsize=(6, 4))
    plt.subplots_adjust(bottom=0.25)  # Adjust the bottom margin to make space for the slider and button

    line_activity, = ax.plot(x, activity[0, :], label='u(x)')

    
    line_input, = ax.plot(x, inputs[0, :], label='Input(x)', linestyle='dashed')

    ax.legend()
    ax.set_ylim(lower_lim_y, upper_lim_y)
    ax.set_xlim(0, x_lim)
    plt.xlabel('x')

    ax_slider = plt.axes([0.25, 0.1, 0.65, 0.03])  # Define the slider's position [left, bottom, width, height]
    slider = Slider(ax_slider, '', 0, activity.shape[0] - 1, valinit=0, valstep=1)
    slider.valtext.set_visible(False)  # hide matplotlib slider values

    ax_reset = plt.axes([0.8, 0.02, 0.1, 0.04])  # Define the reset button's position [left, bottom, width, height]
    reset_button = Button(ax_reset, 'Reset')

    time_label = plt.text(0.5, 0.05, f'Time Step: {slider.val * dt:.2f}', transform=fig.transFigure, ha='center')

    def update(val):
        time_step = int(slider.val)
        line_activity.set_ydata(activity[time_step, :])

        line_input.set_ydata(inputs[time_step, :])

        time_label.set_text(f'Time : {time_step * dt:.2f}')
        fig.canvas.draw_idle()

    def reset(event):
        slider.set_val(0)

    slider.on_changed(update)
    reset_button.on_clicked(reset)

    plt.show()

def plot_space_time_3d(data):
    """
    Plot a 3D surface of the field activity over space and time.
    """

    train = data['train']
    x = data['Locations'].numpy()
    x_lim = x[-1]
    dx = x[1] - x[0]

    trajectory = 1
    current_trajectory = 0

    ## assign data corresponding to trajectory
    for (x0, x0_n, t, y, y_n, u) in train:
        current_trajectory += 1
        dt = t[1][0].numpy()
        t = t.numpy()
        activity = y.numpy()
        inputs = u.numpy()
        if current_trajectory == trajectory:
            break
    
    upper_lim = activity.max()
    lower_lim = activity.min()

    x_mesh, t_mesh = np.meshgrid(x, t)
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Plot the surface
    surf = ax.plot_surface(t_mesh, x_mesh, activity, cmap=plt.get_cmap('plasma'),
                           linewidth=0, antialiased=False)

    # Remove the gray shading
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False

    ax.set_box_aspect([2, 1, 1])

    fig.colorbar(surf, shrink=0.4, aspect=10, pad=0.2)

    ax.zaxis.set_rotate_label(False)

    ax.set_xlabel('t', linespacing=3.2)
    ax.set_ylabel('x', linespacing=3.1)
    ax.set_zlabel('u(x,t)', linespacing=3.4, rotation=0)

    ax.zaxis.labelpad = 10
    ax.set_zlim(lower_lim, upper_lim)

    ax.set_yticks(np.arange(0, x_lim + dx, 2))

    plt.show()

def plot_amari(data_path, plot):
    '''
    visualizes particular plot given RawTrajectoryDataset
    '''

    data_path = Path(data_path)
    with data_path.open('rb') as f:
        data = pickle.load(f)

    if plot == "slider_1d":
        plot_slider_1d(data)

    if plot == "space_time_3d":
        plot_space_time_3d(data)

if __name__ == "__main__":
    plot_amari("data/amari_test_data.pkl", "slider_1d")