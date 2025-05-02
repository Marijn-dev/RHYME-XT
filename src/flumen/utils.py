import torch
import numpy as np
from argparse import ArgumentParser, ArgumentTypeError
import pickle 
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
import torch.nn.functional as F

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
    x0 = torch.Tensor(x0.reshape((1, *x0.shape))).repeat(t.shape[0], *([1] * len(x0.shape)))
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

def trajectory(data,delta):
    trajectory_index = 0
    x0, init_state_noise, t, y, state_noise, control_seq = data[trajectory_index]
    t = t.reshape(-1, 1).flip(0)  # Flip in time
    x0 = x0.reshape(1, *x0.shape).repeat(t.shape[0], *([1] * len(x0.shape)))
    rnn_inputs = torch.empty((t.shape[0], control_seq.shape[0], control_seq.shape[1] + 1))
    lengths = torch.empty((t.shape[0], ), dtype=torch.long)
    
    for idx, (t_, u_) in enumerate(zip(t, rnn_inputs)):
        deltas = torch.ones((control_seq.shape[0], 1))
        
        seq_len = 1 + int(np.floor(t_ / delta))
        lengths[idx] = seq_len
        deltas[seq_len - 1] = ((t_ - delta * (seq_len - 1)) / delta).item()
        deltas[seq_len:] = 0.
        u_[:] = torch.hstack((control_seq, deltas))
   
    u_packed = torch.nn.utils.rnn.pack_padded_sequence(rnn_inputs,
                                                   lengths,
                                                   batch_first=True,
                                                   enforce_sorted=True)
    return y,x0, t, u_packed, rnn_inputs[:, :lengths[0], -1].unsqueeze(-1)

def plot_slider_1d(t,y,inputs,locations):
    """
     Creates an interactive plot with a slider to visualize how activity and inputs change in time.
    """
    x = locations
    x_lim = locations[-1]
    
    upper_lim_y = max([y.max(), inputs.max()])
    lower_lim_y = min([y.min(), inputs.min()])

    dt = t[1]-t[0]
    fig, ax = plt.subplots(figsize=(6, 4))
    plt.subplots_adjust(bottom=0.25)  # Adjust the bottom margin to make space for the slider and button

    line_activity, = ax.plot(x, y[0, :], label='u(x)')

    
    line_input, = ax.plot(x, inputs[0, :], label='I(x)', linestyle='dashed')

    ax.legend()
    ax.set_ylim(lower_lim_y, upper_lim_y)
    ax.set_xlim(0, x_lim)
    plt.xlabel('x')

    ax_slider = plt.axes([0.25, 0.1, 0.65, 0.03])  # Define the slider's position [left, bottom, width, height]
    slider = Slider(ax_slider, '', 0, y.shape[0] - 1, valinit=0, valstep=1)
    slider.valtext.set_visible(False)  # hide matplotlib slider values

    ax_reset = plt.axes([0.8, 0.02, 0.1, 0.04])  # Define the reset button's position [left, bottom, width, height]
    reset_button = Button(ax_reset, 'Reset')

    time_label = plt.text(0.5, 0.05, f'Time Step: {slider.val * dt[0]:.2f}', transform=fig.transFigure, ha='center')

    def update(val):
        time_step = int(slider.val)
        line_activity.set_ydata(y[time_step, :])

        line_input.set_ydata(inputs[time_step, :])

        time_label.set_text(f'Time : {time_step * dt[0]:.2f}')
        fig.canvas.draw_idle()

    def reset(event):
        slider.set_val(0)

    slider.on_changed(update)
    reset_button.on_clicked(reset)

    plt.show()

def plot_space_time_flat_trajectory(y, y_pred):
    '''returns a flat space-time image of the field activity for both y and y_pred'''
    # Convert to numpy for plotting
    y_pred = torch.flip(y_pred, dims=[0])
    y_np = y.detach().cpu().numpy()
    y_np = np.transpose(y_np)
    y_pred_np = y_pred.detach().cpu().numpy()
    y_pred_np = np.transpose(y_pred_np)
    
    # Plot side-by-side heatmaps
    fig, axs = plt.subplots(1, 2, figsize=(6, 3),dpi=80)

    im0 = axs[0].imshow(y_np, aspect='auto', cmap='viridis',vmin=0, vmax=1)
    axs[0].set_title("Ground Truth (y)")
    plt.colorbar(im0, ax=axs[0])

    im1 = axs[1].imshow(y_pred_np, aspect='auto', cmap='viridis')
    axs[1].set_title("Prediction (y_pred)")
    plt.colorbar(im1, ax=axs[1])

    plt.tight_layout()
    return fig

def plot_space_time_flat_trajectory_V2(y, y_pred, time_indices=[0, 100, 400, 600, 800]):
    '''Returns heatmaps and 1D neuron activity plots at selected time points.'''
    # Convert to numpy
    y_pred = torch.flip(y_pred, dims=[0])
    y_np = y.detach().cpu().numpy().T  # shape: (neurons, time)
    y_pred_np = y_pred.detach().cpu().numpy().T
    l1_loss = F.l1_loss(y.cpu(), y_pred.cpu()).item()

    num_times = len(time_indices)
    fig = plt.figure(figsize=(3 * max(4, num_times), 6), dpi=100)
    gs = fig.add_gridspec(2, max(4, num_times), height_ratios=[2, 1])

    # Heatmaps
    ax0 = fig.add_subplot(gs[0, 0:2])
    im0 = ax0.imshow(y_np, aspect='auto', cmap='viridis', vmin=0, vmax=1)
    ax0.set_title("Ground Truth (y)")
    plt.colorbar(im0, ax=ax0)
    for t in time_indices:
        ax0.axvline(x=t, color='red', linestyle='--')

    ax1 = fig.add_subplot(gs[0, 2:4])
    im1 = ax1.imshow(y_pred_np, aspect='auto', cmap='viridis', vmin=0, vmax=1)
    ax1.set_title("Prediction (y_pred)")
    plt.colorbar(im1, ax=ax1)
    for t in time_indices:
        ax1.axvline(x=t, color='red', linestyle='--')

    # Individual line plots for each selected time step
    for i, t in enumerate(time_indices):
        ax = fig.add_subplot(gs[1, i])
        ax.plot(y_np[:, t], label=r"$u(x)$", linestyle='--',color='b')
        ax.plot(y_pred_np[:, t], label=r"$u{\text{pred}}(x)$", linestyle='-',color='r')
        ax.set_title(f"t={t}")
        ax.set_xlabel("Neuron Index")
        ax.set_ylabel("Activation")
        ax.legend()
        ax.grid(True)

    # Overall title
    fig.suptitle(f"L1 Loss over trajectory: {l1_loss:.4f}", fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    return fig

def plot_space_time_flat(t, y, u, locations):
    """
    Plots a flat space-time image of the field activity for both y and u.
    """
    x = locations
    x_lim = locations[-1]

    x_range = [0, x_lim]
    t_range = [0.0, t[-1]]

    # Limits for color scaling
    upper_lim_y = y.max()
    lower_lim_y = y.min()
    
    upper_lim_u = u.max()
    lower_lim_u = u.min()

    fig, axes = plt.subplots(1, 2, figsize=(6, 2))  # Two subplots side by side

    # First subplot: y
    pic1 = axes[0].imshow(np.transpose(y), cmap='plasma', vmin=lower_lim_y, vmax=upper_lim_y,
                           extent=[t_range[0], t_range[1], x_range[0], x_range[1]],
                           interpolation='nearest', origin='lower', aspect='auto')
    axes[0].set_xlabel('t')
    axes[0].set_ylabel('x', rotation=0)
    axes[0].set_title('v(x,t)')
    fig.colorbar(pic1, ax=axes[0])  # Colorbar for y

    # Second subplot: u
    pic2 = axes[1].imshow(np.transpose(u), cmap='plasma', vmin=lower_lim_u, vmax=upper_lim_u,
                           extent=[t_range[0], t_range[1], x_range[0], x_range[1]],
                           interpolation='nearest', origin='lower', aspect='auto')
    axes[1].set_xlabel('t')
    axes[1].set_ylabel('x', rotation=0)
    axes[1].set_title('I(x,t)')
    fig.colorbar(pic2, ax=axes[1])  # Colorbar for u

    plt.tight_layout()
    plt.gcf().set_dpi(300)
    plt.show()


def plot_space_time_3d(t,y,inputs,locations):
    """
    Plot a 3D surface of the field activity over space and time.
    """

    x = locations
    
    upper_lim_y = y.max()
    lower_lim_y = y.min()

    upper_lim_u = inputs.max()
    lower_lim_u = inputs.min()

    x_mesh, t_mesh = np.meshgrid(x, t)

    fig = plt.figure(figsize=(16, 8))  # Wider figure for two subplots

    # First subplot: y
    ax1 = fig.add_subplot(121, projection='3d')  # 1 row, 2 columns, 1st subplot
    surf1 = ax1.plot_surface(t_mesh, x_mesh, y, cmap='plasma', linewidth=0, antialiased=False)

    ax1.xaxis.pane.fill = False
    ax1.yaxis.pane.fill = False
    ax1.zaxis.pane.fill = False
    ax1.set_box_aspect([2, 1, 1])
    ax1.set_xlabel('t', linespacing=3.2)
    ax1.set_ylabel('x', linespacing=3.1)
    ax1.set_zlabel('v(x,t)', linespacing=3.4, rotation=0)
    ax1.set_zlim(lower_lim_y, upper_lim_y)

    fig.colorbar(surf1, ax=ax1, shrink=0.4, aspect=10, pad=0.2)

    # Second subplot: u
    ax2 = fig.add_subplot(122, projection='3d')  # 1 row, 2 columns, 2nd subplot
    surf2 = ax2.plot_surface(t_mesh, x_mesh, inputs, cmap='plasma', linewidth=0, antialiased=False)

    ax2.xaxis.pane.fill = False
    ax2.yaxis.pane.fill = False
    ax2.zaxis.pane.fill = False
    ax2.set_box_aspect([2, 1, 1])
    ax2.set_xlabel('t', linespacing=3.2)
    ax2.set_ylabel('x', linespacing=3.1)
    ax2.set_zlabel('I(x,t)', linespacing=3.4, rotation=0)
    ax2.set_zlim(lower_lim_u, upper_lim_u)

    fig.colorbar(surf2, ax=ax2, shrink=0.4, aspect=10, pad=0.2)

    plt.show()


def plot_space_time_3d_mv(t, y, inputs, locations):
    """
    Plot 3D surfaces of field activities over space and time:
    y[:,:,0] (v), y[:,:,1] (u), and external inputs.
    """

    x = locations
    x_mesh, t_mesh = np.meshgrid(x, t)

    y_u = y[:, :, 0]  # second field (u)
    y_v = y[:, :, 1]  # first field (v)

    fig = plt.figure(figsize=(24, 8))

    # Plot y_v
    ax1 = fig.add_subplot(131, projection='3d')
    surf1 = ax1.plot_surface(t_mesh, x_mesh, y_v, cmap='plasma', linewidth=0, antialiased=False)
    ax1.set_box_aspect([2, 1, 1])
    ax1.set_xlabel('t')
    ax1.set_ylabel('x')
    ax1.set_zlabel('v(x,t)', rotation=0)
    ax1.set_zlim(y_v.min(), y_v.max())
    fig.colorbar(surf1, ax=ax1, shrink=0.4, aspect=10, pad=0.2)

    # Plot y_u
    ax2 = fig.add_subplot(132, projection='3d')
    surf2 = ax2.plot_surface(t_mesh, x_mesh, y_u, cmap='plasma', linewidth=0, antialiased=False)
    ax2.set_box_aspect([2, 1, 1])
    ax2.set_xlabel('t')
    ax2.set_ylabel('x')
    ax2.set_zlabel('u(x,t)', rotation=0)
    ax2.set_zlim(y_u.min(), y_u.max())
    fig.colorbar(surf2, ax=ax2, shrink=0.4, aspect=10, pad=0.2)

    # Plot input
    ax3 = fig.add_subplot(133, projection='3d')
    surf3 = ax3.plot_surface(t_mesh, x_mesh, inputs, cmap='plasma', linewidth=0, antialiased=False)
    ax3.set_box_aspect([2, 1, 1])
    ax3.set_xlabel('t')
    ax3.set_ylabel('x')
    ax3.set_zlabel('I(x,t)', rotation=0)
    ax3.set_zlim(inputs.min(), inputs.max())
    fig.colorbar(surf3, ax=ax3, shrink=0.4, aspect=10, pad=0.2)

    plt.tight_layout()
    plt.show()

def plot_amari(plot,t,y,locations,inputs):
    '''
    visualizes trajectory
    '''

    if plot == "slider_1d":
        plot_slider_1d(t,y,inputs,locations)

    if plot == "space_time_1d":
        plot_space_time_flat(t,y,inputs,locations)

    if plot == "space_time_3d":
        plot_space_time_3d(t,y,inputs,locations)

    if plot == "space_time_3d_mv":
        plot_space_time_3d_mv(t,y,inputs,locations)

if __name__ == "__main__":

    ## assign data corresponding to trajectory
    data_path = Path("data/amari_coupled_difficult_theta1.pkl")
    with data_path.open('rb') as f:
        data = pickle.load(f)

    # get data from trajectory
    current_trajectory = 0
    trajectory = 4
    for (x0, x0_n, t, y, y_n, u) in data['train']:
        current_trajectory += 1
        dt = t[1][0].numpy()
        activity = y.numpy()
        inputs = u.numpy()
        t_lim = t[-1][0]
        if current_trajectory == trajectory:
            break
    print(activity.shape)
    print(inputs.shape)
    plot_amari("space_time_3d_mv",t.numpy(),y.numpy(),data['Locations'].numpy(),u.numpy())
    plot_amari("space_time_1d",t.numpy(),y[:,:,0].numpy(),data['Locations'].numpy(),u.numpy())
    plot_amari("space_time_1d",t.numpy(),y[:,:,1].numpy(),data['Locations'].numpy(),u.numpy())
    # plot_amari("space_time_2dd",t.numpy(),y.numpy(),data['Locations'].numpy(),u.numpy())
    # plot_amari("space_time_3d",trajectory=1)