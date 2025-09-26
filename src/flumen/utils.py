import torch
import numpy as np
from argparse import ArgumentParser, ArgumentTypeError
import pickle 
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
import torch.nn.functional as F
import matplotlib as mpl
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.gridspec import GridSpec
from matplotlib.widgets import Slider

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

def trajectory(data,trajectory_index,delta):
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

def plot_space_time_trajectory(
    y, y_pred,
    time_indices=[0, 20, 40, 60, 80],
    space_indices=[0, 25, 50, 75,99]
):
    '''Returns:
       - Heatmaps: Ground truth, Prediction, Absolute Error.
       - Line plots over space at selected times.
       - Line plots over time at selected neurons.
       - Red dotted lines at selected neuron indices.
    '''
    # Flip and convert tensors to numpy
    y_pred = torch.flip(y_pred, dims=[0])
    y_np = y.detach().cpu().numpy().T  # shape: (neurons, time)
    y_pred_np = y_pred.detach().cpu().numpy().T
    error_np = np.abs(y_np - y_pred_np)
    l1_loss = F.l1_loss(y.cpu(), y_pred.cpu()).item()

    # Determine global y-limits
    global_min = min(y_np.min(), y_pred_np.min())
    global_max = max(y_np.max(), y_pred_np.max())

    num_times = len(time_indices)
    num_space = len(space_indices)
    n_cols = max(4, num_times)
    n_rows = 3

    fig = plt.figure(figsize=(3 * n_cols, 7), dpi=100)
    gs = fig.add_gridspec(n_rows, n_cols)

    # --- Heatmaps ---

    # Ground Truth
    ax0 = fig.add_subplot(gs[0, 0:2])
    im0 = ax0.imshow(y_np, aspect='auto', cmap='viridis', vmin=0, vmax=1)
    ax0.set_title("Ground Truth (y)")
    plt.colorbar(im0, ax=ax0)
    for t in time_indices:
        ax0.axvline(x=t, color='red', linestyle='--')
    for s in space_indices:
        ax0.axhline(y=s, color='red', linestyle=':')

    # Prediction
    ax1 = fig.add_subplot(gs[0, 2:4])
    im1 = ax1.imshow(y_pred_np, aspect='auto', cmap='viridis', vmin=0, vmax=1)
    ax1.set_title("Prediction (y_pred)")
    plt.colorbar(im1, ax=ax1)
    for t in time_indices:
        ax1.axvline(x=t, color='red', linestyle='--')
    for s in space_indices:
        ax1.axhline(y=s, color='red', linestyle=':')

    # Absolute Error (Same colormap)
    ax2 = fig.add_subplot(gs[0, 4:])
    im2 = ax2.imshow(error_np, aspect='auto', cmap='viridis', vmin=0, vmax=1)
    ax2.set_title("Absolute Error |y - y_pred|")
    plt.colorbar(im2, ax=ax2)
#     for t in time_indices:
#         ax2.axvline(x=t, color='blue', linestyle='--')
#     for s in space_indices:
#         ax2.axhline(y=s, color='red', linestyle=':')

    # --- Line plots over space at selected times ---
    for i, t in enumerate(time_indices):
        y_t = y_np[:, t]
        y_t_pred = y_pred_np[:, t]
        L1_loss_t = np.mean(np.abs(y_t - y_t_pred))
        L2_loss_t = np.mean((y_t - y_t_pred) ** 2)
        ax = fig.add_subplot(gs[1, i])
        ax.plot(y_t, label=r"$u(x)$", linestyle='--', color='b')
        ax.plot(y_t_pred, label=r"$u_{\text{pred}}(x)$", linestyle='-', color='r')
        ax.set_ylim(global_min, global_max)
        ax.set_title(f"t={t}\nL1={L1_loss_t:.4f}, L2={L2_loss_t:.4f}")
        ax.set_xlabel("Neuron Index")
        ax.set_ylabel("Activation")
        ax.legend()
        ax.grid(True)

    # --- Line plots over time at selected neurons ---
    for i, idx in enumerate(space_indices):
        y_idx = y_np[idx, :]
        y_pred_idx = y_pred_np[idx, :]
        l1_loss_neuron = np.mean(np.abs(y_idx - y_pred_idx))
        ax = fig.add_subplot(gs[2, i])
        ax.plot(y_idx, label=r"$u(t)$", linestyle='--', color='b')
        ax.plot(y_pred_idx, label=r"$u_{\text{pred}}(t)$", linestyle='-', color='r')
        ax.set_ylim(global_min, global_max)
        ax.set_title(f"Neuron={idx}\nL1={l1_loss_neuron:.4f}")
        ax.set_xlabel("Time")
        ax.set_ylabel("Activation")
        ax.legend()
        ax.grid(True)
    fig.tight_layout()
    return fig


def plot_2D_trajectories(
    y, y_pred_list, t_feed,
    labels=None,
    time_indices=[25, 50, 75],
    space_indices=[21, 50, 78]
    ):
    '''
    Args:
        y: Ground truth array of shape (nt, nx)
        y_pred_list: List of prediction arrays, each of shape (nt, nx)
        t_feed: Time array of shape (nt,)
        labels: List of legend labels [ground_truth, pred1, pred2, ...]
    '''
    
    # === Plot style ===
    mpl.rcParams.update({
        'text.usetex': False,
        'axes.titlesize': 22,
        'axes.labelsize': 22,
        'xtick.labelsize': 16,
        'ytick.labelsize': 16,
        'legend.fontsize': 20,
        'font.size': 18,
        'axes.grid': True,
        'grid.linestyle': '-',
        'grid.alpha': 0.7,
        'lines.linewidth': 1.5,
        'figure.figsize': [10, 4],  # width x height
    })
    colors = ["black", "#0072BD", "#D95319", "#77AC30", "#7E2F8E"]
    linestyles = ['-', '--', ':', '-.', '-']

    y_np = y.T
    y_pred_np_list = [y_pred.T for y_pred in y_pred_list]
    t_feed = np.array(t_feed)
    t_feed = np.flip(t_feed)
    all_data = [y_np] + y_pred_np_list
    global_min = min(arr.min() for arr in all_data)
    global_max = max(arr.max() for arr in all_data)

    x = np.linspace(0, 25, y.shape[1])

    n_cols = min(4, len(time_indices))
    n_rows = 2

    fig, axs = plt.subplots(2, n_cols,dpi=100,
                        sharex='row')

    if n_cols == 1:
        axs = axs.reshape(n_rows, 1)

    x = np.linspace(0, 25, y.shape[1])

    for i, t in enumerate(time_indices):
        ax = axs[0,i]
        ax.plot(x, y_np[:, t], label=labels[0], color=colors[0], linestyle=linestyles[0])
        for j, y_pred_np in enumerate(y_pred_np_list):
            ax.plot(x, y_pred_np[:, t],
                    label=labels[j + 1] if labels else f"Pred {j+1}",
                    color=colors[(j + 1) % len(colors)],
                    linestyle=linestyles[(j + 1) % len(linestyles)])
        ax.set_ylim(global_min, global_max)
        ax.set_title(f"b = {t_feed[t].item():.1f}")
        ax.grid(True)
        # Remove individual labels
        ax.set_xlabel('x')
    

    for i, idx in enumerate(space_indices):
        ax = axs[1, i]
        ax.plot(t_feed, y_np[idx, :], label=labels[0], color=colors[0], linestyle=linestyles[0])
        for j, y_pred_np in enumerate(y_pred_np_list):
            ax.plot(t_feed, y_pred_np[idx, :],
                    label=labels[j + 1] if labels else f"Pred {j+1}",
                    color=colors[(j + 1) % len(colors)],
                    linestyle=linestyles[(j + 1) % len(linestyles)])
        ax.set_ylim(global_min, global_max)
        ax.set_title(f"a = {x[idx]:.0f}")
        ax.grid(True)
        # Remove individual labels
        ax.set_xlabel('t')
        ax.set_ylabel('')

   
    fig.text(0.01, 0.70,r'u(x,t=b)', va='center', ha='center',
         rotation='vertical', fontsize=20)

    # Second row shared y-axis label
    fig.text(0.01, 0.30, r'u(x=a,t)', va='center', ha='center',
         rotation='vertical', fontsize=20)

    # Add legend as before
    handles, legend_labels = axs[0, 0].get_legend_handles_labels()
    legend = fig.legend(handles, legend_labels,
                        loc='center right',
                        frameon=True,
                        framealpha=1.0,
                        edgecolor='black',
                        facecolor='white',
                        ncol=1)
    legend.set_draggable(True)

    fig.tight_layout(rect=[0.02, 0.02, 1, 1],h_pad=2.0,w_pad=1)  # leave space for labels and legend
    plt.show()

def plot_heatmap(y_true, y_pred_list, t_feed, labels=None):
    n_preds = len(y_pred_list)

    mpl.rcParams.update({
        'text.usetex': False,
        'axes.titlesize': 18,
        'axes.labelsize': 18,
        'xtick.labelsize': 16,
        'ytick.labelsize': 16,
        'legend.fontsize': 20,
        'font.size': 18,
        'axes.grid': True,
        'grid.linestyle': '-',
        'grid.alpha': 0.7,
        'lines.linewidth': 1.5,
        'figure.figsize': [10, 4 * (n_preds + 0.9)],  # width fixed, height scales
    })

    y_true_np = y_true.T
    y_pred_np_list = [y_pred.T for y_pred in y_pred_list]

    extent = [0, 50, 0, 25]
    vmin = 0
    vmax = 1

    h_ratios = [1.0] + [1]*n_preds
    fig = plt.figure(constrained_layout=True)
    gs = GridSpec(n_preds + 1, 2, figure=fig, height_ratios=h_ratios, width_ratios=[1, 1], wspace=0.0001,hspace=0.002)

    # Reference plot spanning both columns in first row
    ax_ref = fig.add_subplot(gs[0, :])
    im_ref = ax_ref.imshow(y_true_np, aspect='auto', origin='lower',
                           cmap='viridis', vmin=vmin, vmax=vmax, extent=extent)
    ref_title = labels[0] if labels and len(labels) > 0 else "Reference"
    ax_ref.set_aspect(0.30) 
    ax_ref.set_title(ref_title)
    ax_ref.set_xlabel('t')
    ax_ref.set_ylabel('x')

    # Prediction and error plots below
    for i in range(n_preds):
        ax_pred = fig.add_subplot(gs[i + 1, 0])
        ax_err = fig.add_subplot(gs[i + 1, 1])
        ax_row_title = fig.add_subplot(gs[i + 1, :])
        ax_row_title.axis('off')  # hide axis lines and ticks
        pred_title = labels[i + 1] if labels and len(labels) > i + 1 else f"Prediction {i + 1}"
        ax_row_title.set_title(pred_title, pad=10)

        pred = y_pred_np_list[i]
        im_pred = ax_pred.imshow(pred, aspect='auto', origin='lower',
                                cmap='viridis', vmin=vmin, vmax=vmax, extent=extent)
        ax_pred.set_title(r"$\tilde{u}(x,t)$")
        ax_pred.set_xlabel('t')
        ax_pred.set_ylabel('x')

        error = np.abs(pred - y_true_np)
        im_err = ax_err.imshow(error, aspect='auto', origin='lower',
                               cmap='viridis', vmin=vmin, vmax=vmax, extent=extent)
        ax_err.set_title(r"$|u(x,t)-\tilde{u}(x,t)|$")
        ax_err.set_xlabel('t')
        ax_err.set_ylabel('x')

    # # Shared colorbars: predictions left column, errors right column
    # cbar_pred = fig.colorbar(im_pred, ax=fig.get_axes()[1::2], orientation='vertical', fraction=0.03, pad=0.02)
    # cbar_pred.set_label("")

    cbar_err = fig.colorbar(im_err, ax=fig.get_axes()[2::2], orientation='vertical', fraction=0.03, pad=0.02)
    cbar_err.set_label("")
    plt.show()

def save_GIF(
    y, y_pred_list, t_feed,
    labels=None,
    filename="trajectory.gif",
    fps=50
    ):

    print(f"Creating GIF to save to {filename}...")
    # === Plot style ===
    mpl.rcParams.update({
        'text.usetex': False,
        'axes.titlesize': 18,
        'axes.labelsize': 18,
        'xtick.labelsize': 14,
        'ytick.labelsize': 14,
        'legend.fontsize': 14,
        'font.size': 16,
        'axes.grid': True,
        'grid.linestyle': '-',
        'grid.alpha': 0.7,
        'lines.linewidth': 1.5,
        'figure.figsize': [10, 6],
    })
    colors = ["black", "#0072BD", "#D95319", "#77AC30", "#7E2F8E"]
    linestyles = ['-', '--', ':', '-.', '-']

    # Process data
    y_np = y.T  # (nx, nt)
    y_pred_np_list = [y_pred.T for y_pred in y_pred_list] # if taking predictions
    t_feed = np.array(t_feed)
    t_feed = np.flip(t_feed)
    all_data = [y_np] + y_pred_np_list
    global_min = min(arr.min() for arr in all_data)
    global_max = max(arr.max() for arr in all_data)

    x = np.linspace(0, 25, y.shape[1])
    nt = y.shape[0]

    # Set up figure
    fig, ax = plt.subplots()
    ax.set_xlim(x.min(), x.max())
    ax.set_ylim(global_min, global_max)
    ax.set_xlabel("x")
    ax.set_ylabel("u(x,t)")
    title = ax.set_title("")

    # Plot ground truth
    lines = []
    line_gt, = ax.plot([], [], label=labels[0] if labels else "Ground Truth",
                       color=colors[0], linestyle=linestyles[0])
    lines.append(line_gt)

    # Predictions
    for j, _ in enumerate(y_pred_np_list):
        line_pred, = ax.plot([], [], label=labels[j+1] if labels else f"Pred {j+1}",
                             color=colors[(j+1) % len(colors)],
                             linestyle=linestyles[(j+1) % len(linestyles)])
        lines.append(line_pred)

    # Uncomment if you want to see the activatation threshold
    # Constant θ line (doesn't change)
    # line_const, = ax.plot(x, np.ones_like(x),
    #                       label=r'$\theta$',
    #                       color="red", linestyle="--")
    # Not added to lines → so update() won't touch it

    ax.legend()

    # Update function
    def update(frame):
        # Update ground truth
        lines[0].set_data(x, y_np[:, frame])
        # Update predictions
        for j, y_pred_np in enumerate(y_pred_np_list):
            lines[j+1].set_data(x, y_pred_np[:, frame])
        title.set_text(f"t = {t_feed[frame].item():.2f}")
        return lines + [title]

    # Create animation
    anim = FuncAnimation(fig, update, frames=nt, interval=1500/fps, blit=True)

    # Save to GIF
    anim.save(filename, writer=PillowWriter(fps=fps))

    plt.close(fig)

def plot_slider(
    y, y_pred_list, t_feed,
    labels=None,
):
    # === Plot style ===
    mpl.rcParams.update({
        'text.usetex': False,
        'axes.titlesize': 18,
        'axes.labelsize': 18,
        'xtick.labelsize': 14,
        'ytick.labelsize': 14,
        'legend.fontsize': 14,
        'font.size': 16,
        'axes.grid': True,
        'grid.linestyle': '-',
        'grid.alpha': 0.7,
        'lines.linewidth': 1.5,
        'figure.figsize': [10, 6],
    })

    colors = ["black", "#0072BD", "#D95319", "#77AC30", "#7E2F8E"]
    linestyles = ['-', '--', ':', '-.', '-']

    # Process data
    y_np = y.T  # (nx, nt)
    y_pred_np_list = [y_pred.T for y_pred in y_pred_list] 
    t_feed = np.array(t_feed)
    t_feed = np.flip(t_feed)
    all_data = [y_np] + y_pred_np_list
    global_min = min(arr.min() for arr in all_data)
    global_max = max(arr.max() for arr in all_data)

    x = np.linspace(0, 25, y.shape[1])
    nt = y.shape[0]

    # Set up figure
    fig, ax = plt.subplots()
    plt.subplots_adjust(bottom=0.25)  # leave space for slider
    ax.set_xlim(x.min(), x.max())
    ax.set_ylim(global_min, global_max)
    ax.set_xlabel("x")
    ax.set_ylabel("u(x,t)")
    title = ax.set_title("")

    # Plot ground truth
    lines = []
    line_gt, = ax.plot([], [], label=labels[0] if labels else "Ground Truth",
                       color=colors[0], linestyle=linestyles[0])
    lines.append(line_gt)

    # Predictions
    for j, _ in enumerate(y_pred_np_list):
        line_pred, = ax.plot([], [], label=labels[j+1] if labels else f"Pred {j+1}",
                             color=colors[(j+1) % len(colors)],
                             linestyle=linestyles[(j+1) % len(linestyles)])
        lines.append(line_pred)

    ax.legend()

    # Initialize first frame
    def update_plot(frame):
        lines[0].set_data(x, y_np[:, frame])
        for j, y_pred_np in enumerate(y_pred_np_list):
            lines[j+1].set_data(x, y_pred_np[:, frame])
        title.set_text(f"t = {t_feed[frame].item():.2f}")
        fig.canvas.draw_idle()

    # Draw initial state
    update_plot(0)

    # Slider axis
    ax_slider = plt.axes([0.2, 0.1, 0.65, 0.03])  
    slider = Slider(ax_slider, 'Frame', 0, nt-1, valinit=0, valstep=1)

    # Connect slider to update function
    def on_change(val):
        frame = int(slider.val)
        update_plot(frame)

    slider.on_changed(on_change)

    plt.show()
