import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import torch
from matplotlib.gridspec import GridSpec
from matplotlib.animation import FuncAnimation, PillowWriter

def animate_trajectory(
    y, y_pred_list, t_feed,
    labels=None,
    filename="trajectory.gif",
    fps=5
):
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    from matplotlib.animation import FuncAnimation, PillowWriter
    import numpy as np

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
    y_pred_np_list = [np.flip(y_pred, axis=0).T for y_pred in y_pred_list] # if taking predictions
    # y_pred_np_list = [y_pred.T for y_pred in y_pred_list]   # if taking the input
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

    # Constant θ line (doesn't change)
    # line_const, = ax.plot(x, np.ones_like(x),
    #                       label=r'$\theta$',
    #                       color="red", linestyle="--")
    # Not added to lines → so update() won't touch it

    # ax.legend()

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
    return filename
def plot_trajectory_multiple_trajectories(
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
        'figure.figsize': [25, 8.5],  # width x height
    })
    colors = ["black", "#0072BD", "#D95319", "#77AC30", "#7E2F8E"]
    linestyles = ['-', '--', ':', '-.', '-']

    y_np = y.T
    y_pred_np_list = [np.flip(y_pred, axis=0).T for y_pred in y_pred_list]
    t_feed = np.flip(t_feed)
    all_data = [y_np] + y_pred_np_list
    global_min = min(arr.min() for arr in all_data)
    global_max = max(arr.max() for arr in all_data)

    x = np.linspace(0, 25, y.shape[1])

    n_cols = min(4, len(time_indices))
    n_rows = 2

    fig, axs = plt.subplots(1, n_cols,dpi=100,
                        sharex='row')

    if n_cols == 1:
        axs = axs.reshape(n_rows, 1)

    x = np.linspace(0, 25, y.shape[1])

    for i, t in enumerate(time_indices):
        ax = axs[i]
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
        if i == 0:
            ax.set_ylabel(r'u(x,t=b)')

    # for i, idx in enumerate(space_indices):
    #     ax = axs[1, i]
    #     ax.plot(t_feed, y_np[idx, :], label=labels[0], color=colors[0], linestyle=linestyles[0])
    #     for j, y_pred_np in enumerate(y_pred_np_list):
    #         ax.plot(t_feed, y_pred_np[idx, :],
    #                 label=labels[j + 1] if labels else f"Pred {j+1}",
    #                 color=colors[(j + 1) % len(colors)],
    #                 linestyle=linestyles[(j + 1) % len(linestyles)])
    #     ax.set_ylim(global_min, global_max)
    #     ax.set_title(f"a = {x[idx]:.0f}")
    #     ax.grid(True)
    #     # Remove individual labels
    #     ax.set_xlabel('t')
    #     ax.set_ylabel('')

   
    # fig.text(0.01, 0.30,r'u(x,t=b)', va='center', ha='center',
    #      rotation='vertical', fontsize=20)

    # Second row shared y-axis label
    # fig.text(0.01, 0.30, r'u(x=a,t)', va='center', ha='center',
        #  rotation='vertical', fontsize=20)

    # # Add legend as before
    # handles, legend_labels = axs[0, 0].get_legend_handles_labels()
    # legend = fig.legend(handles, legend_labels,
    #                     loc='center right',
    #                     frameon=True,
    #                     framealpha=1.0,
    #                     edgecolor='black',
    #                     facecolor='white',
    #                     ncol=1)
    # legend.set_draggable(True)

    fig.tight_layout(rect=[0.02, 0.02, 1, 1],h_pad=2.0,w_pad=1)  # leave space for labels and legend

    return fig

def plot_multiple_trajectories_heatmap(y, y_pred_list, t_feed, labels=None):
                                      
    """
    Plots ground truth on top, followed by multiple model predictions as heatmaps.

    Parameters
    ----------
    y_true : np.ndarray
        Ground truth 2D array (e.g., shape (101, 100)).
    predictions : list of np.ndarray
        List of 2D prediction arrays with same shape as y_true.
    model_names : list of str, optional
        Names for the models (same order as predictions).
    extent : list [xmin, xmax, ymin, ymax]
        Coordinate range for imshow.
    vmin, vmax : float
        Color scale min/max.
    """

    n_plots = 1 + len(y_pred_list)  # ground truth + predictions

    # === Plot style ===
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
        'figure.figsize': [12, 6 * n_plots],  # width x height
    })

    y_np = y.T
    y_pred_np_list = [np.flip(y_pred, axis=0).T for y_pred in y_pred_list]
    t_feed = np.flip(t_feed)
    all_data = [y_np] + y_pred_np_list
    global_min = min(arr.min() for arr in all_data)
    global_max = max(arr.max() for arr in all_data)
    
    fig, axes = plt.subplots(n_plots, 1, constrained_layout=True)
    extent=[0, 200, 0, 25]
    vmin=0
    vmax=1
    # If only 1 prediction, axes might not be a list
    if n_plots == 1:
        axes = [axes]

#     # --- Plot ground truth ---
#     im = axes[0].imshow(y_np, aspect='auto', origin='lower',
#                         cmap='viridis', vmin=vmin, vmax=vmax, extent=extent)
#     axes[0].set_title("Ground Truth")
#     axes[0].set_xlabel('Time index')
#     axes[0].set_ylabel('Space index')
#     fig.colorbar(im, ax=axes[0], label='Value')

    # --- Plot predictions ---
    for i, pred in enumerate(all_data):
        im = axes[i].imshow(pred, aspect='auto', origin='lower',
                              cmap='viridis', vmin=vmin, vmax=vmax, extent=extent)
        name = labels[i] if labels else f"Prediction {i+1}"
        axes[i].set_title(name)
        axes[i].set_xlabel('t')
        axes[i].set_ylabel('x')
        # fig.colorbar(im, ax=axes[i], label='')
    # Shared colorbar
    fig.colorbar(im, ax=axes, orientation='vertical', label=r"u(x,t)", fraction=0.04, pad=0.04)
    # fig.tight_layout(pad=4)  # leave space for labels and legend
    return fig

def plot_predictions_with_l1_errors_and_centered_reference(y_true, y_pred_list, t_feed, labels=None):
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
        'figure.figsize': [14, 5.8 * (n_preds + 0.9)],  # width fixed, height scales
    })

    y_true_np = y_true.T
    y_pred_np_list = [np.flip(pred, axis=0).T for pred in y_pred_list]

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
    return fig

def plot_predictions_with_l1_errors_and_centered_reference_flow_only(y_true, y_pred_list, t_feed, labels=None):
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
        'figure.figsize': [10, 4 * (1 + 2*n_preds)],  # width fixed, height scales
    })

    # Prepare data
    y_true_np = y_true.T
    y_pred_np_list = [np.flip(pred, axis=0).T for pred in y_pred_list]

    extent = [0, 200, 0, 25]
    vmin, vmax = 0, 1

    # Create stacked subplots: 1 for reference, then 2 per prediction (pred + error)
    fig, axes = plt.subplots(1 + 2*n_preds, 1, constrained_layout=True)
    fig, axes = plt.subplots(1, 1, constrained_layout=True)

    # Plot reference
    ax_ref = axes[0]
    im_ref = ax_ref.imshow(y_true_np, aspect='auto', origin='lower',
                           cmap='viridis', vmin=vmin, vmax=vmax, extent=extent)
    ref_title = labels[0] if labels and len(labels) > 0 else "Reference"
    ax_ref.set_title(ref_title)
    ax_ref.set_xlabel('t')
    ax_ref.set_ylabel('x')
    # fig.colorbar(im_ref, ax=ax_ref, orientation='vertical', fraction=0.03, pad=0.02)

    # Loop through predictions
    for i, pred in enumerate(y_pred_np_list):
        ax_pred = axes[1 + 2*i]
        ax_err  = axes[2 + 2*i]

        pred_title = labels[i + 1] if labels and len(labels) > i + 1 else f"Prediction {i + 1}"
        
        im_pred = ax_pred.imshow(pred, aspect='auto', origin='lower',
                                 cmap='viridis', vmin=vmin, vmax=vmax, extent=extent)
        ax_pred.set_title(f"{pred_title}")
        ax_pred.set_xlabel('t')
        ax_pred.set_ylabel('x')
        # fig.colorbar(im_pred, ax=ax_pred, orientation='vertical', fraction=0.03, pad=0.02)

        error = np.abs(pred - y_true_np)
        im_err = ax_err.imshow(error, aspect='auto', origin='lower',
                               cmap='viridis', vmin=vmin, vmax=vmax, extent=extent)
        ax_err.set_title(r"$|u(x,t)-\tilde{u}(x,t)|$")
        ax_err.set_xlabel('t')
        ax_err.set_ylabel('x')

    cbar = fig.colorbar(im_err, ax=axes, orientation='vertical', fraction=0.03, pad=0.02)
    cbar.set_label("")

    plt.show()
    return fig

def plot_reference_only(y_true, t_feed, labels=None):
    """
    Plot only the reference heatmap.

    Args:
        y_true: Array of shape (nt, nx)
        t_feed: Time array of shape (nt,)
        labels: Optional list for titles [reference]
    """
    # === Plot style ===
    mpl.rcParams.update({
        'text.usetex': False,
        'axes.titlesize': 18,
        'axes.labelsize': 18,
        'xtick.labelsize': 14,
        'ytick.labelsize': 14,
        'font.size': 16,
        'axes.grid': False,
        'figure.figsize': [10, 4],
    })

    # Prepare data
    # y_true_np = y_true.T  # (nx, nt)
    # y_true_np = np.flip(y_true, axis=0).T # if not ref but model
    t_feed = np.flip(t_feed)
    extent = [t_feed[0], t_feed[-1], 0, 25]  # time on x, space on y
    vmin, vmax = 0, 1
    y_true_np = y_true
    # Plot
    fig, ax = plt.subplots(constrained_layout=True)
    im_ref = ax.imshow(y_true_np, aspect='auto', origin='lower',
                       cmap='viridis', vmin=vmin, vmax=vmax, extent=extent)

    ref_title = labels[0] if labels else "Reference"
    ax.set_title(ref_title)
    ax.set_xlabel('t')
    ax.set_ylabel('x')

    # cbar = fig.colorbar(im_ref, ax=ax, orientation='vertical',
    #                     fraction=0.03, pad=0.02)
    # cbar.set_label("u(x,t)")

    plt.show()
    return fig
    
def main():
    data = np.load('plot/data/don_vs_flow/results_9_L1.npz')
    input_data = np.load('plot/data/don_vs_flow/input_9.npz')
    print(data.files)

    # labels=["Ground truth",  "RHYME-XT","DeepONet", "DeepONet with Shorter Segments"] # Deeponet vs flow
    # labels= ["Ground truth",  "RHYME-XT"] # Flow only
    # labels = ['Ground truth',r'N=2250',r'N=90',r"$N=90 \; (\mu_1)$",r'$N=90 \; (\mu_3)$'] # Transfer learning
    # labels = ['Ground truth',r'$\hat{N}_x=100$',r'$\hat{N}_x=75$',r'$\hat{N}_x=50$',r'$\hat{N}_x=25$'] # Discretization
    # data_pred = [data['y_pred_x100'],data['y_pred_x75'],data['y_pred_x50'],data['y_pred_x25']]
    data_pred = [data['y_pred_flow'],data['y_pred_don']]
    # data_pred = [input_data['input']]
    # # labels=['u(x,t)', 'I(x,t)', 'DeepONet', 'DeepONet with Shorter Segments'],
    # gif_file = animate_trajectory(data['y'],data_pred, data['t_feed'],
    #                             labels=[r'$u(x,t)$', r'RHYME-XT',r'DeepONet'],
    #                             filename="flowvsdeeponet.gif", fps=12)
    # print(f"GIF saved to {gif_file}")
    # plot heatmap, flow only
    # plot_predictions_with_l1_errors_and_centered_reference_flow_only(data['y'],  # shape (nt, nx)
    # [],
    # data['t_feed'],
    # labels=[r''])
    error = np.abs(np.flip(data['y_pred_flow'], axis=0).T - data['y'].T)
    plot_reference_only(error, data['t_feed'], labels=[""])
    return
    # Plot heatmap of predictions with different models
    # fig = plot_predictions_with_l1_errors_and_centered_reference(
    # data['y'],  # shape (nt, nx)
    # data_pred,
    # data['t_feed'],
    # labels=labels)

    ## Plot samples in trajectory
    fig = plot_trajectory_multiple_trajectories(data['y'], data_pred, data['t_feed'], labels=labels,time_indices=[39,40,41,42], space_indices=[])
    plt.show()

if __name__ == "__main__":
    main()