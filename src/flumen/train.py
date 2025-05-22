import torch


def prep_inputs(x0, y, u, lengths, device):
    # sort_idxs = torch.argsort(lengths, descending=True)
    # sort_idxs_exp = sort_idxs.unsqueeze(-1).expand(-1, -1, x0.shape[-1])  # shape: [n_traj, T, dim]
    # sort_idxs_expanded = sort_idxs.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, u.shape[2], u.shape[3])  # shape: [n_traj, T, dim1, dim2]
    # lengths = torch.gather(lengths,dim=1,index=sort_idxs)
    # x0 = torch.gather(x0, dim=1, index=sort_idxs_exp)
    # y = torch.gather(y, dim=1, index=sort_idxs_exp)
    # u = torch.gather(u,dim=1,index=sort_idxs_expanded)

    # # print(lengths.shape)
    # deltas = []
    # for i,length in enumerate(lengths):
    #     deltas.append(u[i,:, :length[0], -1].unsqueeze(-1))
    # deltas = torch.stack(deltas).type(torch.get_default_dtype())
    # print(u.shape)
    # print(lengths.shape)
    # u = torch.nn.utils.rnn.pack_padded_sequence(u,
    #                                             lengths,
    #                                             batch_first=True,
    #                                             enforce_sorted=True)

    sort_idxs = torch.argsort(lengths, descending=True)
    x0 = x0.cpu()
    y = y.cpu()
    u = u.cpu()
    # print(x0.shape)
    x0 = x0[sort_idxs]
    y = y[sort_idxs]
    u = u[sort_idxs]
    lengths = lengths[sort_idxs]

    deltas = u[:, :lengths[0], -1].unsqueeze(-1)

    u = torch.nn.utils.rnn.pack_padded_sequence(u,
                                                lengths,
                                                batch_first=True,
                                                enforce_sorted=True)
    x0 = x0.to(device)
    y = y.to(device)
    u = u.to(device)
    deltas = deltas.to(device)

    return x0, y, u, deltas


def validate(data,locations,loss_fn, model, device):
    data_loss_total = 0. # total data loss over trajectories
    basis_loss_total = 0. # total basis loss over trajectories
    with torch.no_grad():

        # loop over data
        n_trajectories = 0 # keep account of n_trajectories to get trajectory loss
        for trajectories in data:
            x0, y, u, lengths = trajectories
            for i in range(0,x0.shape[0]):
                x0_traj, y_traj, u_traj, deltas_traj = prep_inputs(x0[i],y[i],u[i],lengths[i], device)
                y_pred, basis_functions = model(x0_traj, u_traj, locations.to(device),deltas_traj)
                
                data_loss, basis_loss  = loss_fn(y_traj, y_pred,basis_functions)
                data_loss_total += data_loss.item()             # extra loss in the case of orthogonal loss
                basis_loss_total += basis_loss.item() # data loss
                n_trajectories += 1 

    return data_loss_total / n_trajectories, basis_loss_total / n_trajectories


def train_step(trajectories,locations, loss_fn, model, optimizer, device):
    # loop over trajectories (batch)
    n_trajectories = 0 
    x0, y, u, lengths = trajectories
    optimizer.zero_grad()
    total_loss = 0
    for i in range(0,x0.shape[0]):
        x0_traj, y_traj, u_traj, deltas_traj = prep_inputs(x0[i],y[i],u[i],lengths[i], device)
        y_pred, basis_functions = model(x0_traj, u_traj,locations.to(device),deltas_traj)
        data_loss, basis_loss = loss_fn(y_traj, y_pred,basis_functions)
        loss = data_loss + basis_loss
        total_loss += loss
        n_trajectories += 1

    total_loss /= n_trajectories  
    total_loss.backward()
    optimizer.step()

    return total_loss.item()


class EarlyStopping:

    def __init__(self, es_patience, es_delta=0.):
        self.patience = es_patience
        self.delta = es_delta

        self.best_val_loss = float('inf')
        self.counter = 0
        self.early_stop = False
        self.best_model = False

    def step(self, val_loss):
        self.best_model = False

        if self.best_val_loss - val_loss > self.delta:
            self.best_val_loss = val_loss
            self.best_model = True
            self.counter = 0
        else:
            self.counter += 1

        if self.counter >= self.patience:
            self.early_stop = True
