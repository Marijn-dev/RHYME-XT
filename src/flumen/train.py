import torch


def prep_inputs(x0, y, u, lengths, time_start,time_end,device):
    sort_idxs = torch.argsort(lengths, descending=True)
    x0 = x0[sort_idxs]
    y = y[sort_idxs]
    u = u[sort_idxs]
    lengths = lengths[sort_idxs]
    time_start = time_start[sort_idxs]
    time_end = time_end[sort_idxs]
    deltas = u[:, :lengths[0], -1].unsqueeze(-1)

    # u = torch.nn.utils.rnn.pack_padded_sequence(u,
    #                                             lengths,
    #                                             batch_first=True,
    #                                             enforce_sorted=True)


    x0 = x0.to(device)
    y = y.to(device)
    u = u.to(device)
    time_start = time_start.to(device)
    time_end = time_end.to(device)
    lengths = lengths.to(device)
    deltas = deltas.to(device)


    return x0, y, u, deltas, time_start, time_end,lengths


def validate(data,locations,loss_fn, model, device):
    vl = 0.
    data_loss_total = 0.
    with torch.no_grad():
        for example in data:
            x0, y, u, deltas, time_start, time_end,lengths = prep_inputs(*example, device)
            # iterate over the batch
            for batch_index, length in enumerate(lengths):
                x0_batch = x0[batch_index]
                y_batch = y[batch_index]
                u_batch = u[batch_index]
                for i in range(length): # Here DeepONet works as iterative solver, i.e: it takes its own output as input for the next step
                    u_input = u_batch[i]
                    y_pred,basis_functions = model(x0_batch, u_input, locations.to(device))
                    x0_batch = y_pred
                total_loss, data_loss = loss_fn(y_batch, y_pred,basis_functions)
                vl += total_loss.item()
                data_loss_total += data_loss.item() 
    return vl / len(data), data_loss_total / len(data)


def train_step(example,locations, loss_fn, model, optimizer, device):
    x0, y, u, deltas, time_start, time_end,lengths = prep_inputs(*example, device)
    
    optimizer.zero_grad()
    total_loss = 0.

    # iterate over the batch
    for batch_index, length in enumerate(lengths):
        x0_batch = x0[batch_index]
        y_batch = y[batch_index]
        u_batch = u[batch_index]
        for i in range(length): # Here DeepONet works as iterative solver, i.e: it takes its own output as input for the next step
            u_input = u_batch[i]
            y_pred,basis_functions = model(x0_batch, u_input, locations.to(device))
            x0_batch = y_pred

        total_loss, _ = loss_fn(y_batch, y_pred,basis_functions)

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
