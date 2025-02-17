import torch


def prep_inputs(x0, y, u, lengths, device):
    sort_idxs = torch.argsort(lengths, descending=True)

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


def validate(data, loss_fn, model, device):
    vl = 0.

    with torch.no_grad():
        for example in data:
            x0, y, u, deltas = prep_inputs(*example, device)

            y_pred = model(x0, u, deltas)
            vl += loss_fn(y, y_pred).item()

    return model.state_dim * vl / len(data)


def train_step(example, loss_fn, model, optimizer, device):
    x0, y, u, deltas = prep_inputs(*example, device)

    optimizer.zero_grad()

    y_pred = model(x0, u, deltas)
    loss = model.state_dim * loss_fn(y, y_pred)

    loss.backward()
    optimizer.step()

    return loss.item()


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
