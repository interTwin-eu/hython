import torch
import numpy as np


def predict(Xd, Xs, model, batch_size, device):
    model = model.to(device)
    arr = []
    for i in range(0, Xd.shape[0], batch_size):
        d = torch.Tensor(Xd[i : (i + batch_size)]).to(device)

        s = torch.Tensor(Xs[i : (i + batch_size)]).to(device)
        arr.append(model(d, s).detach().cpu().numpy())
    return np.vstack(arr)