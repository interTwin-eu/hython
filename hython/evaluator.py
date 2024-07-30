import torch
import numpy as np
from hython.datasets.datasets import CubeletsDataset


def predict(Xd, Xs, model, batch_size, device):
    model = model.to(device)
    arr = []
    for i in range(0, Xd.shape[0], batch_size):
        d = torch.Tensor(Xd[i : (i + batch_size)]).to(device)

        s = torch.Tensor(Xs[i : (i + batch_size)]).to(device)
        arr.append(model(d, s).detach().cpu().numpy())
    return np.vstack(arr)


def predict_convlstm(dataset, model, seq_len, device, transpose=False):
    model = model.to(device)
    
    t, c, h, w = dataset.xd.shape
   
    arr = [] # loop over seq_lengh
    for i in range(0, t , seq_len):
        
        xd = torch.FloatTensor(dataset.xd[i:(i + seq_len)].values[None])

        xs = torch.FloatTensor(dataset.xs.values[None]).unsqueeze(1).repeat(1, xd.size(1), 1, 1, 1)
        
        X = torch.concat([xd, xs], 2).to(device)
        
        out = model(X)[0][0] # remove batch
        if transpose:
            out = out.permute(0, 3, 1, 2 )
        
        arr.append(out.detach().cpu().numpy())
    return np.vstack(arr)