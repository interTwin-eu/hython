import torch
import xarray as xr
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


def predict_convlstm(dataset, model, seq_len, device, coords = None, transpose=False):
    """_summary_

    Parameters
    ----------
    dataset : _type_
        _description_
    model : _type_
        _description_
    seq_len : _type_
        _description_
    device : _type_
        _description_
    coords : _type_, optional
        Dimensions ordered as "time","lat", "lon","feat", by default None
    transpose : bool, optional
        _description_, by default False
    """
    model = model.to(device)

    t, c, h, w = dataset.xd.shape

    arr = [] # loop over seq_lengh
    for i in range(0, t , seq_len):

        xd = torch.FloatTensor(dataset.xd[i:(i + seq_len)].values[None])

        xs = torch.FloatTensor(dataset.xs.values[None]).unsqueeze(1).repeat(1, xd.size(1), 1, 1, 1)

        X = torch.concat([xd, xs], 2).to(device)

        out = model(X)[0][0] # remove batch
        if transpose: # -> T F H W
            out = out.permute(0, 3, 1, 2 )

        arr.append(out.detach().cpu().numpy())
    arr = np.vstack(arr)
    if coords is not None: 
        arr = xr.DataArray(arr, coords=coords)
    return arr