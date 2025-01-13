import torch
import xarray as xr
import numpy as np


def predict(dataset, model, batch_size, device, target="y_hat"):

    model.eval()
    
    model = model.to(device)

    n, t, _ = dataset.xd.shape

    arr = []
    for i in range(0, n, batch_size):
        d = torch.from_numpy(dataset.xd[i : (i + batch_size)].values).float().to(device)
        s = torch.from_numpy(dataset.xs[i : (i + batch_size)].values).float().to(device)

        static_bt = s.unsqueeze(1).repeat(1, d.size(1), 1).to(device)

        x_concat = torch.cat(
            (d, static_bt),
            dim=-1,
        )

        out_dict = model(x_concat)

        arr.append(out_dict[target].detach().cpu().numpy())

    return np.vstack(arr)


def predict_convlstm(dataset, model, seq_len, device, coords=None, transpose=False):
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

    arr = []  # loop over seq_lengh
    for i in range(0, t, seq_len):
        xd = torch.FloatTensor(dataset.xd[i : (i + seq_len)].values[None])

        xs = (
            torch.FloatTensor(dataset.xs.values[None])
            .unsqueeze(1)
            .repeat(1, xd.size(1), 1, 1, 1)
        )

        X = torch.concat([xd, xs], 2).to(device)

        out = model(X)[0][0]  # remove batch
        if transpose:  # -> T F H W
            out = out.permute(0, 3, 1, 2)

        arr.append(out.detach().cpu().numpy())
    arr = np.vstack(arr)
    if coords is not None:
        arr = xr.DataArray(arr, coords=coords)
    return arr


