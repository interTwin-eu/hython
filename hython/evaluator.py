import torch
import xarray as xr
import numpy as np
import pandas as pd
from pathlib import Path
from hython.viz.general import plot_distr
from .utils import get_temporal_steps, create_xarray_data
from hython.viz import ts_compare,ts_plot, map_bias, map_kge, map_rmse, map_pearson, map_nse
from hython.metrics import compute_kge_parallel, compute_bias, compute_pbias, compute_nse, compute_pearson, compute_rmse
from typing import List
from hython.datasets import WflowSBMCube
from hython.models import *

def format_with_uncertainty(value, uncertainty):
    return f"{value:.5f} ± {uncertainty:.5f}"

def convert_to_uncertainty_format(df):
    formatted_data = {}
    columns = df.columns
    
    for i in range(0, len(columns), 2):
        value_col = columns[i]
        uncertainty_col = columns[i + 1]
        new_col_name = value_col.replace("_std", "")
        formatted_data[new_col_name] = [
            format_with_uncertainty(val, unc) 
            for val, unc in zip(df[value_col], df[uncertainty_col])
        ]
    
    return pd.DataFrame(formatted_data)

def predict(dataset, dataloader, model, device, target="y_hat"):
    model.eval()

    model = model.to(device)

    # try:
    #     n, t, _ = dataset.xd.shape
    # except:
    #     n, t, _ = list(dataset.xd.dims.values())
        
    arr = []
    for data in dataloader:

        d = data["xd"].to(device)
        s = data["xs"].to(device)

        static_bt = s.unsqueeze(1).repeat(1, d.size(1), 1).to(device)

        x_concat = torch.cat(
            (d, static_bt),
            dim=-1,
        )

        out_dict = model(x_concat)

        arr.append(out_dict[target].detach().cpu().numpy())

    return np.vstack(arr)


def predict_convlstm(dataloader, model, device, target= "y_hat"):
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
    model.eval()

    arr = []  # loop over seq_lengh
    for data in dataloader:
        
        d = data["xd"].to(device)
        s = data["xs"].to(device)

        st = (
            s.unsqueeze(1)
            .repeat(1, d.size(1), 1, 1, 1)
        )

        X = torch.concat([d, st], 2).to(device)

        out = model(X)[target][0]

        arr.append(out.detach().cpu().numpy())
    arr = np.vstack(arr)

    return arr

# def predict_convlstm(dataset, model, seq_len, device, coords=None, transpose=False):
#     """_summary_

#     Parameters
#     ----------
#     dataset : _type_
#         _description_
#     model : _type_
#         _description_
#     seq_len : _type_
#         _description_
#     device : _type_
#         _description_
#     coords : _type_, optional
#         Dimensions ordered as "time","lat", "lon","feat", by default None
#     transpose : bool, optional
#         _description_, by default False
#     """
#     model = model.to(device)

#     try:
#         t, c, h, w   = dataset.xd.shape
#     except:
#         t, c, h, w  = list(dataset.xd.dims.values())

#     arr = []  # loop over seq_lengh
#     for i in range(0, t, seq_len):
#         xd = torch.FloatTensor(dataset.xd[i : (i + seq_len)].values[None])

#         xs = (
#             torch.FloatTensor(dataset.xs.values[None])
#             .unsqueeze(1)
#             .repeat(1, xd.size(1), 1, 1, 1)
#         )

#         X = torch.concat([xd, xs], 2).to(device)

#         out = model(X)[0][0]  # remove batch
#         if transpose:  # -> T F H W
#             out = out.permute(0, 3, 1, 2)

#         arr.append(out.detach().cpu().numpy())
#     arr = np.vstack(arr)
#     if coords is not None:
#         arr = xr.DataArray(arr, coords=coords)
#     return arr

REGISTERED_OUTPUT = ["map", "distr", "ts_compare", "global_metric"]

class Evaluator:
    def __init__(self, cfg):
        self.cfg = cfg
        self.list_output = REGISTERED_OUTPUT

        self.out_dir = Path(self.cfg.evaluator.dir_out)
        if not self.out_dir.exists():
            self.out_dir.mkdir()

    def predict(self, dataset, loader, model, device, target="y_hat", **kwargs):
        """ """
        # TODO: if condition to handle conv_lstm
        if isinstance(model, ConvLSTM):
            coords = kwargs.get("coords")
            return predict_convlstm(dataset, model, self.cfg.seq_length, device, target=target, coords=coords)
        if isinstance(model, CudaLSTM):
            return predict(dataset, loader, model, device, target=target)

    def to_xarray(self, arr, coords, output_shape):
        
        return create_xarray_data(arr, 
                          coords, 
                          output_shape=output_shape
                         )
        
    def get_target(self, dataset):
        """get target from dataset"""
        if isinstance(dataset, WflowSBMCube):
            return dataset.y.to_dataset("feat")
        return dataset.y
        
    def preprocess(self, dataset, loader, model, device, target="y_hat", kwargs_predict = {}) -> List[xr.Dataset]:

        ds_target = self.get_target(dataset)
        #try:
        self.var_names = list(ds_target.data_vars)
        # except:
        #     # FIXME: this is due to WflowSBMCube which requires refactoring
        #     self.var_names = list(ds_target.feat.values)
        
        print("variables: ", self.var_names)
        
        # masking
        mask = ~dataset.mask
        
        # predict, pred has same variable name as target
        coords = xr.Coordinates({"lat":ds_target.lat, "lon":ds_target.lon, "time":ds_target.time, "variable":self.var_names})
        output_shape = {"lat":len(ds_target.lat),"lon":len(ds_target.lon),"time":len(ds_target.time), "variable":len(self.var_names)}
        
        pred_arr = self.predict(dataset, loader, model, device, target=target, **kwargs_predict)
        
        pred = self.to_xarray(pred_arr, coords, output_shape)
        return ds_target.where(mask), pred.where(mask)
        
    def run(self, target, pred):
        """run all from config"""
        for output in self.list_output:
            otype = self.cfg.evaluator.get(output)
            if otype:
                print(f"running {output}")
                getattr(self, f"{output}")(target, pred)
            #else:
            #    print(f"{output} not in registed output list {self.list_output}")
    
    def map(self, target, pred):
        for variable in self.cfg.evaluator.map.var:
            for metric in self.cfg.evaluator.map.metric:
                pred[variable] = pred[variable].chunk({"time": -1, "lat": 50, "lon": 50})
                target[variable] = target[variable].chunk({"time": -1, "lat": 50, "lon": 50})
                if metric == "kge":
                    f, ax = map_kge(target[variable], pred[variable], cartopy=False, tiles=None)
                elif metric == "bias":
                    f, ax = map_bias(target[variable], pred[variable], cartopy=False, tiles=None)
                elif metric == "rmse":
                    f, ax = map_rmse(target[variable], pred[variable])
                elif metric == "nse":
                    f, ax = map_nse(target[variable], pred[variable], cartopy=False, tiles=None)
                elif metric == "pearson":
                    f, ax = map_pearson(target[variable], pred[variable], cartopy=False, tiles=None)
                elif metric == "pbias":
                    f, ax = map_bias(target[variable], pred[variable], cartopy=False, tiles=None , 
                                     percentage_bias=True, color_norm="bounded")
                ax.set_title(f"{variable}_{metric}")
                if self.cfg.evaluator.map.write:
                    f.savefig(f"{str(self.out_dir)}/map_{variable}_{metric}.png")
        
    def distr(self, target, pred):
        for variable in self.cfg.evaluator.distr.var:  
            f, ax = plot_distr(target[variable], pred[variable],title=variable, xlabel=self.cfg.evaluator.var_metadata[variable].unit)
            if self.cfg.evaluator.distr.write:
                f.savefig(f"{str(self.out_dir)}/distr_{variable}.png")
            
    def ts_compare(self, target, pred):
        # TODO: save to disk
        # TODO: ts_compare has an inside loop and does not return ax or fig, fix that.
        lats, lons = [], []
        for i in self.cfg.evaluator.ts_compare.coords:
            lats.append(i[0])
            lons.append(i[1])
        for variable in self.cfg.evaluator.ts_compare.var:  
            ax_dict = ts_compare(target[variable], pred[variable], lat=lats, lon=lons)
        
    def global_metric(self, target, pred):
        """Returns a pandas dataframe with global metrics as columns and
        variables as rows. If in a jupyter notebook, it will also display the
        dataframe."""
        pd.options.display.float_format = '{:.5f}'.format
        outd = {}
        for variable in self.cfg.evaluator.global_metric.var:  
            ind = {}
            ind_std = {}
            for metric in self.cfg.evaluator.global_metric.metric:
                pred[variable] = pred[variable].chunk({"time": -1, "lat": 25, "lon": 25})
                target[variable] = target[variable].chunk({"time": -1, "lat": 25, "lon": 25})
                if metric == "kge":
                    ind[metric] = compute_kge_parallel(target[variable], pred[variable]).mean().compute().item(0)
                    ind_std[metric] = compute_kge_parallel(target[variable], pred[variable]).std().compute().item(0)
                elif metric == "bias":
                    ind[metric] = compute_bias(target[variable], pred[variable]).mean().compute().item(0)
                    ind_std[metric] = compute_bias(target[variable], pred[variable]).std().compute().item(0)
                elif metric == "rmse":
                    ind[metric] = compute_rmse(target[variable], pred[variable]).mean().compute().item(0)
                    ind_std[metric] = compute_rmse(target[variable], pred[variable]).std().compute().item(0)
                elif metric == "nse":
                    ind_std[metric] = compute_nse(target[variable], pred[variable]).std().compute().item(0)
                elif metric == "pearson":
                    ind[metric] = compute_pearson(target[variable], pred[variable]).mean().compute().item(0)
                    ind_std[metric] = compute_pearson(target[variable], pred[variable]).std().compute().item(0)
                elif metric == "pbias":
                    ind[metric] = compute_pbias(target[variable], pred[variable]).mean().compute().item(0)
                    ind_std[metric] = compute_pbias(target[variable], pred[variable]).std().compute().item(0)
            outd[variable] = ind
            outd[variable+"_std"] = ind_std
        df = pd.DataFrame(outd)
        if self.cfg.evaluator.global_metric.write:
            df.to_csv(f"{str(self.out_dir)}/global_metric.csv")

        formatted_df = convert_to_uncertainty_format(df)
        print(formatted_df)
        return df
    
    