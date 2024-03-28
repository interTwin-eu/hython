import numpy as np
import matplotlib.pyplot as plt
import geopandas as gpd
import xarray as xr
import matplotlib.colors as colors
from matplotlib.colors import BoundaryNorm, CenteredNorm
from matplotlib.ticker import MaxNLocator



def plot_sampler(da_bkg, meta, meta_valid, figsize = (10,10), markersize= 10, cmap="terrain"):
    
    vv = da_bkg
    
    vv = vv.assign_coords({"gridcell":(("lat", "lon"), meta.idx_grid_2d)})
    
    vv = vv.assign_coords({"gridcell_valid":(("lat", "lon"), meta_valid.idx_grid_2d)})
    
    tmp = np.zeros(vv.shape).astype(np.bool_)
    for i in meta.idx_sampled_1d:
        tmp[vv.gridcell == i] = True
    
    tmp_valid = np.zeros(vv.shape).astype(np.bool_)
    for i in meta_valid.idx_sampled_1d:
        tmp_valid[vv.gridcell_valid == i] = True
    
    df = vv.where(tmp).to_dataframe().dropna().reset_index()
    
    df_valid = vv.where(tmp_valid).to_dataframe().dropna().reset_index()
    
    gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(x=df.lon, y=df.lat), crs=4326)
    
    gdf_valid = gpd.GeoDataFrame(df_valid, geometry=gpd.points_from_xy(x=df_valid.lon, y=df_valid.lat), crs=4326)
    
    fig, ax = plt.subplots(1,1, figsize=figsize)
    da_bkg.plot(ax = ax, add_colorbar=False, alpha = 0.5, cmap=cmap)
    gdf.plot(ax=ax, color="red", markersize= markersize, label="training")
    gdf_valid.plot(ax=ax, color="black", markersize= markersize, label = "validation")
    plt.legend(bbox_to_anchor=(1.05, 1.05))
    #ax.set_xlim([6, 7.5])
    #ax.set_ylim([45.5, 46.5])
    
    
    return fig,ax


def compute_pbias(y_in: xr.DataArray, yhat_in, dim="time", offset=0):
    
    y = y_in.copy() + offset
    yhat = yhat_in.copy() + offset
    if isinstance(y, xr.DataArray) or isinstance(yhat, xr.DataArray): 
        return 100*(( yhat - y).sum(dim=dim, skipna=False) / y.sum(dim=dim,skipna=False))
    else:
        return 100* np.sum(yhat -y, axis=2) / np.sum(y, axis=2)    

def compute_bias(y_in: xr.DataArray, yhat_in, dim="time", offset=0):
    y = y_in.copy() + offset
    yhat = yhat_in.copy() + offset
    if isinstance(y, xr.DataArray) or isinstance(yhat, xr.DataArray): 
        return (yhat - y).sum(dim=dim, skipna=False) / len(yhat)
    else:
        return np.sum(yhat -y, axis=2) / len(yhat)  


def map_pearson(y: xr.DataArray, yhat, dim="time"):
    p = xr.corr(y, yhat, dim=dim)
    fig, ax = plt.subplots(1,1)
    i = ax.imshow(p, cmap="RdBu", norm=colors.CenteredNorm())
    fig.colorbar(i, ax=ax, label="Pearson corr coeff")

def map_pbias(y: xr.DataArray, yhat, dim="time", figsize = (10,10), label_1 = "wflow", label_2 = "LSTM", kwargs_imshow = {}, offset = 0, return_pbias = False):
    cmap = plt.colormaps['RdBu']
    vmin = kwargs_imshow.get("vmin", False)

    pbias = compute_pbias(y, yhat, dim, offset=offset)
    fig, ax = plt.subplots(1,1, figsize = figsize)  
    
    if vmin:
        ticks = [l*10 for l in range(-10,11, 1)]
        norm = BoundaryNorm(ticks, ncolors=cmap.N, clip=True)
        norm.vmin = kwargs_imshow.pop("vmin")
        norm.vmax = kwargs_imshow.pop("vmax")
        i = ax.imshow(pbias, cmap=cmap, norm=norm, **kwargs_imshow)
        fig.colorbar(i, ax=ax, shrink=0.5, label=f"{label_2} < {label_1}    %     {label_2} > {label_1}", ticks = ticks )
    else:
        norm = CenteredNorm()
        i = ax.imshow(pbias, cmap=cmap, norm= norm, **kwargs_imshow)
        
        fig.colorbar(i, ax=ax, shrink=0.5, label=f"{label_2} < {label_1}    %     {label_2} > {label_1}")
    if return_pbias:
        return pbias

def map_bias(y: xr.DataArray, yhat, dim="time", unit = "mm", figsize = (10,10), label_1 = "wflow", label_2 = "LSTM", kwargs_imshow = {}, offset = 0, return_bias = False):
    cmap = plt.colormaps['RdBu']
    vmin = kwargs_imshow.get("vmin", False)

    bias = compute_bias(y, yhat, dim, offset=offset)
    fig, ax = plt.subplots(1,1, figsize = figsize)  
    
    if vmin:
        ticks = [l*10 for l in range(-10,11, 1)]
        norm = BoundaryNorm(ticks, ncolors=cmap.N, clip=True)
        norm.vmin = kwargs_imshow.pop("vmin")
        norm.vmax = kwargs_imshow.pop("vmax")
        i = ax.imshow(bias, cmap=cmap, norm=norm, **kwargs_imshow)
        fig.colorbar(i, ax=ax, shrink=0.5, label=f"{label_2} < {label_1}    {unit}     {label_2} > {label_1}", ticks = ticks )
    else:
        norm = CenteredNorm()
        i = ax.imshow(bias, cmap=cmap, norm= norm, **kwargs_imshow)
        
        fig.colorbar(i, ax=ax, shrink=0.5, label=f"{label_2} < {label_1}    {unit}     {label_2} > {label_1}")
    if return_bias:
        return bias

def map_at_timesteps(y: xr.DataArray, yhat: xr.DataArray, dates = None, label_pred = "LSTM", label_target = "wflow"):
    ts = dates if dates else y.time.dt.date.values 
    
    for t in dates:
        fig, ax = plt.subplots(1,2, figsize= (20,15))
        fig.subplots_adjust(hspace=0.3)
        vmax = np.nanmax([yhat.sel(time=t),y.sel(time=t)])
        
        l1 = ax[0].imshow(yhat.sel(time=t), vmax=vmax)
        ax[0].set_title("LSTM", fontsize=28)
        fig.colorbar(l1, ax=ax[0],shrink=0.3)
        
        l2 = ax[1].imshow(y.sel(time=t), vmax=vmax)
        ax[1].set_title("wflow", fontsize=28)
        fig.colorbar(l2, ax=ax[1],shrink=0.3)
        fig.suptitle(t, y = 0.8, fontsize=20, fontweight="bold")
        fig.tight_layout()
        
        
        
def ts_compare(y: xr.DataArray, yhat, lat= [], lon = [], label_1 = "wflow", label_2 = "LSTM", bkg_map = None):
    time = y.time.values
    for ilat,ilon in zip(lat, lon):
        ax_dict = plt.figure(layout="constrained", figsize=(20,6)).subplot_mosaic(
        """
        AC
        BC
        """,
        width_ratios=[4, 1]
        )
        iy = y.sel(lat = ilat,lon = ilon, method="nearest")
        iyhat = yhat.sel(lat = ilat,lon = ilon, method="nearest") 
        ax_dict["A"].plot(time, iyhat, label = label_2)
        ax_dict["A"].plot(time, iy, label= label_1)
        ax_dict["A"].legend()
        ax_dict["B"].scatter(iy,iyhat, s=1)
        xmin = np.nanmin( np.concatenate([iy, iyhat] )) - 0.05
        xmax = np.nanmax( np.concatenate([iy, iyhat] )) + 0.05
        ax_dict["B"].set_xlim(xmin, xmax)
        ax_dict["B"].set_ylim(xmin, xmax)
        ax_dict["B"].axline((0, 0), (1, 1), color="black", linestyle="dashed")
        ax_dict["B"].set_ylabel(label_2)
        ax_dict["B"].set_xlabel(label_1)
        df = gpd.GeoDataFrame([],geometry=gpd.points_from_xy(x=[ilon], y=[ilat]))
        if bkg_map is not None:
            bkg_map.plot(ax=ax_dict["C"], add_colorbar=False, cmap="terrain")
        else:
            y.mean("time").plot(ax=ax_dict["C"], add_colorbar=False)
        df.plot(ax=ax_dict["C"], markersize=20, color="red")
        plt.title(f"lat, lon:  ({ ilat}, {ilon})")