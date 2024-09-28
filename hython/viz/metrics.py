import numpy as np
import matplotlib.pyplot as plt
import geopandas as gpd
import xarray as xr
import matplotlib.colors as colors
from matplotlib.colors import BoundaryNorm, CenteredNorm
from matplotlib.ticker import MaxNLocator
import cartopy.crs as ccrs
from cartopy.io.img_tiles import QuadtreeTiles

from hython.metrics import compute_bias, compute_rmse, compute_pbias

def set_norm(color_norm, color_bounds, ticks, ncolors, clip=False):
    if color_norm == "bounded":
        norm = BoundaryNorm(ticks, ncolors=ncolors, clip=clip)
        norm.vmin = color_bounds[0]
        norm.vmax = color_bounds[-1]
    elif color_norm == "unbounded":
        norm = CenteredNorm()
    else:
        raise NotImplementedError
    return norm


def compute_kge(y_true, y_pred):
    if np.any(np.isnan(y_true)) or np.any(np.isnan(y_pred)):
        return np.array([np.nan, np.nan, np.nan, np.nan])

    # r = np.corrcoef(observed, simulated)[1, 0]
    # alpha = np.std(simulated, ddof=1) /np.std(observed, ddof=1)
    # beta = np.mean(simulated) / np.mean(observed)
    # kge = 1 - np.sqrt(np.power(r-1, 2) + np.power(alpha-1, 2) + np.power(beta-1, 2))

    m1, m2 = np.mean(y_true, axis=0), np.mean(y_pred, axis=0)
    num_r = np.sum((y_true - m1) * (y_pred - m2), axis=0)
    den_r = np.sqrt(np.sum((y_true - m1) ** 2, axis=0)) * np.sqrt(
        np.sum((y_pred - m2) ** 2, axis=0)
    )
    r = num_r / den_r
    beta = m2 / m1
    gamma = (np.std(y_pred, axis=0) / m2) / (np.std(y_true, axis=0) / m1)
    kge = 1.0 - np.sqrt((r - 1.0) ** 2 + (beta - 1.0) ** 2 + (gamma - 1.0) ** 2)

    return np.array([kge, r, gamma, beta])


def compute_kge_parallel(y_target, y_pred):
    kge = xr.apply_ufunc(
        compute_kge,
        y_target,
        y_pred,
        input_core_dims=[["time"], ["time"]],
        output_core_dims=[["kge"]],
        output_dtypes=[float],
        vectorize=True,
        dask="parallelized",
        dask_gufunc_kwargs={"output_sizes": {"kge": 4}},
    )

    kge = kge.assign_coords({"kge": ["kge", "r", "alpha", "beta"]})
    return kge


def map_pearson(y: xr.DataArray, yhat, dim="time"):
    p = xr.corr(y, yhat, dim=dim)
    fig, ax = plt.subplots(1, 1)
    i = ax.imshow(p, cmap="RdBu", norm=colors.CenteredNorm())
    fig.colorbar(i, ax=ax, label="Pearson corr coeff")


def map_kge(
    y: xr.DataArray,
    yhat,
    dim="time",
    unit="",
    figsize=(10, 10),
    label_1="wflow",
    label_2="LSTM",
    kwargs_imshow={},
    return_kge=False,
    ticks=None,
    title = None
):
    cmap = plt.colormaps["Greens"]
    vmin = kwargs_imshow.get("vmin", False)

    minx, miny, maxx, maxy = y.rio.bounds()

    kge = compute_kge_parallel(y, yhat)
    kge = kge.chunk({"kge": 1})
    kge = kge.sel(kge="kge")

    # fig, ax = plt.subplots(1,1, figsize = figsize, projection=ccrs.PlateCarree())
    map_proj = ccrs.PlateCarree()
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
    ax.set_extent([minx, maxx, miny, maxy], crs=ccrs.PlateCarree())
    try:
        ax.add_wms(wms="http://vmap0.tiles.osgeo.org/wms/vmap0", layers=["basic"])
    except:
        pass

    if vmin:
        norm = BoundaryNorm(ticks, ncolors=cmap.N, clip=True)
        norm.vmin = kwargs_imshow.pop("vmin")
        norm.vmax = kwargs_imshow.pop("vmax")
        # i = ax.imshow(rmse, cmap=cmap, norm=norm, **kwargs_imshow)
        i = kge.plot(
            ax=ax,
            norm=norm,
            cmap=cmap,
            transform=ccrs.PlateCarree(),
            add_colorbar=False,
        )

        fig.colorbar(i, ax=ax, shrink=0.5, label=f"KGE", ticks=ticks)
    else:
        norm = CenteredNorm()
        i = kge.plot(
            ax=ax,
            norm=norm,
            cmap=cmap,
            transform=ccrs.PlateCarree(),
            add_colorbar=False,
        )
        # i = ax.imshow(rmse, cmap=cmap, norm= norm, **kwargs_imshow)

        fig.colorbar(i, ax=ax, shrink=0.5, label=f"KGE")
    if title:    
        plt.title(title)
    if return_kge:
        return fig, ax, kge
    else:
        return fig, ax


# MAPS

def map_rmse(
    y_true: xr.DataArray,
    y_pred: xr.DataArray,
    figsize=(10, 10),
    label_1="wflow",
    label_2="LSTM",
    title="",
    color_norm = "unbounded",
    color_bounds = [-100,100],
    color_bad = None,
    color_ticks = None,
    matplot_kwargs = dict(alpha=1),
    alpha_gridlines = 0.1,
    tiles = QuadtreeTiles(),
    scale = 13,
    map_extent = [],
    return_computation=False,
    unit = "mm"
    ):
    
    # COMPUTE
    rmse = compute_rmse(y_true, y_pred)

    # MATPLOTLIB PARAMETERS
    cmap = plt.colormaps["RdYlGn"]

    norm = set_norm(color_norm, color_bounds, color_ticks, cmap.N, clip=True)

    if color_bad is not None:
        cmap.set_bad(color_bad)

    # CARTOPY STUFF
    map_proj = ccrs.PlateCarree()
    if len(map_extent) == 0:
        minx, miny, maxx, maxy = y_true.rio.bounds()
    else:
        minx, miny, maxx, maxy = map_extent

    # PLOT
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
    ax.set_extent([minx, maxx, miny, maxy], crs=ccrs.PlateCarree())

    gl = ax.gridlines(draw_labels=True, dms=False, x_inline=False, y_inline=False, alpha=alpha_gridlines)
    
    if tiles is not None:
        ax.add_image(tiles, scale)

    p = rmse.plot(
        ax=ax,
        norm=norm,
        cmap=cmap,
        transform=ccrs.PlateCarree(),
        add_colorbar = False,
        **matplot_kwargs
    )

    fig.colorbar(
        p,
        ax=ax,
        shrink=0.5,
        label=f"{label_2} < {label_1}    {unit}     {label_2} > {label_1}",
        ticks=color_ticks,
    )

    plt.title(title)
    
    if return_computation:
        return fig, ax, rmse
    else:
        return fig, ax

def map_gamma(
    y_true: xr.DataArray,
    y_pred: xr.DataArray,
    figsize=(10, 10),
    label_1="wflow",
    label_2="LSTM",
    title="",
    color_norm = "bounded",
    color_bounds = [0, 2],
    color_bad = None,
    color_ticks = None,
    matplot_kwargs = dict(alpha=1),
    alpha_gridlines = 0.1,
    tiles = QuadtreeTiles(),
    scale = 13,
    map_extent = [],
    return_computation=False,
    unit = ""
    ):
    
    # COMPUTE
    out = compute_gamma(y_true, y_pred)


    # MATPLOTLIB PARAMETERS
    cmap = plt.colormaps["RdYlGn"]
    if color_ticks is None:
        color_ticks = [c*0.1 for c in range(0, 21, 1)]
    norm = set_norm(color_norm, color_bounds, color_ticks, cmap.N, clip=True)

    if color_bad is not None:
        cmap.set_bad(color_bad)

    # CARTOPY STUFF
    map_proj = ccrs.PlateCarree()
    if len(map_extent) == 0:
        minx, miny, maxx, maxy = y_true.rio.bounds()
    else:
        minx, miny, maxx, maxy = map_extent

    # PLOT
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
    ax.set_extent([minx, maxx, miny, maxy], crs=ccrs.PlateCarree())

    gl = ax.gridlines(draw_labels=True, dms=False, x_inline=False, y_inline=False, alpha=alpha_gridlines)
    
    if tiles is not None:
        ax.add_image(tiles, scale)

    p = out.plot(
        ax=ax,
        norm=norm,
        cmap=cmap,
        transform=ccrs.PlateCarree(),
        add_colorbar = False,
        **matplot_kwargs
    )

    fig.colorbar(
        p,
        ax=ax,
        shrink=0.5,
        label=f"{label_2} < {label_1}    {unit}     {label_2} > {label_1}",
        ticks=color_ticks,
    )

    plt.title(title)
    
    if return_computation:
        return fig, ax, bias
    else:
        return fig, ax

def map_bias(
    y_true: xr.DataArray,
    y_pred: xr.DataArray,
    figsize=(10, 10),
    label_1="wflow",
    label_2="LSTM",
    title="",
    color_norm = "bounded",
    color_bounds = [-100,100],
    color_bad = None,
    color_ticks = None,
    matplot_kwargs = dict(alpha=1),
    alpha_gridlines = 0.1,
    tiles = QuadtreeTiles(),
    scale = 13,
    map_extent = [],
    return_computation=False,
    percentage_bias=True,
    unit = None
    ):
    
    # COMPUTE
    if percentage_bias:
        bias = compute_pbias(y_true, y_pred)
        unit = unit if unit is not None else "%"
    else:
        bias = compute_bias(y_true, y_pred)
        unit = unit if unit is not None else "mm"

    # MATPLOTLIB PARAMETERS
    cmap = plt.colormaps["RdYlGn"]
    if color_ticks is None and percentage_bias is True:
        color_ticks = [c*10 for c in range(-10, 11, 1)]
    norm = set_norm(color_norm, color_bounds, color_ticks, cmap.N, clip=True)

    if color_bad is not None:
        cmap.set_bad(color_bad)

    # CARTOPY STUFF
    map_proj = ccrs.PlateCarree()
    if len(map_extent) == 0:
        minx, miny, maxx, maxy = y_true.rio.bounds()
    else:
        minx, miny, maxx, maxy = map_extent

    # PLOT
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
    ax.set_extent([minx, maxx, miny, maxy], crs=ccrs.PlateCarree())

    gl = ax.gridlines(draw_labels=True, dms=False, x_inline=False, y_inline=False, alpha=alpha_gridlines)
    
    if tiles is not None:
        ax.add_image(tiles, scale)

    p = bias.plot(
        ax=ax,
        norm=norm,
        cmap=cmap,
        transform=ccrs.PlateCarree(),
        add_colorbar = False,
        **matplot_kwargs
    )

    fig.colorbar(
        p,
        ax=ax,
        shrink=0.5,
        label=f"{label_2} < {label_1}    {unit}     {label_2} > {label_1}",
        ticks=color_ticks,
    )

    plt.title(title)
    
    if return_computation:
        return fig, ax, bias
    else:
        return fig, ax