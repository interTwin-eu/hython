import numpy as np
import matplotlib.pyplot as plt
import geopandas as gpd
import xarray as xr
import matplotlib.colors as colors
from matplotlib.colors import BoundaryNorm, CenteredNorm
from matplotlib.ticker import MaxNLocator
import cartopy.crs as ccrs
from cartopy.io.img_tiles import QuadtreeTiles

from hython.metrics.custom import (
    compute_bias,
    compute_rmse,
    compute_pbias,
    compute_gamma,
    compute_kge_parallel,
    compute_pearson,
    compute_variance,
    compute_nse_parallel
)


# MAPS
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

def return_map_extent(ds, map_extent):
    if len(map_extent) == 0:
        try:
            minx, miny, maxx, maxy = ds.rio.bounds()
        except:
            ds = ds.rio.set_spatial_dims(x_dim="lon", y_dim="lat")
            minx, miny, maxx, maxy = ds.rio.bounds()
    else:
        minx, miny, maxx, maxy = map_extent
    return minx, miny, maxx, maxy


def map_kge(
    y_true: xr.DataArray,
    y_pred: xr.DataArray,
    figsize=(10, 10),
    title="",
    color_norm="bounded",
    color_bounds=[-0.5, 1],
    color_bad=None,
    color_ticks=None,
    matplot_kwargs=dict(alpha=1),
    alpha_gridlines=0.1,
    cartopy=False,
    tiles=QuadtreeTiles(),
    scale=13,
    map_extent=[],
    return_computation=False,
    fig=None,
    ax=None
):
    # COMPUTE
    kge = compute_kge_parallel(y_true, y_pred)
    #kge = kge.chunk({"kge": 1})
    #kge = kge.sel(kge="kge")

    # MATPLOTLIB PARAMETERS
    cmap = plt.colormaps["Blues"]
    if color_ticks is None:
        color_ticks = np.linspace(color_bounds[0], color_bounds[-1], 16)

    norm = set_norm(color_norm, color_bounds, color_ticks, cmap.N, clip=True)

    if color_bad is not None:
        cmap.set_bad(color_bad)

    # PLOT
    if fig is None:
        fig = plt.figure(figsize=figsize)
    if ax is None:
        if cartopy:
            map_proj = ccrs.PlateCarree()

            minx, miny, maxx, maxy = return_map_extent(y_true, map_extent)

            ax = fig.add_subplot(1, 1, 1, projection=map_proj)

            ax.set_extent([minx, maxx, miny, maxy], crs=map_proj)
            
        else:
            ax = fig.add_subplot(1, 1, 1)


    if tiles is not None:
        ax.add_image(tiles, scale)
    if cartopy:
        p = kge.plot(
            ax=ax,
            norm=norm,
            cmap=cmap,
            transform=map_proj,
            add_colorbar=False,
            **matplot_kwargs,
        )
    else:
        p = kge.plot(
            ax=ax,
            norm=norm,
            cmap=cmap,
            #transform=map_proj,
            add_colorbar=False,
            **matplot_kwargs,
        )

    fig.colorbar(
        p,
        ax=ax,
        shrink=0.5,
        label=f"kge",
        ticks=color_ticks,
    )

    plt.title(title)

    if return_computation:
        return fig, ax, kge
    else:
        return fig, ax


def map_rmse(
    y_true: xr.DataArray,
    y_pred: xr.DataArray,
    figsize=(10, 10),
    label_1="wflow",
    label_2="LSTM",
    title="",
    color_norm="unbounded",
    color_bounds=[-100, 100],
    color_bad=None,
    color_ticks=None,
    matplot_kwargs=dict(alpha=1),
    alpha_gridlines=0.1,
    tiles=QuadtreeTiles(),
    scale=13,
    map_extent=[],
    return_computation=False,
    unit="mm",
    skipna=False,
):
    # COMPUTE
    rmse = compute_rmse(y_true, y_pred, skipna=skipna)

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

    gl = ax.gridlines(
        draw_labels=True,
        dms=False,
        x_inline=False,
        y_inline=False,
        alpha=alpha_gridlines,
    )

    if tiles is not None:
        ax.add_image(tiles, scale)

    p = rmse.plot(
        ax=ax,
        norm=norm,
        cmap=cmap,
        transform=ccrs.PlateCarree(),
        add_colorbar=False,
        **matplot_kwargs,
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
    color_norm="bounded",
    color_bounds=[0, 2],
    color_bad=None,
    color_ticks=None,
    color_cmap="viridis",
    matplot_kwargs=dict(alpha=1),
    alpha_gridlines=0.1,
    tiles=QuadtreeTiles(),
    scale=13,
    map_extent=[],
    return_computation=False,
    unit="",
):
    # COMPUTE
    out = compute_gamma(y_true, y_pred)

    # MATPLOTLIB PARAMETERS
    cmap = plt.colormaps[color_cmap]
    if color_ticks is None:
        color_ticks = np.linspace(color_bounds[0], color_bounds[-1], 21)
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

    gl = ax.gridlines(
        draw_labels=True,
        dms=False,
        x_inline=False,
        y_inline=False,
        alpha=alpha_gridlines,
    )

    if tiles is not None:
        ax.add_image(tiles, scale)

    p = out.plot(
        ax=ax,
        norm=norm,
        cmap=cmap,
        transform=ccrs.PlateCarree(),
        add_colorbar=False,
        **matplot_kwargs,
    )

    fig.colorbar(
        p,
        ax=ax,
        shrink=0.5,
        label=f"{label_2} < {label_1}        {label_2} > {label_1}",
        ticks=color_ticks,
    )

    plt.title(title)

    if return_computation:
        return fig, ax, out
    else:
        return fig, ax

def map_nse(
    y_true: xr.DataArray,
    y_pred: xr.DataArray,
    figsize=(10, 10),
    title="",
    color_norm="bounded",
    color_bounds=[-0.5, 1],
    color_bad = None,
    color_ticks=None,
    matplot_kwargs=dict(alpha=1),
    cartopy = False,
    alpha_gridlines=0.1,
    tiles=QuadtreeTiles(),
    scale=13,
    map_extent=[],
    return_computation=False,
    skipna=False,
    fig=None,
    ax=None,
):
    # COMPUTE

    nse = compute_nse_parallel(y_true, y_pred)

    # MATPLOTLIB PARAMETERS
    cmap = plt.colormaps["Blues"]
    if color_ticks is None:
        color_ticks = np.linspace(color_bounds[0], color_bounds[-1], 16)

    norm = set_norm(color_norm, color_bounds, color_ticks, cmap.N, clip=True)



    if color_bad is not None:
        cmap.set_bad(color_bad)

    # PLOT
    if fig is None:
        fig = plt.figure(figsize=figsize)
    if ax is None:
        if cartopy:
            map_proj = ccrs.PlateCarree()

            minx, miny, maxx, maxy = return_map_extent(y_true, map_extent)

            ax = fig.add_subplot(1, 1, 1, projection=map_proj)

            ax.set_extent([minx, maxx, miny, maxy], crs=map_proj)
            
        else:
            ax = fig.add_subplot(1, 1, 1)


    # gl = ax.gridlines(
    #     draw_labels=True,
    #     dms=False,
    #     x_inline=False,
    #     y_inline=False,
    #     alpha=alpha_gridlines,
    # )

    if tiles is not None:
        ax.add_image(tiles, scale)
    if cartopy:
        p = nse.plot(
            ax=ax,
            norm=norm,
            cmap=cmap,
            transform=map_proj,
            add_colorbar=False,
            **matplot_kwargs,
        )
    else:
        p = nse.plot(
            ax=ax,
            norm=norm,
            cmap=cmap,
            #transform=map_proj,
            add_colorbar=False,
            **matplot_kwargs,
        )

    fig.colorbar(
        p,
        ax=ax,
        shrink=0.5,
        label=f"nse",
        ticks=color_ticks,
    )

    plt.title(title)

    if return_computation:
        return fig, ax, nse
    else:
        return fig, ax


def map_bias(
    y_true: xr.DataArray,
    y_pred: xr.DataArray,
    figsize=(10, 10),
    label_1="wflow",
    label_2="LSTM",
    title="",
    color_norm="unbounded",
    color_bounds=[-100, 100],
    color_bad=None,
    color_ticks=None,
    matplot_kwargs=dict(alpha=1),
    cartopy = False,
    alpha_gridlines=0.1,
    tiles=QuadtreeTiles(),
    scale=13,
    map_extent=[],
    return_computation=False,
    percentage_bias=False,
    unit=None,
    skipna=False,
    fig=None,
    ax=None,
):
    # COMPUTE
    if percentage_bias:
        bias = compute_pbias(y_true, y_pred, skipna=skipna).compute()
        unit = unit if unit is not None else "%"
    else:
        bias = compute_bias(y_true, y_pred, skipna=skipna).compute()
        unit = unit if unit is not None else "mm"

    # MATPLOTLIB PARAMETERS
    cmap = plt.colormaps["RdYlGn"]

    if color_ticks is None and percentage_bias is True:
        color_ticks = [c * 10 for c in range(-10, 11, 1)]

    norm = set_norm(color_norm, color_bounds, color_ticks, cmap.N, clip=True)

    if color_bad is not None:
        cmap.set_bad(color_bad)

    # PLOT
    if fig is None:
        fig = plt.figure(figsize=figsize)
    if ax is None:
        if cartopy:
            map_proj = ccrs.PlateCarree()

            minx, miny, maxx, maxy = return_map_extent(y_true, map_extent)

            ax = fig.add_subplot(1, 1, 1, projection=map_proj)

            ax.set_extent([minx, maxx, miny, maxy], crs=map_proj)
            
        else:
            ax = fig.add_subplot(1, 1, 1)


    # gl = ax.gridlines(
    #     draw_labels=True,
    #     dms=False,
    #     x_inline=False,
    #     y_inline=False,
    #     alpha=alpha_gridlines,
    # )

    if tiles is not None:
        ax.add_image(tiles, scale)
    if cartopy:
        p = bias.plot(
            ax=ax,
            norm=norm,
            cmap=cmap,
            transform=map_proj,
            add_colorbar=False,
            **matplot_kwargs,
        )
    else:
        p = bias.plot(
            ax=ax,
            norm=norm,
            cmap=cmap,
            #transform=map_proj,
            add_colorbar=False,
            **matplot_kwargs,
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


def map_pearson(
    y_true: xr.DataArray,
    y_pred: xr.DataArray,
    figsize=(10, 10),
    label_1="wflow",
    label_2="LSTM",
    title="",
    color_norm="bounded",
    color_bounds=[-1, 1],
    color_bad=None,
    color_ticks=None,
    color_cmap="RdBu",
    matplot_kwargs=dict(alpha=1),
    alpha_gridlines=0.1,
    cartopy=False,
    tiles=QuadtreeTiles(),
    scale=13,
    map_extent=[],
    return_computation=False,
    corr_type="pearson",
    fig=None,
    ax= None
):
    # COMPUTE
    if corr_type == "pearson":
        out = compute_pearson(y_true, y_pred)
    else:
        raise NotImplementedError

    # MATPLOTLIB PARAMETERS
    cmap = plt.colormaps[color_cmap]
    if color_ticks is None:
        color_ticks = np.linspace(color_bounds[0], color_bounds[-1], 21)
    norm = set_norm(color_norm, color_bounds, color_ticks, cmap.N, clip=True)

    if color_bad is not None:
        cmap.set_bad(color_bad)

    # CARTOPY STUFF
    map_proj = ccrs.PlateCarree()
    minx, miny, maxx, maxy = return_map_extent(y_true, map_extent)

    # PLOT
    if fig is None:
        fig = plt.figure(figsize=figsize)
    if ax is None:
        if cartopy:
            map_proj = ccrs.PlateCarree()

            minx, miny, maxx, maxy = return_map_extent(y_true, map_extent)

            ax = fig.add_subplot(1, 1, 1, projection=map_proj)

            ax.set_extent([minx, maxx, miny, maxy], crs=map_proj)
            
        else:
            ax = fig.add_subplot(1, 1, 1)

    # gl = ax.gridlines(
    #     draw_labels=True,
    #     dms=False,
    #     x_inline=False,
    #     y_inline=False,
    #     alpha=alpha_gridlines,
    # )

    if tiles is not None:
        ax.add_image(tiles, scale)

    if cartopy:
        p = out.plot(
            ax=ax,
            norm=norm,
            cmap=cmap,
            transform=map_proj,
            add_colorbar=False,
            **matplot_kwargs,
        )
    else:
        p = out.plot(
            ax=ax,
            norm=norm,
            cmap=cmap,
            #transform=ccrs.PlateCarree(),
            add_colorbar=False,
            **matplot_kwargs,
        )

    fig.colorbar(
        p,
        ax=ax,
        shrink=0.5,
        label="correlation",
        ticks=color_ticks,
    )

    plt.title(title)

    if return_computation:
        return fig, ax, out
    else:
        return fig, ax


def map_variance(
    ds: xr.DataArray,
    figsize=(10, 10),
    title="",
    color_bad=None,
    color_cmap="viridis",
    matplot_kwargs=dict(alpha=1),
    alpha_gridlines=0.1,
    tiles=QuadtreeTiles(),
    scale=13,
    map_extent=[],
    return_computation=False,
    std=True,
):
    # COMPUTE
    if std:
        out = compute_variance(ds, std=True)
        col_lab = "standard deviation"
    else:
        out = compute_variance(ds)
        col_lab = "variance"

    # MATPLOTLIB PARAMETERS
    cmap = plt.colormaps[color_cmap]

    if color_bad is not None:
        cmap.set_bad(color_bad)

    # CARTOPY STUFF
    map_proj = ccrs.PlateCarree()
    minx, miny, maxx, maxy = return_map_extent(ds, map_extent)

    # PLOT
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
    ax.set_extent([minx, maxx, miny, maxy], crs=ccrs.PlateCarree())

    gl = ax.gridlines(
        draw_labels=True,
        dms=False,
        x_inline=False,
        y_inline=False,
        alpha=alpha_gridlines,
    )

    if tiles is not None:
        ax.add_image(tiles, scale)

    p = out.plot(
        ax=ax,
        cmap=cmap,
        transform=ccrs.PlateCarree(),
        add_colorbar=False,
        **matplot_kwargs,
    )

    fig.colorbar(p, ax=ax, shrink=0.5, label=col_lab)

    plt.title(title)

    if return_computation:
        return fig, ax, out
    else:
        return fig, ax
