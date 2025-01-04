import numpy as np
import matplotlib.pyplot as plt
import geopandas as gpd
import xarray as xr
from matplotlib.colors import BoundaryNorm, ListedColormap



def show_train_val_curve(epochs, target_variables, loss_label, loss_history, metric_history):
    lepochs = list(range(1, epochs + 1))

    rows = len(target_variables) + len(metric_history[f"train_{target_variables[0]}"][0])
    figsize = (12, rows*3)

    fig, axs = plt.subplots( rows, 1, figsize= figsize, sharex=True)

    axs[0].plot(lepochs, [i.detach().cpu().numpy() for i in loss_history['train']], marker='.', linestyle='-', color='b', label='Training')
    axs[0].plot(lepochs, [i.detach().cpu().numpy() for i in loss_history['val']], marker='.', linestyle='-', color='r', label='Validation')
    axs[0].set_title('Loss')
    axs[0].set_ylabel(loss_label)
    axs[0].grid(True)
    axs[0].legend(bbox_to_anchor=(1,1))

    # metrics
    for variable in target_variables:
        metrics = metric_history[f'train_{variable}'][0].keys()
        for i, m in enumerate(metrics):
            m_train = [im[m] for im in metric_history[f'train_{variable}']] 
            m_val = [im[m] for im in metric_history[f'val_{variable}']] 
            
            axs[i+1].plot(lepochs, m_train , marker='.', linestyle='-', color='b', label='Training')
            axs[i+1].plot(lepochs,m_val, marker='.', linestyle='-', color='r', label='Validation')
            axs[i+1].set_title(variable)
            axs[i+1].set_ylabel(m)
            axs[i+1].grid(True)
            axs[i+1].legend(bbox_to_anchor=(1,1))

    return fig, axs


def plot_sampler(
    da_bkg, meta, meta_valid, figsize=(10, 10), markersize=10, cmap="terrain"
):
    vv = da_bkg

    vv = vv.assign_coords({"gridcell": (("lat", "lon"), meta.idx_grid_2d)})

    vv = vv.assign_coords({"gridcell_valid": (("lat", "lon"), meta_valid.idx_grid_2d)})

    tmp = np.zeros(vv.shape).astype(np.bool_)
    for i in meta.idx_sampled_1d_nomissing:
        tmp[vv.gridcell == i] = True

    tmp_valid = np.zeros(vv.shape).astype(np.bool_)
    for i in meta_valid.idx_sampled_1d_nomissing:
        tmp_valid[vv.gridcell_valid == i] = True

    df = vv.where(tmp[::-1]).to_dataframe().dropna().reset_index()

    df_valid = vv.where(tmp_valid[::-1]).to_dataframe().dropna().reset_index()

    gdf = gpd.GeoDataFrame(
        df, geometry=gpd.points_from_xy(x=df.lon, y=df.lat), crs=4326
    )

    gdf_valid = gpd.GeoDataFrame(
        df_valid, geometry=gpd.points_from_xy(x=df_valid.lon, y=df_valid.lat), crs=4326
    )

    fig, ax = plt.subplots(1, 1, figsize=figsize)
    # p = da_bkg.plot(ax = ax, add_colorbar=True, alpha = 0.3, cmap=cmap, cbar_kwargs={"shrink":0.3})
    # p.colorbar.ax.set_ylabel('Elevation (m a.s.l.)', labelpad=10)
    # #p.colorbar.ax.set_label('standard deviation', rotation=270, labelpad=15)
    # gdf.plot(ax=ax, color="red", markersize= markersize, label="training")
    # gdf_valid.plot(ax=ax, color="black", markersize= markersize, label = "validation")
    # plt.legend(bbox_to_anchor=(1, 0.9), frameon = False)
    # plt.title("")
    # plt.gca().set_axis_off()
    # #ax.set_xlim([6, 7.5])
    # #ax.set_ylim([45.5, 46.5])

    cmap = plt.colormaps["terrain"]
    # cmap = ListedColormap(["black", "gold", "lightseagreen", "purple", "blue"])
    vmin = 0
    vmax = 5
    ticks = [
        -0.5,
        0.5,
        1.5,
        2.5,
        3.5,
        4.5,
    ]  # np.linspace(start=vmin + 0.5, stop=vmax, num=vmax+1)

    labels = {
        0: "0-500",
        1: "500-1000",
        2: "1000-1500",
        3: "1500-2000",
        4: "2000-2500",
        5: ">2500",
    }

    norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5, 4.5], ncolors=cmap.N, clip=True)

    norm.vmin = vmin
    norm.vmax = vmax
    p = da_bkg.plot.imshow(
        cmap=cmap,
        norm=norm,
        # vmin=vmin, vmax=vmax,
        add_colorbar=True,
        cbar_kwargs={"shrink": 0.3},
        ax=ax,
    )

    ilabels = [labels.get(i, "No Data") for i in range(vmax + 1)]
    p.colorbar.set_ticks(ticks, labels=ilabels)

    p.colorbar.ax.set_ylabel("Elevation bands (m a.s.l.)", rotation=270, labelpad=10)
    plt.axis("off")
    plt.title("")

    return fig, ax


def map_at_timesteps(
    y: xr.DataArray,
    yhat: xr.DataArray,
    dates=None,
    label_pred="LSTM",
    label_target="wflow",
):
    ts = dates if dates else y.time.dt.date.values

    for t in dates:
        fig, ax = plt.subplots(1, 2, figsize=(20, 15))
        fig.subplots_adjust(hspace=0.3)
        vmax = np.nanmax([yhat.sel(time=t), y.sel(time=t)])

        l1 = ax[0].imshow(yhat.sel(time=t), vmax=vmax)
        ax[0].set_title("LSTM", fontsize=28)
        fig.colorbar(l1, ax=ax[0], shrink=0.3)

        l2 = ax[1].imshow(y.sel(time=t), vmax=vmax)
        ax[1].set_title("wflow", fontsize=28)
        fig.colorbar(l2, ax=ax[1], shrink=0.3)
        fig.suptitle(t, y=0.8, fontsize=20, fontweight="bold")
        fig.tight_layout()


def ts_plot(
    y: xr.DataArray,
    yhat,
    smy,
    smyhat,
    precip,
    temp,
    lat=[],
    lon=[],
    label_1="wflow_sbm",
    label_2="LSTM",
    el=None,
    lc=None,
):
    time = y.time.values
    time2 = smy.time.values
    for ilat, ilon in zip(lat, lon):
        fig, ax = plt.subplots(
            4, 1, figsize=(20, 7), gridspec_kw={"height_ratios": [1, 2, 3, 3]}
        )
        # ax_dict = plt.figure(layout="constrained", figsize=(20,5)).subplot_mosaic(
        # """
        # A
        # """,
        # height_ratios=[1]
        # )
        iy = y.sel(lat=ilat, lon=ilon, method="nearest")
        iyhat = yhat.sel(lat=ilat, lon=ilon, method="nearest")

        smiy = smy.sel(lat=ilat, lon=ilon, method="nearest") * 10  # 10 mm
        smiyhat = smyhat.sel(lat=ilat, lon=ilon, method="nearest") * 10

        # ax_dict["A"].plot(time, iyhat, label = label_2)
        # ax_dict["A"].plot(time, iy, label= label_1)
        # ax_dict["A"].legend()
        ax[0].plot(time, temp, color="black", label="T")
        ax[0].set_ylabel("T (℃)", fontsize=16)
        ax[0].get_xaxis().set_visible(False)

        ax[1].bar(
            time, precip, 0.5, alpha=0.8, fill="black", color="black", label="precip"
        )
        ax[1].set_ylabel("Pr (mm)", fontsize=16)
        ax[1].get_xaxis().set_visible(False)

        ax[2].legend(loc="upper right")
        ax[2].plot(time, iyhat, label=label_2)
        ax[2].plot(time, iy, label=label_1, color="red")
        ax[2].set_ylabel("ET (mm)", fontsize=16)
        ax[2].legend(loc="upper right", frameon=False, fontsize=20)
        ax[2].get_xaxis().set_visible(False)

        # ax[3].legend(loc="upper right")
        ax[3].plot(time2, smiyhat, label=label_2, color="red")
        ax[3].plot(time2, smiy, label=label_1)
        ax[3].set_ylabel("SM (mm)", fontsize=16)
        ax[3].xaxis.set_tick_params(labelsize=16)
        # ax[1].legend(loc="upper right",frameon=False)

        # ax2 = ax[1].twinx()
        # ax2.bar(time,-precip, 0.5, alpha=0.8, fill="black", color="black", label="precip")
        # ax2.set_ylabel("Precipitation (mm)")
        # ax[1].set_legend(frameon=False)
        fig.text(0.13, 0.54, f"{el} (m a.s.l.)", size=20)
        fig.text(0.13, 0.44, f"{lc}", size=20)


def map_points(lat=[], lon=[], bkg_map=None):
    ax_dict = plt.figure(layout="constrained", figsize=(20, 6)).subplot_mosaic(
        """
    A
    """,
        height_ratios=[1],
    )

    df = gpd.GeoDataFrame([], geometry=gpd.points_from_xy(x=lon, y=lat))
    if bkg_map is not None:
        # bkg_map.plot(ax=ax_dict["A"], add_colorbar=False, cmap="terrain")
        from matplotlib.colors import ListedColormap

        cmap = ListedColormap(["black", "gold", "lightseagreen", "purple", "blue"])
        vmin = 0
        vmax = 4

        labels = {
            0: "Artificial surfaces",
            1: "Agricultural areas",
            2: "Forest and seminatural areas",
            3: "Wetlands",
            4: "Water bodies",
        }

        p = bkg_map.plot.imshow(
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            add_colorbar=True,
            cbar_kwargs={"shrink": 0.5},
            ax=ax_dict["A"],
        )

        ticks = np.linspace(start=vmin + 0.5, stop=vmax - 0.5, num=vmax + 1)
        ilabels = [labels.get(i, "No Data") for i in range(vmax + 1)]
        p.colorbar.set_ticks(ticks, labels=ilabels)

        p.colorbar.ax.set_ylabel("", rotation=270)
        plt.axis("off")
        plt.title("")
        # plt.gca().set_axis_off()
    else:
        y.mean("time").plot(ax=ax_dict["A"], add_colorbar=False)
    df.plot(ax=ax_dict["A"], markersize=20, color="red")
    plt.title("")


def ts_compare(
    y: xr.DataArray,
    yhat,
    lat=[],
    lon=[],
    label_1="wflow",
    label_2="LSTM",
    bkg_map=None,
    save=False,
):
    time = y.time.values
    for ilat, ilon in zip(lat, lon):
        ax_dict = plt.figure(layout="constrained", figsize=(20, 6)).subplot_mosaic(
            """
        AC
        BC
        """,
            width_ratios=[4, 1],
        )
        iy = y.sel(lat=ilat, lon=ilon, method="nearest")
        iyhat = yhat.sel(lat=ilat, lon=ilon, method="nearest")
        ax_dict["A"].plot(time, iyhat, label=label_2)
        ax_dict["A"].plot(time, iy, label=label_1)
        ax_dict["A"].legend()
        ax_dict["B"].scatter(iy, iyhat, s=1)
        xmin = np.nanmin(np.concatenate([iy, iyhat])) - 0.05
        xmax = np.nanmax(np.concatenate([iy, iyhat])) + 0.05
        ax_dict["B"].set_xlim(xmin, xmax)
        ax_dict["B"].set_ylim(xmin, xmax)
        ax_dict["B"].axline((0, 0), (1, 1), color="black", linestyle="dashed")
        ax_dict["B"].set_ylabel(label_2)
        ax_dict["B"].set_xlabel(label_1)
        df = gpd.GeoDataFrame([], geometry=gpd.points_from_xy(x=[ilon], y=[ilat]))
        if bkg_map is not None:
            bkg_map.plot(ax=ax_dict["C"], add_colorbar=False, cmap="terrain")
        else:
            y.mean("time").plot(ax=ax_dict["C"], add_colorbar=False)
        df.plot(ax=ax_dict["C"], markersize=20, color="red")
        plt.title(f"lat, lon:  ({ ilat}, {ilon})")
        if save:
            fig = plt.gcf()
            fig.savefig(save)


def show_cubelet_tile(
    dataset,
    n=10,
    dynamic_var_idx=0,
    static_var_idx=0,
    target_var_idx=0,
    seq_step_idx=10,
    data_idx=1,
    target_names=None,
    dynamic_names=None,
    static_names=None,
):
    idx = np.random.randint(0, len(dataset), n)
    if isinstance(seq_step_idx, list):
        tx, ts, ty = dataset[data_idx]
        for t in range(seq_step_idx[0], seq_step_idx[-1]):
            fig, axs = plt.subplots(1, 3, figsize=(10, 5))
            p1 = axs[0].imshow(tx[t, dynamic_var_idx, ...])  # L C H W
            if dynamic_names:
                title = f"{dynamic_names[dynamic_var_idx]} (forcing)"
            else:
                title = "forcing"
            axs[0].set_title(title)
            plt.colorbar(p1, fraction=0.046, pad=0.04)
            axs[0].axis("off")

            p2 = axs[1].imshow(ts[t, static_var_idx, ...])
            if static_names:
                title = f"{static_names[static_var_idx]} (static)"
            else:
                title = "static"
            axs[1].set_title(title)
            axs[1].axis("off")
            plt.colorbar(p2, fraction=0.046, pad=0.04)

            p3 = axs[2].imshow(ty[t, target_var_idx, ...])
            if target_names:
                title = f"{target_names[target_var_idx]} (target)"
            else:
                title = "target"
            axs[2].set_title(title)
            plt.colorbar(p3, fraction=0.046, pad=0.04)
            axs[2].axis("off")

    else:
        for i in idx:
            tx, ts, ty = dataset[i]
            fig, axs = plt.subplots(1, 3, figsize=(10, 5))
            p1 = axs[0].imshow(tx[seq_step_idx, dynamic_var_idx, ...])  # L C H W
            if dynamic_names:
                title = f"{dynamic_names[dynamic_var_idx]} (forcing)"
            else:
                title = "forcing"
            axs[0].set_title(title)
            plt.colorbar(p1, fraction=0.046, pad=0.04)
            axs[0].axis("off")

            p2 = axs[1].imshow(ts[seq_step_idx, static_var_idx, ...])
            if static_names:
                title = f"{static_names[static_var_idx]} (static)"
            else:
                title = "static"
            axs[1].set_title(title)
            axs[1].axis("off")
            plt.colorbar(p2, fraction=0.046, pad=0.04)

            p3 = axs[2].imshow(ty[seq_step_idx, target_var_idx, ...])
            if target_names:
                title = f"{target_names[target_var_idx]} (target)"
            else:
                title = "target"
            axs[2].set_title(title)
            plt.colorbar(p3, fraction=0.046, pad=0.04)
            axs[2].axis("off")
