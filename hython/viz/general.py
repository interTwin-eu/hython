import numpy as np
import matplotlib.pyplot as plt
import geopandas as gpd
import xarray as xr
from matplotlib.colors import BoundaryNorm, ListedColormap

# assumes a shape 
def plot_spatio_temporal_data_cluster(spatio_temporal_data, categorical_data, num_points=10, method='random_sampling', cluster_name=None, seed = None, **mpl_kwargs):
    """
    Plot spatio-temporal data based on categorical values.
    
    Parameters:
        spatio_temporal_data (xarray.DataArray): The spatio-temporal data to be plotted.
        categorical_data (xarray.DataArray): The categorical data with cluster values.
        num_points (int): The number of random points to sample from each category, the default is 10 (if method='random').
        method (str): Method of selecting points. Options are 'random_sampling' or 'summary'.
    """
    # Initialize dictionary to store sampled points for each category
    samples = {}
    time = spatio_temporal_data.time.to_numpy()

    if seed is not None:
        np.random.seed(seed)
    
    categories = np.unique(categorical_data.values[~np.isnan(categorical_data.values)])
    
    print(f"cat: {categories}")
    if cluster_name is not None:
        cat = {n:c for c,n in zip(categories,cluster_name)}
    for category in categories:
        # Get the indices for the current category
        category_mask = categorical_data == category
        category_points = np.argwhere(category_mask.values)  # Get the indices where the category matches

        if method == 'random_sampling':
            # Randomly select points from the available ones
            if len(category_points) > num_points:
                selected_indices = np.random.choice(len(category_points), size=num_points, replace=False)
            else:
                selected_indices = np.arange(len(category_points))  # If less than num_points available, take all

            selected_points = category_points[selected_indices]

            # Extract the time series data for the selected points
            category_sample = []
            for point in selected_points:
                #import pdb;pdb.set_trace()
                x_idx, y_idx = point[0], point[1]
                category_sample.append(spatio_temporal_data[:, x_idx, y_idx].values)  # Extract time series for the point

            samples[category] = np.array(category_sample)

            if cluster_name is not None:
                cat.update({category:selected_points})
        
        elif method == 'summary':
            # Use the category mask to directly extract the spatio-temporal data points for the current category
            masked_data = spatio_temporal_data.where(category_mask, drop=True)  # This will give you the data with NaNs where mask is False
            
            # Reshape data to ensure (n_points, time) where n_points = x * y
            all_points = masked_data.stack(points=('lon', 'lat')).values  # Stack spatial dimensions into one dimension
        
            # Filter out rows where all time steps are NaN, keeping points with at least some valid data
            valid_points_mask = ~np.isnan(all_points).all(axis=0)  # Find columns where there is valid data
            all_points = all_points[:, valid_points_mask]
        
            if all_points.shape[1] == 0:
                raise ValueError("No valid points found in the selected category.")
        
            # Calculate quantiles and other statistics using numpy's built-in functions
            quantiles = {
                '10th': np.percentile(all_points, 10, axis=1),
                '90th': np.percentile(all_points, 90, axis=1),
                'min': np.nanmin(all_points, axis=1),
                'max': np.nanmax(all_points, axis=1),
                'median': np.nanmedian(all_points, axis=1)
            }
        
            samples[category] = quantiles

    # Create a figure with subplots (1 row per category)
    num_categories = len(samples.keys())
    if num_categories == 0:
        raise ValueError("No valid categories found in the data to plot.")
    
    fig, axs = plt.subplots(num_categories, 1 , sharex=True, **mpl_kwargs)

    # If there's only one category, axs is not a list, so we need to handle it
    if num_categories == 1:
        axs = [axs]

    for i, (category, category_samples) in enumerate(samples.items()):
        ax = axs[i]

        if method == 'random_sampling':
            for sample in category_samples:
                ax.plot(time, sample)

            ax.set_ylabel('Data Value')
            ax.set_title(f'{cluster_name if cluster_name else "Cluster"}: {category}')
        
        elif method == 'summary':
            ax.plot(time, category_samples['10th'], label='10th Percentile', color='blue', linestyle='--')
            ax.plot(time, category_samples['90th'], label='90th Percentile', color='red', linestyle='--')
            ax.plot(time, category_samples['median'], label='Median', color='green')
            ax.fill_between(time, category_samples['min'], category_samples['max'], color='gray', alpha=0.2, label='Min-Max Range')

            ax.set_ylabel('Data Value')
            ax.set_title(f'{cluster_name if cluster_name else "Cluster"}: {category}')
            ax.legend(loc='upper right')

    # Set common x-label for all subplots
    plt.xlabel('Time')
    plt.tight_layout()  # Adjust layout to prevent overlap
    plt.show()
    
    return fig, axs, cat

def show_train_val_curve(
    epochs, target_variables, loss_label, loss_history, metric_history
):
    lepochs = list(range(1, epochs + 1))

    rows = len(target_variables) * len(
        metric_history[f"train_{target_variables[0]}"][0]
    ) + 1
    figsize = (12, rows * 3)

    fig, axs = plt.subplots(rows, 1, figsize=figsize, sharex=True)

    axs[0].plot(
        lepochs,
        [i.detach().cpu().numpy() for i in loss_history["train"]],
        marker=".",
        linestyle="-",
        color="b",
        label="Training",
    )
    axs[0].plot(
        lepochs,
        [i.detach().cpu().numpy() for i in loss_history["val"]],
        marker=".",
        linestyle="-",
        color="r",
        label="Validation",
    )
    axs[0].set_title("Loss")
    axs[0].set_ylabel(loss_label)
    axs[0].grid(True)
    axs[0].legend(bbox_to_anchor=(1, 1))

    # metrics
    ii = 0
    for variable in target_variables:
        metrics = metric_history[f"train_{variable}"][0].keys()
        for i, m in enumerate(metrics):
            m_train = [im[m] for im in metric_history[f"train_{variable}"]]
            m_val = [im[m] for im in metric_history[f"val_{variable}"]]

            axs[ii + 1].plot(
                lepochs, m_train, marker=".", linestyle="-", color="b", label="Training"
            )
            axs[ii + 1].plot(
                lepochs, m_val, marker=".", linestyle="-", color="r", label="Validation"
            )
            axs[ii + 1].set_title(variable)
            axs[ii + 1].set_ylabel(m)
            axs[ii + 1].grid(True)
            axs[ii + 1].legend(bbox_to_anchor=(1, 1))

            ii += 1

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
    precip = None,
    lat=[],
    lon=[],
    label_1="wflow",
    label_2="LSTM",
    bkg_map=None,
    save=False,
    ds_meta = None
):
               
    if ds_meta:
        el = ds_meta["wflow_dem"]
        lc = ds_meta["wflow_landuse"] 

    iel= ""
    ilc= ""
    time = y.time.values
    for ilat, ilon in zip(lat, lon):

        if ds_meta:
            iel = int(el.sel(lat=ilat, lon=ilon, method="nearest").item(0))
            ilc = int(lc.sel(lat=ilat, lon=ilon, method="nearest").item(0))

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
        if precip is not None:
            iprecip = precip.sel(lat=ilat, lon=ilon, method="nearest")
            ax2 = ax_dict["A"].twinx()
            ax2.bar(time, -iprecip, label="precip", color="black", alpha=0.5)
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
        plt.title(f"lat, lon:  ({ ilat}, {ilon}), el: {iel}, lc: {ilc}")
        plt.show()
        if save:
            fig = plt.gcf()
            fig.savefig(save)
    return ax_dict


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




def plot_distr(true: xr.DataArray, pred: xr.DataArray, bins:int=100, title = "ssm", xlabel="ssm [mm/mm]"):
    f, ax = plt.subplots(figsize=(8,5))
    true.plot(bins=bins, color="blue", ax=ax, label="true")
    pred.plot(bins=bins, alpha=0.5, color="orange", ax=ax, label="pred")   
    plt.legend()
    plt.title(title)
    plt.xlabel(xlabel)
    return f, ax