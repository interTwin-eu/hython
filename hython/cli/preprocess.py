"""Description

"""

from jsonargparse import CLI  # type: ignore
import xarray as xr
import numpy as np
from hython.utils import build_mask_dataarray, write_to_zarr
from numcodecs.abc import Codec
from hython.preprocessor import *


def train(
    # Specifi wflow model folder containing dynamic, static parameters and wflow outputs
    dir_wflow_model: str,
    # Directory of all different wflow models
    dir_input: str,
    # Input and target files
    file_dynamic: str,
    file_static: str,
    file_target: str,
    # Outputs
    dir_output: str,
    file_output: str,
    compressor: Codec,  # compressor for zarr output
    # Selected parameters and variables
    static_names: list,
    dynamic_names: list,
    target_names: list,
    # Dimensions filters
    soil_layers: list,
    temporal_range: list,
    # Masks
    mask_from_static: list,  # layers from static that should be masked out
    rename_mask: list,  # rename the layers
):
    dynamics = xr.open_dataset(f"{dir_input}/{dir_wflow_model}/{file_dynamic}")
    statics = xr.open_dataset(f"{dir_input}/{dir_wflow_model}/{file_static}")
    targets = xr.open_dataset(f"{dir_input}/{dir_wflow_model}/{file_target}")

    try:
        dynamics = dynamics.rename({"latitude": "lat", "longitude": "lon"})
        statics = statics.rename({"latitude": "lat", "longitude": "lon"})
    except:
        pass

    targets = targets.isel(lat=slice(None, None, -1))

    temporal_range = slice(*temporal_range)

    # === MASKING & FILTERING ==========================

    # filter soil layers
    if len(soil_layers) > 0:
        if len(soil_layers) == 1:
            statics = statics.sel(layer=soil_layers).squeeze("layer")
            targets = targets.sel(layer=soil_layers).squeeze("layer")

        else:
            raise NotImplementedError(
                "Preprocessing multiple soil layers not yet implemented"
            )

    # masking, TODO: Improve the logic.
    masks = []

    for i, mask in enumerate(mask_from_static):
        if i == 0:
            masks.append(np.isnan(statics[mask]).rename(rename_mask[i]))
        else:
            masks.append((statics[mask] > 0).astype(np.bool_).rename(rename_mask[i]))

    masks = build_mask_dataarray(masks, names=rename_mask)

    # select variables
    if "all_variables" not in static_names:
        statics = statics[static_names]
    else:
        # drop some of the variable with time dimensions (e.g. LAI) TODO: think about allowing that
        statics = statics.drop_dims("time")

    dynamics = dynamics[dynamic_names]
    targets = targets[target_names]

    # === RESHAPING ================================

    Xd = reshape(dynamics, type="dynamic", return_type="xarray")
    Xs = reshape(statics, type="static", return_type="xarray")
    Y = reshape(targets, type="target", return_type="xarray")

    # attrs to pass to output
    ATTRS = {
        "shape_label": masks.isel(mask_layer=0).dims,
        "shape": masks.isel(mask_layer=0).shape,
    }

    # remove as it cause serialization issues
    Xd.attrs.pop("_FillValue", None)

    # === WRITING TO DISK =====================================

    file_output = f"{dir_output}/{file_output}"

    write_to_zarr(
        Xd,
        url=file_output,
        group="xd",
        storage_options={"compressor": compressor},
        chunks="auto",
        append_on_time=True,
        multi_index="gridcell",
        append_attrs=ATTRS,
        overwrite=True,
    )

    write_to_zarr(
        Y,
        url=file_output,
        group="y",
        storage_options={"compressor": compressor},
        chunks="auto",
        append_on_time=True,
        multi_index="gridcell",
        append_attrs=ATTRS,
    )

    write_to_zarr(
        Xs,
        url=file_output,
        group="xs",
        storage_options={"compressor": compressor},
        chunks="auto",
        multi_index="gridcell",
        append_attrs=ATTRS,
    )

    write_to_zarr(
        masks,
        url=file_output,
        group="mask",
        storage_options={"compressor": compressor},
        overwrite=True,
    )


if __name__ == "__main__":
    CLI(as_positional=False)
