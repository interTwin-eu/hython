import xarray as xr
from numpy.typing import NDArray
from dask.array.core import Array as DaskArray
import zarr
import cf_xarray as cfxr

def write_to_zarr(
    arr: DaskArray | xr.DataArray,
    url,
    group=None,
    storage_options={},
    overwrite=True,
    chunks="auto",
    clear_zarr_storage=False,
    append_on_time=False,
    time_chunk_size=200,
    multi_index=None,
    append_attrs: dict = None,
):
    if isinstance(arr, DaskArray):
        arr = arr.rechunk(chunks=chunks)
        arr.to_zarr(
            url=url,
            storage_options=storage_options,
            overwrite=overwrite,
            component=group,
        )

    if isinstance(arr, xr.DataArray) or isinstance(arr, xr.Dataset):
        original_dataarray_attrs = arr.attrs

        if overwrite:
            overwrite = "w"
        else:
            overwrite = "r"
        if chunks:
            arr = arr.chunk(chunks=chunks)
        
        if isinstance(arr, xr.DataArray):
            shape = arr.shape
        else:
            shape = list(arr.sizes.values())


        if append_attrs:
            arr.attrs.update(append_attrs)

        if multi_index:
            arr = arr.to_dataset(name=group)
            arr = cfxr.encode_multi_index_as_compress(arr, multi_index)

        if append_on_time:
            fs_store = zarr.storage.FSStore(
                url, storage_options=storage_options, mode=overwrite
            )

            if clear_zarr_storage:
                fs_store.clear()


            # initialize
            init = arr.isel(time=slice(0, time_chunk_size)).persist()
            # init[group].attrs.clear()

            init.to_zarr(fs_store, consolidated=True, group=group, mode=overwrite)


            for t in range(time_chunk_size, shape[1], time_chunk_size):  # append time
                arr.isel(time=slice(t, t + time_chunk_size)).to_zarr(
                    fs_store, append_dim="time", consolidated=True, group=group
                )
        else:
            arr.to_zarr(
                store=url, storage_options=storage_options, mode=overwrite, group=group
            )

def read_from_zarr(url, group=None, multi_index=None, engine="xarray"):
    if engine == "xarray":
        ds = xr.open_dataset(url, group=group, engine="zarr")
        if multi_index:
            ds = cfxr.decode_compress_to_multi_index(ds, multi_index)
        return ds