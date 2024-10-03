import numpy as np
import xarray as xr
from dask.array import expand_dims, nanmean, nanstd, nanmin, nanmax


FUNCS = {"minmax": [nanmin, nanmax], "standardize": [nanmean, nanstd]}

SCALER = {
    "minmax": lambda arr, axis, m1, m2: (arr - np.expand_dims(m1, axis=axis))
    / (np.expand_dims(m2, axis=axis) - np.expand_dims(m1, axis=axis)),
    "standardize": lambda arr, axis, m1, m2: (arr - expand_dims(m1, axis=axis))
    / expand_dims(m2, axis=axis),
}

SCALER_XARRAY = {
    "minmax": lambda arr, axis, m1, m2: (arr - m1) / (m2 - m1),
    "standardize": lambda arr, axis, m1, m2: (arr - m1) / m2,
}


DENORM_XARRAY = {
    "standardize": lambda arr, axis, m1, m2: (arr * m2) + m1,
    "minmax": lambda arr, axis, m1, m2: (arr * (m2 - m1)) + m1
}

DENORM = {
    "standardize": lambda arr, axis, m1, m2: (arr * expand_dims(m2, axis=axis))
    + expand_dims(m1, axis=axis)
}

TYPE = {
    "NTC": {"space": 0, "time": 1, "spacetime": (0, 1)},  # N T C
    "NCTHW": {"space": (1, 2), "time": 1, "spacetime": (1, 2, 3)},  # C T H W
    "NLCHW": {"space": (2, 3), "time": 0, "spacetime": (0, 2, 3)},
    "xarray_dataset": {
        "space": ("lat", "lon"),
        "time": ("time"),
        "spacetime": ("time", "lat", "lon"),
    },
}


class Normalizer:
    def __init__(
        self,
        method: str,
        type: str,
        axis_order: str,
        dask_compute: bool = False,
        save_stats: str = None,
    ):
        self.method = method
        self.axis_order = axis_order
        self.type = type
        self.dask_compute = dask_compute
        self.save_stats = save_stats

        self.stats_iscomputed = False

        self._set_axis()

    def _set_axis(self):
        axis_order = TYPE.get(self.axis_order)

        self.axis = axis_order.get(self.type, None)

    def _get_scaler(self):
        if "xarray" in self.axis_order:
            scaler = SCALER_XARRAY.get(self.method, None)
        else:
            scaler = SCALER.get(self.method, None)

        if scaler is None:
            raise NameError(f"Scaler for {self.method} does not exists")
        else:
            return scaler

    def _get_funcs(self):
        funcs = FUNCS.get(self.method, False)
        if funcs is None:
            raise NameError(f"{self.method} does not exists")
        else:
            return funcs

    def compute_stats(self, arr):
        if "xarray" in self.axis_order:
            if self.method == "standardize":
                if self.dask_compute:
                    self.computed_stats = [
                        arr.mean(self.axis).compute(),
                        arr.std(self.axis).compute(),
                    ]
                else:
                    self.computed_stats = [arr.mean(self.axis), arr.std(self.axis)]
            else:
                if self.dask_compute:
                    self.computed_stats = [
                        arr.min(self.axis).compute(),
                        arr.max(self.axis).compute(),
                    ]
                else:
                    self.computed_stats = [arr.min(self.axis), arr.max(self.axis)]

        else:
            funcs = self._get_funcs()
            self.computed_stats = [f(arr, axis=self.axis).compute() for f in funcs]

        if self.save_stats is not None and self.stats_iscomputed is False:
            self.write_stats(self.save_stats)

        self.stats_iscomputed = True

    def normalize(self, arr, read_from=None, write_to=None):
        scale_func = self._get_scaler()

        if read_from is not None:
            self.read_stats(read_from)

        if self.stats_iscomputed is False:
            self.compute_stats(arr)

        if write_to is not None:
            self.write_stats(write_to)

        if self.dask_compute:
            return scale_func(arr, self.axis, *self.computed_stats).compute()
        else:
            return scale_func(arr, self.axis, *self.computed_stats)

    def denormalize(self, arr, fp=None):
        if self.method == "standardize":
            if "xarray" in self.axis_order:
                func = DENORM_XARRAY.get(self.method)
            else:
                func = DENORM.get(self.method)

            if fp is not None:
                self.read_stats(fp)
                m, std = self.computed_stats
                return func(arr, self.axis, m, std)
            else:
                m, std = self.computed_stats
                if "xarray" in self.axis_order:
                    if isinstance(arr, np.ndarray):
                        std = std.to_dataarray().values
                        m = m.to_dataarray().values
                    else:
                        pass
                    return func(arr, self.axis, m, std)
                else:
                    return func(arr, self.axis, m, std)
        else:
            if "xarray" in self.axis_order:
                func = DENORM_XARRAY.get(self.method)
            else:
                func = DENORM.get(self.method)
            if fp is not None:
                self.read_stats(fp)
                m1, m2 = self.computed_stats
                return func(arr, self.axis, m1, m2)
            else:
                m1, m2 = self.computed_stats
                if "xarray" in self.axis_order:
                    if isinstance(arr, np.ndarray):
                        m2 = m2.to_dataarray().values 
                        m1 = m1.to_dataarray().values
                    else:
                        pass
                    return func(arr, self.axis, m1, m2)
                else:
                    return func(arr, self.axis, m1, m2)

    def write_stats(self, fp):
        print(f"write stats to {fp}")
        if "xarray" in self.axis_order:
            xarr = xr.DataArray(["m1", "m2"], coords=[("stats", ["m1", "m2"])])
            ds = xr.concat(self.computed_stats, dim=xarr)
            ds.to_netcdf(fp, mode="w")
        else:
            np.save(fp, self.computed_stats)

    def read_stats(self, fp):
        print(f"read from {fp}")
        self.stats_iscomputed = True
        if "xarray" in self.axis_order:
            ds = xr.open_dataset(fp)
            ds.close()  # closing so that I can overwrite it with write stats
            self.computed_stats = [
                ds.sel(stats="m1", drop=True),
                ds.sel(stats="m2", drop=True),
            ]

        else:
            self.computed_stats = np.load(fp)

    def get_stats(self):
        return self.computed_stats
