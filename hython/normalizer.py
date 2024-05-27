import numpy as np

from dask.array.core import Array as DaskArray

from dask.array import expand_dims, nanmean, nanstd, nanmin, nanmax


STATS = {"minmax": [nanmin, nanmax], "standardize": [nanmean, nanstd]}

SCALE = {
    "minmax": lambda arr, axis, m1, m2: (arr - m1) / (m2 - m1),
    "standardize": lambda arr, axis, m1, m2: (arr - expand_dims(m1, axis=axis))
    / expand_dims(m2, axis=axis),
}

TYPE = {"time": 1, "spacetime": (0, 1), "space": 0}


class Normalizer:
    def __init__(
        self, method: str, type: str, dir_temporary: str = None, caching: bool = False
    ):
        self.method = method
        self.type = type
        self.stats_iscomputed = False
        self.dir_temporary = dir_temporary
        self.caching = caching

    def _get_axis(self):
        self.axis = TYPE.get(self.type, False)

    def compute_stats(self, arr):
        funcs = STATS.get(self.method, False)

        self._get_axis()

        print(self.axis)

        if funcs:
            return [f(arr, axis=self.axis).compute() for f in funcs]

    def _scale(self):
        return SCALE.get(self.method, None)

    def normalize(self, arr, fp=None):
        # get stats, if stats was not computed yet
        if self.stats_iscomputed is False:
            print("computing")
            self.computed_stats = self.compute_stats(arr)

        if fp is not None:
            print("loading precomputed stats")
            self.read_stats(fp)

        scale_func = self._scale()

        if scale_func is not None:
            return scale_func(arr, self.axis, *self.computed_stats)
        else:
            print("bla")

    def denormalize(self, arr, fp=None):
        if self.method == "standardize":
            if fp is not None:
                self.read_stats(fp)
                m, std = self.computed_stats
                return (arr * std) + m
            else:
                m, std = self.computed_stats
                return (arr * std) + m
        else:
            raise NotImplementedError()

    def save_stats(self, fp):
        for stat in self.computed_stats:
            np.save(fp, stat)

    def read_stats(self, fp):
        self.stats_iscomputed = True
        self.computed_stats = np.load(fp)
