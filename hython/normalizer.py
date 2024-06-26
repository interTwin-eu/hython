import numpy as np
from dask.array import expand_dims, nanmean, nanstd, nanmin, nanmax


FUNCS = {"minmax": [nanmin, nanmax], 
         "standardize": [nanmean, nanstd]}

SCALER = {
    "minmax": lambda arr, axis, m1, m2: (arr - np.expand_dims(m1, axis=axis)) / (np.expand_dims(m2, axis=axis)  - np.expand_dims(m1, axis=axis)),
    "standardize": lambda arr, axis, m1, m2: (arr - expand_dims(m1, axis=axis)) / expand_dims(m2, axis=axis)
    }


DENORM = {
    "standardize": lambda arr, axis, m1, m2:  (arr * expand_dims(m2, axis=axis)) + expand_dims(m1, axis=axis)
}

TYPE = {"1D":{"space": 0, "time": 1, "spacetime": (0, 1)} , # N T C
        "2D":{"space": (1, 2), "time": 1, "spacetime": (1, 2, 3)} } # C T H W


class Normalizer:
    def __init__(
        self, method: str,  type: str, shape:str, save_stats: bool = False
    ):
        self.method = method
        self.shape = shape 
        self.type = type
        self.save_stats = save_stats

        self.stats_iscomputed = False

        self._set_axis()

    def _set_axis(self):
        self.axis = TYPE.get(self.shape).get(self.type, False)
        #self.axis = TYPE.get(self.type, False)

    def _get_scaler(self):
        scaler = SCALER.get(self.method, None)
        if scaler is None: raise NameError(f"Scaler for {self.method} does not exists") 
        else: return scaler
    
    def _get_funcs(self):
        funcs = FUNCS.get(self.method, False)
        if funcs is None: raise NameError(f"{self.method} does not exists") 
        else: return funcs 

    def compute_stats(self, arr):
        
        print("compute stats")

        funcs = self._get_funcs()
        
        self.stats_iscomputed = True

        self.computed_stats =  [f(arr, axis=self.axis).compute() for f in funcs]


    def normalize(self, arr, read_from=None, write_to = None):

        scale_func = self._get_scaler()
        
        if read_from is not None:
            self.read_stats(read_from)

        if self.stats_iscomputed is False:
            self.compute_stats(arr)

        if write_to is not None:
            self.write_stats(write_to)

        return scale_func(arr, self.axis, *self.computed_stats)

    def denormalize(self, arr, fp=None):
        if self.method == "standardize":

            func = DENORM.get(self.method)
            
            if fp is not None:
                self.read_stats(fp)
                m, std = self.computed_stats
                return func(arr, self.axis, m, std)
            else:

                m, std = self.computed_stats
                return func(arr, self.axis, m, std)
        else:
            raise NotImplementedError()

    def write_stats(self, fp):
        print("save stats")
        np.save(fp, self.computed_stats)

    def read_stats(self, fp):
        print("read stats")
        self.stats_iscomputed = True
        self.computed_stats = np.load(fp)

    def get_stats(self):
        return self.computed_stats
