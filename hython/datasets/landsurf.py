from . import *
import itertools
import logging
from hython.preprocessor import Preprocessor

LOGGER = logging.getLogger(__name__)


class PBMDataset(BaseDataset):
    """Dataset returns sequences with length equal to calibration period
    """

    def __init__(self, cfg, scaler, is_train=True, period="train", scale_ontraining=False):
        super().__init__()
        self.cfg = self.validate_config(cfg)
        self.scale_ontraining = scale_ontraining
        self.scaler = scaler

        self.downsampler = self.cfg[f"{period}_downsampler"]

        self.period = period
        self.period_range = slice(*cfg[f"{period}_temporal_range"])
        self.scaling_static_range = self.cfg.get("scaling_static_range")

        self.target_has_missing_dates = self.cfg.get("target_has_missing_dates")
        
        urls, xarray_kwargs = get_source_url(cfg)

        # load datasets
        data_dynamic = read_from_zarr(url=urls["dynamic_inputs"], chunks="auto", **xarray_kwargs)
        data_static = read_from_zarr(url=urls["static_inputs"], chunks="auto", **xarray_kwargs)
        data_target = read_from_zarr(url=urls["target_variables"], chunks="auto", **xarray_kwargs)
        #data_head_input = read_from_zarr(url=urls["static_parameter_inputs"], chunks="auto", **xarray_kwargs)
   
        head_model_input_list = []
        # load only non calibration params
        for i in self.cfg.head_model_inputs:
            if i is not None and i != "cal_param":
                head_model_input_list.extend(self.cfg.head_model_inputs[i])

        # select 
        self.xd = data_dynamic[self.to_list(cfg.dynamic_inputs)].sel(time=self.period_range)
        self.xs = data_static[self.to_list(cfg.static_inputs)]
        self.y = data_target[self.to_list(cfg.target_variables)].sel(time=self.period_range)

        #self.xp =  data_head_input[head_model_input_list]
        
        # subset dynamic inputs to the target timestep available
        
        if self.target_has_missing_dates is not None:
            self.xd = self.xd.sel(time=self.y.time)

        # TODO: ensure they are all float32
        # head_layer mask
        head_mask = read_from_zarr(url=urls["mask_variables"], chunks="auto", **xarray_kwargs)
        self.head_mask = head_mask[self.to_list(self.cfg.mask_variables)].to_array().any("variable")

        # == DATASET INDICES AND MASKING
        # target mask, observation
        if urls.get("target_variables_mask", None):
            self.target_mask = xr.open_dataset(urls["target_variables_mask"]).mask
        else:
            self.target_mask = self.y.isnull().all("time")[self.to_list(cfg.target_variables)[0]]
        
        # static mask, predictors
        if urls.get("static_inputs_mask", None):
            self.static_mask = read_from_zarr(url=urls["static_inputs_mask"], chunks="auto", **xarray_kwargs)[self.to_list(self.cfg.static_inputs_mask)[0]]
        else:
            self.static_mask = self.xs.isnull()[self.to_list(self.cfg.static_inputs)].to_array().any("variable")
        
        self.mask_wflow_missing = read_from_zarr(url=urls["mask_variables"])[self.to_list(self.cfg.mask_variables)]["mask_missing"]

        self.mask = self.target_mask | self.head_mask | self.static_mask    

            
        if not self.cfg.data_lazy_load: # loading in memory
            self.xd = self.xd.load()
            self.xs = self.xs.load()
            self.y = self.y.load()
            self.mask = self.mask.load()

        # Find indices of valid cells (not mask)
        if self.period != "test":
            self.coord_cells = np.argwhere(~self.mask.values) # indices of non-zero
        elif self.period == "test":
            # during test, don't need to mask so to reconstruct original dataset shape
            self.temp = np.ones(self.mask.shape).astype(bool)
            self.coord_cells = np.argwhere(self.temp)

        # Linear index, map integer sequence to tuple of (lat, lon) coordinates
        self.cell_linear_index = np.arange(0, len(self.coord_cells), 1)

        self.time_index = np.arange(0, len(self.xd.time.values), 1)

        # Reduce dataset size
        if self.downsampler is not None:
            self.cell_linear_index, self.time_index = self.downsampler.sampling_idx([self.cell_linear_index, self.time_index])

        # Generate dataset indices 
        #if self.cfg.downsampling_temporal_dynamic:
            # This assumes that the time index for sampling the sequences are generated at runtime.
            # The dataset returns the whole time series for each data sample
            # In this way it is possible to generate new random time indices every epoch to dynamically subsample the time domain. 
        self.coord_samples = self.coord_cells[self.cell_linear_index]

        # Normalize
        self.scaler.load_or_compute(
            self.xd, "dynamic_inputs", is_train, axes=("lat","lon", "time")
        )

        # IS_TRAIN = True
        self.scaler.load_or_compute(
            self.xs, "static_inputs", True, axes=("lat","lon")
        )

        self.scaler.load_or_compute(
            self.y, "target_variables", 
            True, # compute every dataset initialization
            axes=None, 
            reference = data_dynamic, 
            period_range=self.period_range
        )
        
        
        # self.scaler.load_or_compute( 
        #     self.xp, "head_model_inputs", is_train, axes=("lat","lon")
        # )

        self.xd = self.scaler.transform(self.xd, "dynamic_inputs")
        self.xs = self.scaler.transform(self.xs, "static_inputs")
        # transform target variable to reference statistics
        self.y = self.scaler.transform(self.y, "target_variables")
        # standardise transformed target variable to match emulator
        # that was trained with standardised reference
        #self.y = (self.y - self.y.mean("time")) / self.y.std("time")

        # self.xp = self.scaler.transform(self.xp, "head_model_inputs").compute()
        
        if is_train: # write if train
            if not self.scaler.use_cached: # write if not reading from cache
                self.scaler.write("dynamic_inputs")
                self.scaler.write("static_inputs")
                self.scaler.write("target_variables")
                # self.scaler.write("head_model_inputs")
            else: # if reading from cache
                if self.scaler.flag_stats_computed: # if stats were not found in cache
                    self.scaler.write("dynamic_inputs")
                    self.scaler.write("static_inputs")
                    self.scaler.write("target_variables")
                    # self.scaler.write("head_model_inputs")

    def __len__(self):
        return len((range(len(self.coord_samples))))

    def __getitem__(self, index):
        idx_lat, idx_lon = self.coord_samples[index]

        ds_pixel_dynamic = self.xd.isel(lat=idx_lat, lon=idx_lon) # lat, lon, time -> time
        ds_pixel_target = self.y.isel(lat=idx_lat, lon=idx_lon)
        ds_pixel_static = self.xs.isel(lat=idx_lat, lon=idx_lon)
        # ds_pixel_head_input = self.xp.isel(lat=idx_lat, lon=idx_lon)

        ds_pixel_dynamic = ds_pixel_dynamic.to_array().transpose("time", "variable") # time -> time, feature
        ds_pixel_target = ds_pixel_target.to_array().transpose("time", "variable") # time -> time, feature

        ds_pixel_static = ds_pixel_static.to_array()
        ds_pixel_head_input = ds_pixel_head_input.to_array()
        
        # TODO: remove call to float
        xd  = torch.tensor(ds_pixel_dynamic.values).float()
        xs = torch.tensor(ds_pixel_static.values).float()
        y = torch.tensor(ds_pixel_target.values).float()
        # xp = torch.tensor(ds_pixel_head_input.values).float()

        return {"xd": xd, "xs": xs, "y": y} #, "xp":xp}
