"""Wflow_sbm emulators"""
from . import *
import itertools
import logging
from hython.preprocessor import Preprocessor

LOGGER = logging.getLogger(__name__)


class WflowSBM_HPC(BaseDataset):
    def __init__(
        self, cfg, scaler, is_train=True, period="train"
    ):
        self.scaler = scaler
        self.seq_len = cfg.seq_length
        self.cfg = self.validate_config(cfg)

        self.preprocessor = Preprocessor(cfg)

        self.downsampler = self.cfg[f"{period}_downsampler"]

        self.period = period
        self.period_range = slice(*cfg[f"{period}_temporal_range"])

        self.target_has_missing_dates = self.cfg.get("target_has_missing_dates", False)

        urls, xarray_kwargs = get_source_url(cfg)

        self.scaling_static_range = self.cfg.get("scaling_static_range")
        
        data_dynamic = read_from_zarr(url=urls["dynamic_inputs"], chunks="auto", **xarray_kwargs).sel(time=self.period_range)
        data_static = read_from_zarr(url=urls["static_inputs"], chunks="auto", **xarray_kwargs)
 
        self.xd = data_dynamic[self.to_list(cfg.dynamic_inputs)] # list comprehension handle omegaconf lists
        self.xs = data_static[self.to_list(cfg.static_inputs)]
        self.y = data_dynamic[self.to_list(cfg.target_variables)]

        # subset dynamic inputs to the target timestep available
        if self.target_has_missing_dates:
            self.xd = self.xd.sel(time=self.y.time)

        if not self.cfg.data_lazy_load: # loading in memory
            self.xd = self.xd.load()
            self.xs = self.xs.load()
            self.y = self.y.load()

        # == DATASET INDICES AND MASKING

        if self.cfg.mask_variables is not None and self.period != "test":
            # During training and validation remove cells marked as mask.
            self.mask = data_static[self.to_list(self.cfg.mask_variables)].to_array().any("variable")
            self.cell_coords = np.argwhere(~self.mask.values)
        elif self.period == "test": 
            # No masking during testing, however computing mask is still useful.
            self.mask = data_static[self.to_list(self.cfg.mask_variables)].to_array().any("variable")
            shape = list(self.xs.dims.values())
            self.cell_coords =  np.argwhere(np.ones(shape).astype(bool))

        # Compute cell (spatial) index 
        self.cell_linear_index  = np.arange(0, len(self.cell_coords ), 1)
        
        # Compute sequence (temporal) index
        # Each cell has a time series of equal length, so the sequence index is the same for every cell
        if self.period == "test":
            self.time_index = np.arange(0, len(self.xd.time.values), 1)
        else:
            self.time_index = np.arange(0, len(self.xd.time.values) - self.seq_len, 1)
        
        # == DOWNSAMPLING

        # Downsample spatial and temporal indices based on rule
        if self.downsampler is not None:
            self.cell_linear_index , self.time_index = self.downsampler.sampling_idx([self.cell_linear_index , self.time_index])

        if self.period == "test":
            self.spacetime_index = self.cell_linear_index
        else:
            self.spacetime_index = list(itertools.product(*[
                                                        self.cell_coords[self.cell_linear_index ].tolist(), 
                                                        self.time_index.tolist() 
                                                        ]))  

            self.spacetime_index = unnest(self.spacetime_index)

        # == SOME USEFUL PARAMETERS
        self.lat_size = len(self.xd.lat)
        self.lon_size = len(self.xd.lon)
        self.time_size = len(self.xd.time)
        self.dynamic_coords = self.xd.coords
        self.static_coords = self.xs.coords


        #  === PREPROCESS/TRANSFORM VARIABLES

        if self.cfg.get("preprocessor") is not None:
            self.xs = self.preprocessor.process(self.xs, "static_inputs")
            self.xd = self.preprocessor.process(self.xd, "dynamic_inputs")
            self.y = self.preprocessor.process(self.y, "target_variables")
        
        # == SCALING 

        self.scaler.load_or_compute(
            self.xd, "dynamic_inputs", is_train, axes=("lat","lon", "time")
        )

        self.scaler.load_or_compute(
            self.xs, "static_inputs", is_train, axes=("lat","lon")
        )

        self.scaler.load_or_compute(
            self.y, "target_variables", is_train, axes=("lat", "lon", "time")
        )

        self.xd = self.scaler.transform(self.xd, "dynamic_inputs")


        self.y = self.scaler.transform(self.y, "target_variables")


        self.xs = self.scaler.transform(self.xs, "static_inputs")
        



        # == WRITE SCALING STATS

        if is_train: # write if train
            if not self.scaler.use_cached: # write if not reading from cache
                self.scaler.write("dynamic_inputs")
                self.scaler.write("static_inputs")
                self.scaler.write("target_variables")
            else: # if reading from cache
                if self.scaler.flag_stats_computed: # if stats were not found in cache
                    self.scaler.write("dynamic_inputs")
                    self.scaler.write("static_inputs")
                    self.scaler.write("target_variables")


        # Pre-compute static data once
        self.static_tensor = torch.tensor(self.xs.to_array().values).float()
        
        # Pre-compute dynamic data shapes once
        self.dynamic_shape = self.xd[self.cfg.dynamic_inputs[0]].shape
        self.target_shape = self.y[self.cfg.target_variables[0]].shape
    

        # Convert to tensors once during initialization
        self.xd = self.xd.to_stacked_array(
            new_dim="feat", sample_dims=["time", "lat", "lon"]
        ).transpose("time", "feat", "lat", "lon").astype("float32")
        self.y = self.y.to_stacked_array(
            new_dim="feat", sample_dims=["time", "lat", "lon"]
        ).transpose("time", "feat", "lat", "lon").astype("float32")
        self.xs = self.xs.to_stacked_array(
            new_dim="feat", sample_dims=["lat", "lon"]
        ).transpose("feat", "lat", "lon").astype("float32")

        # Pre-process once
        if not self.cfg.data_lazy_load:  # Only if we're not doing lazy loading
            # Convert xarray to pre-processed tensors
            self.xd = torch.from_numpy(
                self.xd.transpose("time", "feat", "lat", "lon").values
            )
            self.y = torch.from_numpy(
                self.y.transpose("time", "feat", "lat", "lon").values
            )
            self.xs = torch.from_numpy(
                self.xs.transpose("feat", "lat", "lon").values
            )

    def __len__(self):
        return len(self.spacetime_index)

    def __getitem__(self, index):
        
        if self.period == "test":
            idx_lat, idx_lon = self.cell_coords[index]

            xd = self.xd[:,:,idx_lat, idx_lon]

            y = self.y[:,:,idx_lat, idx_lon]
            
            xs = self.xs[:, idx_lat, idx_lon]
        else:
            idx_lat, idx_lon, idx_time = self.spacetime_index[index]

            xd = self.xd[idx_time:idx_time + self.seq_len,:,idx_lat, idx_lon]

            y = self.y[idx_time:idx_time + self.seq_len,:,idx_lat, idx_lon]
            
            xs = self.xs[:, idx_lat, idx_lon]

        return {"xd": xd, "xs": xs, "y": y}

class WflowSBM_Pool(BaseDataset):
    """Training dataset for the multicycle pool archive (H1, H2, H3, H4).

    `WflowSBM_HPC` reads a full lat/lon map with one parameter set per cell.
    This class reads the accumulating pool archive instead: a stacked `cell`
    axis of runs x pool_size rows, one row per (run, base cell). The two cannot
    share a code path - the sample axis, the scaler axes and the reshape order
    all differ - so the old class is left untouched for the full-map runs that
    still use it.

    A sample is (run, base cell, time window): same cell, same weather,
    different parameters. That difference is what shows how the parameters act.
    """

    def __init__(
        self, cfg, scaler, is_train=True, period="train"
    ):
        self.scaler = scaler
        self.seq_len = cfg.seq_length
        self.cfg = self.validate_config(cfg)

        self.preprocessor = Preprocessor(cfg)

        self.downsampler = self.cfg[f"{period}_downsampler"]

        self.period = period
        self.period_range = slice(*cfg[f"{period}_temporal_range"])

        self.target_has_missing_dates = self.cfg.get("target_has_missing_dates", False)

        urls, xarray_kwargs = get_source_url(cfg)

        self.scaling_static_range = self.cfg.get("scaling_static_range")
        
        data_dynamic = read_from_zarr(url=urls["dynamic_inputs"], chunks="auto", **xarray_kwargs).sel(time=self.period_range)
        data_static = read_from_zarr(url=urls["static_inputs"], chunks="auto", **xarray_kwargs)
 
        self.xd = data_dynamic[self.to_list(cfg.dynamic_inputs)] # list comprehension handle omegaconf lists
        self.xs = data_static[self.to_list(cfg.static_inputs)]
        self.y = data_dynamic[self.to_list(cfg.target_variables)]

        # subset dynamic inputs to the target timestep available
        if self.target_has_missing_dates:
            self.xd = self.xd.sel(time=self.y.time)

        if not self.cfg.data_lazy_load: # loading in memory
            self.xd = self.xd.load()
            self.xs = self.xs.load()
            self.y = self.y.load()

        # == DATASET INDICES AND MASKING

        # The archive is a stacked `cell` axis: runs x pool_size rows, one row
        # per (run, base cell). The pool was masked and chosen when the archive
        # was built (A1), so there is nothing left to mask out and nothing to
        # flatten - a cell *is* a sample. `mask` is still computed because
        # callers read it, but it should be all-False.
        self.pool_size = data_static.attrs.get("pool_size", self.xs.sizes["cell"])
        self.n_runs = self.xs.sizes["cell"] // self.pool_size

        if self.cfg.mask_variables is not None:
            self.mask = data_static[self.to_list(self.cfg.mask_variables)].to_array().any("variable")
        else:
            self.mask = None

        # `cell_coords` keeps its name and its meaning - where a row sits on the
        # map - but it now comes from the coordinates the archive carries rather
        # than from argwhere over a 2D grid.
        self.cell_coords = np.stack(
            [self.xs.lat_i.values, self.xs.lon_i.values], axis=1
        ) if "lat_i" in self.xs.coords else None

        # == ALIGNMENT (H5)

        self.check_alignment()

        # == DOWNSAMPLING

        # Captured before `build_sample_index`, which needs them, and before
        # the xarray objects become tensors further down.
        self.cell_size = self.xs.sizes["cell"]
        self.time_size = len(self.xd.time)

        self.build_sample_index()

        # == SOME USEFUL PARAMETERS
        # `cell_size` and `time_size` are set above, before the index build.
        self.dynamic_coords = self.xd.coords
        self.static_coords = self.xs.coords


        #  === PREPROCESS/TRANSFORM VARIABLES

        if self.cfg.get("preprocessor") is not None:
            self.xs = self.preprocessor.process(self.xs, "static_inputs")
            self.xd = self.preprocessor.process(self.xd, "dynamic_inputs")
            self.y = self.preprocessor.process(self.y, "target_variables")
        
        # == SCALING 

        # Reduce over the stacked cell axis. Stacking makes this simpler than a
        # `cycle` dimension would have: reducing over ("cell", "time") already
        # covers every run, where a cycle axis would have survived the
        # reduction and left one set of numbers per cycle.
        self.scaler.load_or_compute(
            self.xd, "dynamic_inputs", is_train, axes=("cell", "time")
        )

        self.scaler.load_or_compute(
            self.xs, "static_inputs", is_train, axes=("cell",)
        )

        self.scaler.load_or_compute(
            self.y, "target_variables", is_train, axes=("cell", "time")
        )

        self.xd = self.scaler.transform(self.xd, "dynamic_inputs")


        self.y = self.scaler.transform(self.y, "target_variables")


        self.xs = self.scaler.transform(self.xs, "static_inputs")
        



        # == WRITE SCALING STATS

        if is_train: # write if train
            if not self.scaler.use_cached: # write if not reading from cache
                self.scaler.write("dynamic_inputs")
                self.scaler.write("static_inputs")
                self.scaler.write("target_variables")
            else: # if reading from cache
                if self.scaler.flag_stats_computed: # if stats were not found in cache
                    self.scaler.write("dynamic_inputs")
                    self.scaler.write("static_inputs")
                    self.scaler.write("target_variables")


        # Pre-compute static data once
        self.static_tensor = torch.tensor(self.xs.to_array().values).float()
        
        # Pre-compute dynamic data shapes once
        self.dynamic_shape = self.xd[self.cfg.dynamic_inputs[0]].shape
        self.target_shape = self.y[self.cfg.target_variables[0]].shape
    

        # Convert to tensors once during initialization
        self.xd = self.xd.to_stacked_array(
            new_dim="feat", sample_dims=["time", "cell"]
        ).transpose("time", "feat", "cell").astype("float32")
        self.y = self.y.to_stacked_array(
            new_dim="feat", sample_dims=["time", "cell"]
        ).transpose("time", "feat", "cell").astype("float32")
        self.xs = self.xs.to_stacked_array(
            new_dim="feat", sample_dims=["cell"]
        ).transpose("feat", "cell").astype("float32")

        # Pre-process once
        if not self.cfg.data_lazy_load:  # Only if we're not doing lazy loading
            # Convert xarray to pre-processed tensors
            self.xd = torch.from_numpy(self.xd.values)
            self.y = torch.from_numpy(self.y.values)
            self.xs = torch.from_numpy(self.xs.values)

    def check_alignment(self):
        """Refuse an archive whose stores do not line up (H5).

        The stores are opened separately and matched purely by position, so a
        difference in length or order pairs one run's theta with another run's
        vwc. Nothing downstream notices: the shapes still work, training still
        converges, and the surrogate is just quietly wrong. These checks cost
        microseconds and turn that into a startup failure.

        The plan proposed comparing `xs.lat` against `np.tile(xd.lat, runs)`,
        which does not apply here - the forcing is repeated per run in the
        archive, so `xd` already carries all `runs x pool_size` rows and the
        two lat arrays compare directly.
        """
        n = self.xs.sizes["cell"]
        for name, obj in (("dynamic", self.xd), ("target", self.y)):
            if obj.sizes["cell"] != n:
                raise ValueError(
                    f"archive stores disagree: static has {n} cells, {name} "
                    f"has {obj.sizes['cell']}. They are paired by position, so "
                    f"this would marry one run's parameters to another's vwc."
                )

        if n % self.pool_size:
            raise ValueError(
                f"{n} rows is not a whole number of runs of "
                f"pool_size={self.pool_size}."
            )

        # The real check. `n % pool_size == 0` passes for a 16,800-row run
        # followed by two 300-row runs, because 17,400 divides by 300.
        if "run" in self.xs.coords:
            expected = np.repeat(np.arange(self.n_runs, dtype="int32"), self.pool_size)
            if not np.array_equal(self.xs.run.values, expected):
                raise ValueError(
                    f"the `run` coordinate is not {self.n_runs} contiguous "
                    f"blocks of {self.pool_size} rows. The archive was appended "
                    f"with a run of the wrong length."
                )

        # Same cells, in the same order, in both stores.
        for coord in ("lat", "lon"):
            if coord in self.xs.coords and coord in self.xd.coords:
                if not np.array_equal(self.xs[coord].values, self.xd[coord].values):
                    raise ValueError(
                        f"static and dynamic stores disagree on `{coord}`: the "
                        f"same row index points at different cells of the map."
                    )

        # Every run must cover the same base cells in the same order, or
        # `idx % pool_size` no longer identifies a base cell.
        if "lat_i" in self.xs.coords and self.n_runs > 1:
            base = self.xs.lat_i.values[:self.pool_size]
            if not np.array_equal(self.xs.lat_i.values, np.tile(base, self.n_runs)):
                raise ValueError(
                    "runs do not cover the same base cells in the same order."
                )

    def build_sample_index(self):
        """Build the (cell, time) sample index from scratch.

        Called by `__init__`, and again by `set_epoch` when the downsampler
        redraws (H3). It always starts from the full axes, because
        `sampling_idx` returns a subset: running it on its own output would
        shrink the selection again every epoch.

        It reads `cell_size`/`time_size` rather than `self.xs`/`self.xd`. By the
        time `set_epoch` first fires, `__init__` has replaced both with torch
        tensors, which have no `.sizes` and no `.time`.
        """
        # Compute cell (spatial) index
        self.cell_linear_index = np.arange(0, self.cell_size, 1)

        # Compute sequence (temporal) index
        # Each cell has a time series of equal length, so the sequence index is the same for every cell
        if self.period == "test":
            self.time_index = np.arange(0, self.time_size, 1)
        else:
            self.time_index = np.arange(0, self.time_size - self.seq_len, 1)

        # Downsample spatial and temporal indices based on rule
        if self.downsampler is not None:
            self.cell_linear_index , self.time_index = self.downsampler.sampling_idx([self.cell_linear_index , self.time_index])

        if self.period == "test":
            self.spacetime_index = self.cell_linear_index
        else:
            # (cell, time) pairs, cell-major: the order of
            # `itertools.product(cells, times)`, which `create_xarray_data`
            # reshapes predictions back with. Built with NumPy because it is
            # rebuilt every epoch (H9): the tuple list took 15 s and 5 GB at
            # 52,000 rows x 975 days.
            cells = np.asarray(self.cell_linear_index, dtype=np.int64)
            times = np.asarray(self.time_index, dtype=np.int64)
            self.spacetime_index = np.column_stack(
                [np.repeat(cells, len(times)), np.tile(times, len(cells))]
            )

    def set_epoch(self, epoch: int):
        """Redraw this epoch's cells, if the downsampler resamples (H3).

        A no-op unless the downsampler takes an epoch. `PoolDownsampler` only
        resamples for `split="train"`, so the validation set is unchanged from
        epoch to epoch and its loss stays comparable.
        """
        if self.downsampler is None or not hasattr(self.downsampler, "set_epoch"):
            return
        self.downsampler.set_epoch(epoch)
        self.build_sample_index()

    def __len__(self):
        return len(self.spacetime_index)

    def __getitem__(self, index):
        """One sample: (run, base cell, time window).

        All three stores are indexed by the same `idx_cell`, so the static, the
        forcing and the target of a row always come from the same archive row.
        The forcing is repeated per run in the archive, so no `% pool_size`
        lookup is needed here.
        """
        if self.period == "test":
            idx_cell = self.cell_linear_index[index]

            xd = self.xd[:, :, idx_cell]

            y = self.y[:, :, idx_cell]

            xs = self.xs[:, idx_cell]
        else:
            idx_cell, idx_time = self.spacetime_index[index]

            xd = self.xd[idx_time:idx_time + self.seq_len, :, idx_cell]

            y = self.y[idx_time:idx_time + self.seq_len, :, idx_cell]

            xs = self.xs[:, idx_cell]

        return {"xd": xd, "xs": xs, "y": y}

class WflowSBM(BaseDataset):
    def __init__(
        self, cfg, scaler, is_train=True, period="train"
    ):
        self.scaler = scaler

        self.cfg = self.validate_config(cfg)

        self.preprocessor = Preprocessor(cfg)

        self.downsampler = self.cfg[f"{period}_downsampler"]

        self.period = period
        self.period_range = slice(*cfg[f"{period}_temporal_range"])

        self.target_has_missing_dates = self.cfg.get("target_has_missing_dates", False)

        urls, xarray_kwargs = get_source_url(cfg)

        self.scaling_static_range = self.cfg.get("scaling_static_range")

        data_dynamic = read_from_zarr(url=urls["dynamic_inputs"], chunks="auto", **xarray_kwargs).sel(time=self.period_range)
        data_static = read_from_zarr(url=urls["static_inputs"], chunks="auto", **xarray_kwargs)
        
        self.xd = data_dynamic[self.to_list(cfg.dynamic_inputs)] # list comprehension handle omegaconf lists
        self.xs = data_static[self.to_list(cfg.static_inputs)]
        self.y = data_dynamic[self.to_list(cfg.target_variables)]

        # subset dynamic inputs to the target timestep available
        if self.target_has_missing_dates:
            self.xd = self.xd.sel(time=self.y.time)

        if not self.cfg.data_lazy_load: # loading in memory
            self.xd = self.xd.load()
            self.xs = self.xs.load()
            self.y = self.y.load()

        # == DATASET INDICES AND MASKING

        if self.cfg.mask_variables is not None and self.period != "test":
            # During training and validation remove cells marked as mask.
            self.mask = data_static[self.to_list(self.cfg.mask_variables)].to_array().any("variable")
            self.coords = np.argwhere(~self.mask.values)
        elif self.period == "test": 
            # No masking during testing, however computing mask is still useful.
            self.mask = data_static[self.to_list(self.cfg.mask_variables)].to_array().any("variable")
            shape = list(self.xs.dims.values())
            self.coords =  np.argwhere(np.ones(shape).astype(bool))

        # Compute cell (spatial) index 
        self.cell_index = np.arange(0, len(self.coords), 1)
        
        # Compute sequence (temporal) index
        # Each cell has a time series of equal length, so the sequence index is the same for every cell
        if self.cfg.dynamic_downsampler is not None or self.period == "test":
            self.time_index = np.arange(0, len(self.xd.time.values), 1)
        else:
            self.time_index = np.arange(self.seq_len, len(self.xd.time.values), 1)
        
        # == DOWNSAMPLING

        # Downsample spatial and temporal indices based on rule
        if self.downsampler is not None:
            self.cell_index, self.time_index = self.downsampler.sampling_idx([self.cell_index, self.time_index])

        # Generate dataset samples
        if self.cfg.dynamic_downsampler is not None:
            # Only downsample the spatial index, the time index to downsample the sequences, is generated at runtime.
            # Therefore the dataset samples are sequences of max time series length. 
            self.coord_samples = self.coords[self.cell_index]
        else:
            # The dataset samples are the combination of cell and time indices
            self.coord_samples = list(itertools.product(*(self.coords[self.cell_index].tolist(), self.time_index.tolist() )))  
        

        #  === PREPROCESS/TRANSFORM VARIABLES

        if self.cfg.get("preprocessor") is not None:
            self.xs = self.preprocessor.process(self.xs, "static_inputs")
            self.xd = self.preprocessor.process(self.xd, "dynamic_inputs")
            self.y = self.preprocessor.process(self.y, "target_variables")
        

        # == SCALING 

        self.scaler.load_or_compute(
            self.xd, "dynamic_inputs", is_train, axes=("lat","lon", "time")
        )
        
        self.scaler.load_or_compute(
            self.xs, "static_inputs", is_train, axes=("lat","lon")
        )

        self.scaler.load_or_compute(
            self.y, "target_variables", is_train, axes=("lat", "lon", "time")
        )

        self.xd = self.scaler.transform(self.xd, "dynamic_inputs")

        self.y = self.scaler.transform(self.y, "target_variables")

        # if self.scaling_static_range is not None:
        #     LOGGER.info(f"Scaling static inputs with {self.scaling_static_range}")   
        #     scaling_static_reordered = {
        #         k: self.cfg.scaling_static_range[k]
        #         for k in self.cfg.static_inputs
        #         if k in self.cfg.scaling_static_range
        #     }

        #     self.static_scale, self.static_center = self.get_scaling_parameter(
        #         scaling_static_reordered, self.cfg.static_inputs, output_type="xarray"
        #     )
        #     self.xs = self.scaler.transform_custom_range(
        #         self.xs, self.static_scale, self.static_center
        #     )
        # else:
        self.xs = self.scaler.transform(self.xs, "static_inputs")

        
        # == WRITE SCALING STATS

        if is_train: # write if train
            if not self.scaler.use_cached: # write if not reading from cache
                self.scaler.write("dynamic_inputs")
                self.scaler.write("static_inputs")
                self.scaler.write("target_variables")
            else: # if reading from cache
                if self.scaler.flag_stats_computed: # if stats were not found in cache
                    self.scaler.write("dynamic_inputs")
                    self.scaler.write("static_inputs")
                    self.scaler.write("target_variables")

    def __len__(self):
        return len((range(len(self.coord_samples))))

    def __getitem__(self, index):
        

        # if self.cfg.downsampling_temporal_dynamic:
        lat, lon = self.coord_samples[index]

        ds_pixel_dynamic = self.xd.isel(lat=lat, lon=lon) # lat, lon, time -> time
        ds_pixel_target = self.y.isel(lat=lat, lon=lon)
        ds_pixel_static = self.xs.isel(lat=lat, lon=lon)

        ds_pixel_dynamic = ds_pixel_dynamic.to_array().transpose("time", "variable") # time -> time, feature
        ds_pixel_target = ds_pixel_target.to_array().transpose("time", "variable") # time -> time, feature


        ds_pixel_static = ds_pixel_static.to_array()
        
        xd  = torch.tensor(ds_pixel_dynamic.values).float()
        xs = torch.tensor(ds_pixel_static.values).float()
        y = torch.tensor(ds_pixel_target.values).float()
        # else:
        #     idx_cell, idx_time = self.coord_samples[index]

        #     idx_lat, idx_lon = idx_cell
        #     # TODO: check
        #     ds_pixel_dynamic = self.xd.isel(lat=idx_lat, 
        #                                             lon=idx_lon, 
        #                                             time=slice(idx_time - self.seq_len + 1, idx_time + 1)) # lat, lon, time -> time

        #     ds_pixel_target = self.y.isel(lat=idx_lat, 
        #                                             lon=idx_lon, 
        #                                             time=slice(idx_time - self.seq_len + 1, idx_time + 1)) 
            
        #     ds_pixel_static = self.xs.isel(lat=idx_lat, lon=idx_lon)
    
        #     ds_pixel_dynamic = ds_pixel_dynamic.to_array().transpose("time", "variable") # time -> time, feature
        #     ds_pixel_target = ds_pixel_target.to_array().transpose("time", "variable") # time -> time, feature
        #     ds_pixel_static = ds_pixel_static.to_array()
            
        #     xd  = torch.tensor(ds_pixel_dynamic.values).float()
        #     xs = torch.tensor(ds_pixel_static.values).float()
        #     y = torch.tensor(ds_pixel_target.values).float()

        return {"xd": xd, "xs": xs, "y": y}

def warmup_window(times, period_range: slice, warmup_steps: int):
    """Find the time steps of a period plus the warm-up before it (H10).

    The surrogate is trained with ``seq_length`` days of spin-up, so its first
    days from a cold state are poor. Each sequence therefore starts
    ``warmup_steps`` steps before the period and the warm-up is not scored.
    If the data starts too late for a full warm-up, the first scored day moves
    later, so that every scored day has the full warm-up.

    :param times: sorted datetime64 array, the time axis of the forcing.
    :param period_range: slice of two dates, the period to score.
    :param warmup_steps: number of time steps before the first scored step.
    :return: (first, scored_start, last), indices into ``times``: the sequence
        is ``first..last`` and the scored part is ``scored_start..last``.
    """
    times = np.asarray(times)
    start = np.datetime64(period_range.start, "ns")
    end = np.datetime64(period_range.stop, "ns")
    i0 = int(np.searchsorted(times, start, side="left"))
    last = int(np.searchsorted(times, end, side="right")) - 1
    scored_start = max(i0, warmup_steps)
    if scored_start > last:
        raise ValueError(
            f"period {period_range.start}..{period_range.stop} has no step left to "
            f"score after a warm-up of {warmup_steps} steps (data starts {times[0]})"
        )
    return scored_start - warmup_steps, scored_start, last


class WflowSBMCal(BaseDataset):
    """Dataset returns sequences with length equal to calibration period

    With ``warmup_steps`` > 0, each sequence starts that many steps before the
    period, and the target of the warm-up steps is NaN, so it is not scored.
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

        data_head_input = read_from_zarr(url=urls["static_parameter_inputs"], chunks="auto", **xarray_kwargs)
   
        head_model_input_list = []
        # load only non calibration params
        for i in self.cfg.head_model_inputs:
            if i is not None and i != "cal_param":
                head_model_input_list.extend(self.cfg.head_model_inputs[i])
        print(head_model_input_list)
        # select 
        self.xd = data_dynamic[self.to_list(cfg.dynamic_inputs)].sel(time=self.period_range)
        self.xs = data_static[self.to_list(cfg.static_inputs)]
        self.y = data_target[self.to_list(cfg.target_variables)].sel(time=self.period_range)

        self.xp =  data_head_input[head_model_input_list]
        
        # subset dynamic inputs to the target timestep available
        
        if self.target_has_missing_dates is not None:
            self.xd = self.xd.sel(time=self.y.time)

        # H10: warm-up. The sequences given to the model start `warmup_steps`
        # before the period. self.xd and self.y above stay the period only,
        # for the masks and the scaling statistics, which do not change.
        self.warmup_steps = self.cfg.get("warmup_steps") or 0
        if self.warmup_steps > 0:
            times = data_dynamic.time.values
            first, scored_start, last = warmup_window(times, self.period_range, self.warmup_steps)
            self.xd_seq = data_dynamic[self.to_list(cfg.dynamic_inputs)].isel(time=slice(first, last + 1))
            # Keep every forcing day, so the model sees consecutive days.
            # Days missing from the target file become NaN.
            y_seq = data_target[self.to_list(cfg.target_variables)].reindex(time=self.xd_seq.time)
            self.y_seq = y_seq.where(y_seq.time >= times[scored_start])
            LOGGER.info(
                f"{period}: warm-up {times[first]} .. {times[scored_start - 1]}, "
                f"scored {times[scored_start]} .. {times[last]}"
            )

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

        # Compute time index
        #if self.cfg.downsampling_temporal_dynamic or self.period == "test":
        self.time_index = np.arange(0, len(self.xd.time.values), 1)
        # else:
        #     self.time_index = np.arange((self.seq_len -1), # size to index 
        #                                 len(self.xd.time.values), # not inclusive 
        #                                 1)

        # Reduce dataset size
        if self.downsampler is not None:
            self.cell_linear_index, self.time_index = self.downsampler.sampling_idx([self.cell_linear_index, self.time_index])

        # Generate dataset indices 
        #if self.cfg.downsampling_temporal_dynamic:
            # This assumes that the time index for sampling the sequences are generated at runtime.
            # The dataset returns the whole time series for each data sample
            # In this way it is possible to generate new random time indices every epoch to dynamically subsample the time domain. 
        self.coord_samples = self.coord_cells[self.cell_linear_index]
        #else:
            # Combined cell and time indices
        #    self.coord_samples = list(itertools.product(*(self.coord_cells[self.cell_linear_index].tolist(), self.time_index.tolist() )))  

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
        
        
        self.scaler.load_or_compute(
            self.xp, "head_model_inputs", is_train, axes=("lat","lon")
        )

        # From here on the model needs the sequences with their warm-up
        if self.warmup_steps > 0:
            self.xd, self.y = self.xd_seq, self.y_seq
            del self.xd_seq, self.y_seq
            if not self.cfg.data_lazy_load:
                self.xd = self.xd.load()
                self.y = self.y.load()

        self.xd = self.scaler.transform(self.xd, "dynamic_inputs")
        self.xs = self.scaler.transform(self.xs, "static_inputs")
        # transform target variable to reference statistics
        self.y = self.scaler.transform(self.y, "target_variables")
        # standardise transformed target variable to match emulator
        # that was trained with standardised reference
        #self.y = (self.y - self.y.mean("time")) / self.y.std("time")

        self.xp = self.scaler.transform(self.xp, "head_model_inputs").compute()
        
        # == PREPARE PARAMETERS FOR REG
        # if self.cfg.regularization is not None:
        #     miss_param = self.get_missing_regularization_parameter(cfg)
        #     self.xs_reg = data_static[miss_param]
        #     if not self.cfg.data_lazy_load:
        #         self.xs_reg.load()
        #     #self.scaler.transform_inverse_custom_range

        if is_train: # write if train
            if not self.scaler.use_cached: # write if not reading from cache
                self.scaler.write("dynamic_inputs")
                self.scaler.write("static_inputs")
                self.scaler.write("target_variables")
                self.scaler.write("head_model_inputs")
            else: # if reading from cache
                if self.scaler.flag_stats_computed: # if stats were not found in cache
                    self.scaler.write("dynamic_inputs")
                    self.scaler.write("static_inputs")
                    self.scaler.write("target_variables")
                    self.scaler.write("head_model_inputs")

    def __len__(self):
        return len((range(len(self.coord_samples))))

    def __getitem__(self, index):
        idx_lat, idx_lon = self.coord_samples[index]

        ds_pixel_dynamic = self.xd.isel(lat=idx_lat, lon=idx_lon) # lat, lon, time -> time
        ds_pixel_target = self.y.isel(lat=idx_lat, lon=idx_lon)
        ds_pixel_static = self.xs.isel(lat=idx_lat, lon=idx_lon)
        ds_pixel_head_input = self.xp.isel(lat=idx_lat, lon=idx_lon)

        ds_pixel_dynamic = ds_pixel_dynamic.to_array().transpose("time", "variable") # time -> time, feature
        ds_pixel_target = ds_pixel_target.to_array().transpose("time", "variable") # time -> time, feature

        ds_pixel_static = ds_pixel_static.to_array()
        ds_pixel_head_input = ds_pixel_head_input.to_array()
        
        # TODO: remove call to float
        xd  = torch.tensor(ds_pixel_dynamic.values).float()
        xs = torch.tensor(ds_pixel_static.values).float()
        y = torch.tensor(ds_pixel_target.values).float()
        xp = torch.tensor(ds_pixel_head_input.values).float()

        return {"xd": xd, "xs": xs, "y": y, "xp":xp}

class WflowSBMCube(BaseDataset):
    def __init__(
        self, cfg, scaler, is_train=True, period="train", scale_ontraining=False
    ):
        self.scale_ontraining = scale_ontraining
        self.scaler = scaler
        self.cfg = cfg

        self.preprocessor = Preprocessor(cfg)

        self.downsampler = self.cfg[f"{period}_downsampler"]

        self.period = period
        self.period_range = slice(*cfg[f"{period}_temporal_range"])

        urls, xarray_kwargs = get_source_url(cfg)

        self.scaling_static_range = self.cfg.get("scaling_static_range")

        data_dynamic = read_from_zarr(url=urls["dynamic_inputs"], chunks="auto", **xarray_kwargs).sel(time=self.period_range)
        data_static = read_from_zarr(url=urls["static_inputs"], chunks="auto",**xarray_kwargs)

        self.xd = data_dynamic[self.to_list(cfg.dynamic_inputs)]
        self.xs = data_static[self.to_list(cfg.static_inputs)]
        self.y = data_dynamic[self.to_list(cfg.target_variables)]

        
        self.shape = self.xd[self.cfg.dynamic_inputs[0]].shape

        if self.cfg.mask_variables is not None and self.period != "test":
            # apply mask 
            mask = data_static[self.to_list(self.cfg.mask_variables)].to_array().any("variable")
            self.mask = mask
        elif self.period == "test": # no masking when period is test 
            # FIXME mask is still useful
            mask = data_static[self.to_list(self.cfg.mask_variables)].to_array().any("variable")
            self.mask = mask


        (
            self.cbs_spatial_idxs,
            self.cbs_missing_idxs,
            self.cbs_degenerate_idxs,
            self.cbs_spatial_slices,
        ) = compute_cubelet_spatial_idxs(
            self.shape,
            self.cfg.batch_size["xsize"],
            self.cfg.batch_size["ysize"],
            self.cfg.batch_overlap["xover"],
            self.cfg.batch_overlap["yover"],
            self.cfg.keep_spatial_degenerate_cubelet,
            masks=self.mask,
            missing_policy=self.cfg.missing_policy,
        )

        (
            self.cbs_time_idxs,
            self.cbs_degenerate_idxs,
            self.cbs_time_slices,
        ) = compute_cubelet_time_idxs(
            self.shape,
            self.cfg.batch_size["tsize"],
            self.cfg.batch_overlap["tover"],
            self.cfg.keep_temporal_degenerate_cubelet,
            masks=self.mask,
        )

        cbs_tuple_idxs = compute_cubelet_tuple_idxs(
            self.cbs_spatial_idxs, self.cbs_time_idxs
        )
        cbs_slices = compute_cubelet_slices(
            self.cbs_spatial_slices, self.cbs_time_slices
        )

        self.cbs_mapping_idxs = cbs_mapping_idx_slice(cbs_tuple_idxs, cbs_slices)

        if self.downsampler is not None:
            # DOWNSAMPLE THE REMAINING INDEXES AFTER REMOVING MISSING AND DEGENERATED
            # return a subset of the cbs_mapping_idxs
            # TODO: also self.cbs_time_idxs and self.cbs_spatial_idxs should be updated
            self.cbs_mapping_idxs = self.downsampler.sampling_idx(self.cbs_mapping_idxs)

        if self.cfg.get("preprocessor") is not None:
            self.xs = self.preprocessor.process(self.xs, "static_inputs")
            self.xd = self.preprocessor.process(self.xd, "dynamic_inputs")
            self.y = self.preprocessor.process(self.y, "target_variables")

        # == SOME USEFUL PARAMETERS
        self.lat_size = len(self.xd.lat)
        self.lon_size = len(self.xd.lon)
        self.time_size = len(self.xd.time)
        self.dynamic_coords = self.xd.coords
        self.static_coords = self.xs.coords


        self.scaler.load_or_compute(
            self.xd, "dynamic_inputs", is_train, axes=("time", "lat", "lon")
        )

        self.scaler.load_or_compute(
            self.xs, "static_inputs", is_train, axes=("lat", "lon")
        )

        self.scaler.load_or_compute(
            self.y, "target_variables", is_train, axes=("time", "lat", "lon")
        )

        self.xd = self.scaler.transform(self.xd, "dynamic_inputs")

        self.y = self.scaler.transform(self.y, "target_variables")

        self.xs = self.scaler.transform(self.xs, "static_inputs")

        if is_train: # write if train
            if not self.scaler.use_cached: # write if not reading from cache
                self.scaler.write("dynamic_inputs")
                self.scaler.write("static_inputs")
                self.scaler.write("target_variables")
            else: # if reading from cache
                if self.scaler.flag_stats_computed: # if stats were not found in cache
                    self.scaler.write("dynamic_inputs")
                    self.scaler.write("static_inputs")
                    self.scaler.write("target_variables")

        xd_data_vars = list(self.xd.data_vars)
        self.xd = self.xd.to_stacked_array(
            new_dim="feat", sample_dims=["time", "lat", "lon"]
        )  # time, lat, lon , feat
        self.xd = self.xd.transpose("time", "feat", "lat", "lon")  # T C H W
        self.xd = self.xd.astype("float32")
        self.xd = self.xd.drop_vars(["feat", "variable"]).assign_coords(
            {"feat": xd_data_vars}
        )

        y_data_vars = list(self.y.data_vars)
        self.y = self.y.to_stacked_array(
            new_dim="feat", sample_dims=["time", "lat", "lon"]
        )
        self.y = self.y.transpose("time", "feat", "lat", "lon")  # T C H W
        self.y = self.y.astype("float32")
        self.y = self.y.drop_vars(["feat", "variable"]).assign_coords(
            {"feat": y_data_vars}
        )

        xs_data_vars = list(self.xs.data_vars)
        self.xs = self.xs.to_stacked_array(
            new_dim="feat", sample_dims=["lat", "lon"]
        )  # H W C
        self.xs = self.xs.transpose("feat", "lat", "lon")
        self.xs = self.xs.astype("float32")
        self.xs = self.xs.drop_vars(["feat", "variable"]).assign_coords(
            {"feat": xs_data_vars}
        )

        if not self.cfg.data_lazy_load: # loading in memory
            self.xd = self.xd.load()
            self.xs = self.xs.load()
            self.y = self.y.load()

        # TODO: improve this
        self.xd = self.xd.fillna(self.cfg.fill_missing)
        self.y = self.y.fillna(self.cfg.fill_missing)
        self.xs = self.xs.fillna(self.cfg.fill_missing)

    def __len__(self):
        return len(self.cbs_mapping_idxs)

    def get_indexes(self):
        return list(range(len(self.cbs_mapping_idxs)))

    def __getitem__(self, index):

        cubelet_idx = list(self.cbs_mapping_idxs.keys())[index]

        time_slice = self.cbs_mapping_idxs[cubelet_idx]["time"]
        lat_slice = self.cbs_mapping_idxs[cubelet_idx]["lat"]
        lon_slice = self.cbs_mapping_idxs[cubelet_idx]["lon"]

        # xr.Dataarray to np.ndarray, this triggers loading in memory, in case persist = False
        xd = self.xd[time_slice, :, lat_slice, lon_slice].values  # L C H W
        y = self.y[time_slice, :, lat_slice, lon_slice].values  # L C H W

        # np.ndarray ot torch.tensor
        xd = torch.tensor(xd)
        y = torch.tensor(y)

        xs = self.xs[:, lat_slice, lon_slice].values  # C H W
        xs = torch.tensor(xs)

        if self.cfg.static_to_dynamic:
            xs = xs.unsqueeze(0).repeat(xd.size(0), 1, 1, 1)

        return {"xd": xd, "xs": xs, "y": y}

class Wflow2dCal(BaseDataset):
    def __init__(
        self, cfg, scaler, is_train=True, period="train", scale_ontraining=False
    ):
        self.scale_ontraining = scale_ontraining
        self.scaler = scaler
        self.cfg = cfg

        self.downsampler = self.cfg[f"{period}_downsampler"]

        self.period = slice(*cfg[f"{period}_temporal_range"])

        file_path = f"{cfg.data_dir}/{cfg.data_file}"

        self.xd = read_from_zarr(
            url="/mnt/CEPH_PROJECTS/InterTwin/hydrologic_data/surrogate_input/adg1km_eobs_original.zarr",
            group="xd",
        ).sel(time=self.period)[list(self.cfg.dynamic_inputs)]

        self.xs = read_from_zarr(
            url="/mnt/CEPH_PROJECTS/InterTwin/hydrologic_data/param_learning_input/predictor_test.zarr"
        ).drop_vars("spatial_ref")[list(self.cfg.static_inputs)]

        self.y = xr.open_dataset(
            "/mnt/CEPH_PROJECTS/InterTwin/hydrologic_data/SSM-RT0-SIG0-R-CRRL/processed/daily/adige_2018-2021.nc",
            mask_and_scale=True,
        ).sel(time=self.period)

        self.shape = self.xd[self.cfg.dynamic_inputs[0]].shape

        self.xd = self.xd.rio.set_crs(4326).rio.write_crs()
        self.xd.rio.set_spatial_dims(x_dim="lon", y_dim="lat", inplace=True)

        self.xd = self.xd.sel(time=self.y.time)

        self.xs = self.xs.rename({"x": "lon", "y": "lat"})

        if self.cfg.mask_variables is not None:
            self.mask = (
                read_from_zarr(url=file_path, group="mask")
                .mask.sel(mask_layer=self.cfg.mask_variables)
                .any(dim="mask_layer")
            )
        else:
            self.mask = None

        (
            self.cbs_spatial_idxs,
            self.cbs_missing_idxs,
            self.cbs_degenerate_idxs,
            self.cbs_spatial_slices,
        ) = compute_cubelet_spatial_idxs(
            self.shape,
            self.cfg.batch_size["xsize"],
            self.cfg.batch_size["ysize"],
            self.cfg.batch_overlap["xover"],
            self.cfg.batch_overlap["yover"],
            self.cfg.keep_spatial_degenerate_cubelet,
            masks=self.mask,
            missing_policy=self.cfg.missing_policy,
        )

        (
            self.cbs_time_idxs,
            self.cbs_degenerate_idxs,
            self.cbs_time_slices,
        ) = compute_cubelet_time_idxs(
            self.shape,
            self.cfg.batch_size["tsize"],
            self.cfg.batch_overlap["tover"],
            self.cfg.keep_temporal_degenerate_cubelet,
            masks=self.mask,
        )

        cbs_tuple_idxs = compute_cubelet_tuple_idxs(
            self.cbs_spatial_idxs, self.cbs_time_idxs
        )
        cbs_slices = compute_cubelet_slices(
            self.cbs_spatial_slices, self.cbs_time_slices
        )

        self.cbs_mapping_idxs = cbs_mapping_idx_slice(cbs_tuple_idxs, cbs_slices)

        if self.downsampler is not None:
            # DOWNSAMPLE THE REMAINING INDEXES AFTER REMOVING MISSING AND DEGENERATED
            # return a subset of the cbs_mapping_idxs
            # TODO: also self.cbs_time_idxs and self.cbs_spatial_idxs should be updated
            self.cbs_mapping_idxs = self.downsampler.sampling_idx(self.cbs_mapping_idxs)

            # Scaling

        self.scaler.load_or_compute(
            self.xd, "dynamic_inputs", is_train, axes=("time", "lat", "lon")
        )

        self.scaler.load_or_compute(
            self.xs, "static_inputs", is_train, axes=("lat", "lon")
        )

        self.scaler.load_or_compute(
            self.y, "target_variables", is_train, axes=("time", "lat", "lon")
        )

        if not self.scale_ontraining:
            self.xd = self.scaler.transform(self.xd, "dynamic_inputs")
            self.xs = self.scaler.transform(self.xs, "static_inputs")
            self.y = self.scaler.transform(self.y, "target_variables")

        if is_train:
            self.scaler.write("dynamic_inputs")
            self.scaler.write("static_inputs")
            self.scaler.write("target_variables")

        self.xd = self.xd.to_stacked_array(
            new_dim="feat", sample_dims=["time", "lat", "lon"]
        )  # time, lat, lon , feat
        self.xd = self.xd.transpose("time", "feat", "lat", "lon")  # T C H W
        self.xd = self.xd.astype("float32")

        self.y = self.y.to_stacked_array(
            new_dim="feat", sample_dims=["time", "lat", "lon"]
        )
        self.y = self.y.transpose("time", "feat", "lat", "lon")  # T C H W
        self.y = self.y.astype("float32")

        self.xs = self.xs.to_stacked_array(
            new_dim="feat", sample_dims=["lat", "lon"]
        )  # H W C
        self.xs = self.xs.transpose("feat", "lat", "lon")
        self.xs = self.xs.astype("float32")

        if self.cfg.persist:
            self.xd = self.xd.compute()
            self.y = self.y.compute()
            self.xs = self.xs.compute()

        # TODO: fix this
        self.xd = self.xd.fillna(self.cfg.fill_missing)
        self.y = self.y.fillna(self.cfg.fill_missing)
        self.xs = self.xs.fillna(self.cfg.fill_missing)

        self.top_layer_res = -1 * self.forcing.lat.diff("lat").values[0] / 2

    def __len__(self):
        return len(self.cbs_mapping_idxs)

    def get_indexes(self):
        return list(range(len(self.cbs_mapping_idxs)))

    def __getitem__(self, index):
        cubelet_idx = list(self.cbs_mapping_idxs.keys())[index]

        # print(index, cubelet_idx)

        time_slice = self.cbs_mapping_idxs[cubelet_idx]["time"]
        lat_slice = self.cbs_mapping_idxs[cubelet_idx]["lat"]
        lon_slice = self.cbs_mapping_idxs[cubelet_idx]["lon"]

        # xr.Dataarray to np.ndarray, this triggers loading in memory, in case persist = False
        forcing = self.xd[time_slice, :, lat_slice, lon_slice]  # .values # L C H W
        target = self.y[time_slice, :, lat_slice, lon_slice]  # .values # L C H W

        latmin = forcing.lat.values.min()
        latmax = forcing.lat.values.max()
        lonmin = forcing.lon.values.min()
        lonmax = forcing.lon.values.max()

        predictor = self.xs.sel(
            lat=slice(latmax + self.top_layer_res, latmin - self.top_layer_res),
            lon=slice(lonmin - self.top_layer_res, lonmax + self.top_layer_res),
        ).values

        forcing = forcing.values
        target = target.values

        # np.ndarray ot torch.tensor
        forcing = torch.FloatTensor(forcing)
        target = torch.FloatTensor(target)
        predictor = torch.FloatTensor(predictor)

        if self.cfg.lstm_1d:
            # Super slow when persist == False
            # If True means that the xsize and ysize is equal to 1

            # xd = xd.flatten(2,3) # L C H W => L C N
            xd = xd.squeeze()  # L C H W, but H W is size 1,1 => L C
            # xd = torch.permute(xd, (2, 0, 1)) # N L C, , but N = 1
            # xd = x.squeeze(0)

            # y = y.flatten(2,3) # L C H W => L C N
            y = y.squeeze()  # L C H W, but H W is size 1,1 => L C
            # y = torch.permute(y, (2, 0, 1)) # N L C, but N = 1
            # y = y.squeeze(0)
            if self.xs is not None:
                xs = xs.squeeze()  # C H W => C N

        if self.static_to_dynamic:
            if self.lstm_1d:
                # print(xs.shape)
                predictor = predictor.unsqueeze(0).repeat(
                    forcing.size(0),
                    1,
                )
            else:
                predictor = predictor.unsqueeze(0).repeat(forcing.size(0), 1, 1, 1)
        return predictor, forcing, target
