import torch
import xarray as xr
import numpy as np
import numpy.typing as npt
from torch.utils.data import Dataset

from hython.utils import (compute_cubelet_spatial_idxs, 
                          compute_cubelet_time_idxs, 
                          cbs_mapping_idx_slice,
                          compute_cubelet_slices, 
                          compute_cubelet_tuple_idxs,
                          compute_grid_indices,
                          cbs_mapping_idx_slice_notime
)
try:
    import xbatcher
except:
    pass

class LSTMDataset(Dataset):
    def __init__(
        self,
        xd: xr.DataArray | npt.NDArray,
        y: xr.DataArray | npt.NDArray,
        xs: xr.DataArray | npt.NDArray = None,
        original_domain_shape = (),
        mask = None,
        downsampler = None, 
        normalizer_dynamic = None,
        normalizer_static = None,
        normalizer_target = None
    ):

        self.shape = original_domain_shape

        self.xd = xd
        self.y = y
        self.xs = xs

        self.downsampler = downsampler


        # compute indexes

        ishape = self.shape[0]  # rows (y, lat)
        jshape = self.shape[1]  # columns (x, lon)

        irange = np.arange(0, ishape, 1)
        jrange = np.arange(0, jshape, 1)

        self.grid_idx_2d = compute_grid_indices(shape=self.shape)
        self.grid_idx_1d = self.grid_idx_2d.flatten()
        print(self.grid_idx_1d)

        # IF DOWNSAMPLING 
        # Reduces the available indexes to a valid subset
        if downsampler:
            # Same keep only indexes that satisfy some rule
            self.grid_idx_1d_downsampled = self.downsampler.sampling_idx(self.grid_idx_2d, self.shape)

        # IF REMOVE MISSING FROM MASK
        # Reduces the available indexes to a valid subset
        if mask is not None:
            # This actually does not touch the dataset, only remove indexes corresponding to missing values from the available indexes
            idx_nan = self.grid_idx_2d[mask]
            
            if downsampler:
                self.grid_idx_1d_valid = np.setdiff1d(self.grid_idx_1d_downsampled, idx_nan)
            else:
                self.grid_idx_1d_valid = np.setdiff1d(self.grid_idx_1d, idx_nan)
        else:
            if downsampler:
                self.grid_idx_1d_valid = self.grid_idx_1d_downsampled
            else:
                self.grid_idx_1d_valid = self.grid_idx_1d

        # NORMALIZE BASED IF MAKS AND IF DOWNSAMPLING
        if normalizer_dynamic is not None:
            # this normalize the data corresponding to valid indexes
            
            if normalizer_dynamic.stats_iscomputed: # validation or test
                self.xd = normalizer_dynamic.normalize(self.xd)
            else:
                # compute stats for training
                normalizer_dynamic.compute_stats(self.xd[self.grid_idx_1d_valid])
                self.xd = normalizer_dynamic.normalize(self.xd)

        if normalizer_static is not None:     
            if normalizer_static.stats_iscomputed: # validation or test
                self.xs = normalizer_static.normalize(self.xs)
            else:
                if downsampler:
                    normalizer_static.compute_stats(self.xs[self.grid_idx_1d_valid])
                else:
                    normalizer_static.compute_stats(self.xs)
                    
                self.xs = normalizer_static.normalize(self.xs)
        if normalizer_target is not None:
            if normalizer_target.stats_iscomputed: # validation or test
                self.y = normalizer_target.normalize(self.y)
            else:
                normalizer_target.compute_stats(self.y[self.grid_idx_1d_valid])
                self.y = normalizer_target.normalize(self.y)
                
        
        self.xs = self.xs.astype(np.float32)

        if isinstance(self.xd, xr.DataArray):
            self.xd = torch.tensor(self.xd.values)
            self.y = torch.tensor(self.y.values)
            self.xs = torch.tensor(self.xs.values)
        else:
            self.xd = torch.tensor(self.xd)
            self.y = torch.tensor(self.y)
            self.xs = torch.tensor(self.xs)


    def __len__(self):
        return len(self.grid_idx_1d_valid)

    def get_indexes(self):
        return list(range(len(self.grid_idx_1d_valid)))
    
    def __getitem__(self, index):
        
        item_index = self.grid_idx_1d_valid[index]

        if self.xs is not None:
            return self.xd[item_index], self.xs[item_index], self.y[item_index]
        else:
            return self.xd[item_index], self.y[item_index]

class XBatchDataset(Dataset):
    """
    Returns batches of Ntile,seq L T C H W

    The problem with this shape is that I cannot sample in time as the data points mixes both space (tile) and time (seq)
    
    """
    def __init__(self, 
                 xd,
                 y, 
                 xs = None,
                 lstm = False, 
                 join = False,
                 xbatcher_kwargs = {},
                 transform = None):

        self.join = join
        
        self.xd_gen = self._get_xbatcher_generator(xd, **xbatcher_kwargs)


        print( "dynamic: ", len(self.xd_gen))
        
        self.y_gen = self._get_xbatcher_generator(y, **xbatcher_kwargs)

        #xbatcher_kwargs["input_dims"].pop("time")
        
        self.xs_gen = self._get_xbatcher_generator(xs, **xbatcher_kwargs)
        self.transform = transform
        self.lstm = lstm

        print("static: ", len(self.xs_gen))
    def __len__(self):
        return len(self.xd_gen)
    
    def __getitem__(self, index):
        
        
        
        
        if self.lstm:
            
            # Now using this: Ntile[index] C Nseq L H W => Npixel L C
            ## Ntile,seq[index] C L H W => Npixel L C
            
            xd = self.xd_gen[index].torch.to_tensor().flatten(2) # C L Npixel
            #print(xd.shape)
            xd = torch.permute(xd, (2, 1, 0)) # Npixel L C  
            #print(xd.shape)
            y = self.y_gen[index].torch.to_tensor().flatten(2) # C L Npixel
            y = torch.permute(y, (2, 1, 0)) # # Npixel L C  
            
            if self.xs_gen is not None:
                # Ntile[index] C H W
                # Ntile,seq[index] C L H W
                xs = self.xs_gen[index].torch.to_tensor().flatten(2) # C L Npixel
                #print(xs.shape)
                #xs = xs.unsqueeze(1).repeat(1, xd.size(1), 1) # C T Npixel
                #xs = xs.unsqueeze(2).repeat(1, 1, xd.size(2), 1) # C T Nseq Npixel

        else:
            # Ntile,seq[index] C T H W => Npixel T C
            
            xd = self.xd_gen[index].transpose("time", "variable",...).torch.to_tensor() # C T H W => T C H W
            y = self.y_gen[index].transpose("time", "variable",...).torch.to_tensor() # C T H W => T C H W

            if self.xs_gen is not None:
                xs = self.xs_gen[index].transpose("time", "variable",...).torch.to_tensor() 
                xs = xs.to(torch.float32) #torch.transpose(xs, 0,1)
                #import pdb;pdb.set_trace()
                #xs = xs.unsqueeze(0).repeat(xd.size(0), 1, 1, 1) # T C H W

                
                
        if self.transform:
            xd = self.transform(xd)  
            
        if self.xs_gen is not None:
            if self.join:
                return xd, y
            else:
                return xd, xs, y
        else:
            return xd, y

    def _get_xbatcher_generator(self, ds, input_dims, concat_input_dims=True, batch_dims= {}, input_overlap={}, preload_batch=False):

        time = input_dims.get("time", None)

        
        if ds is None:
            return None
        
        if time is None:

            gen = xbatcher.BatchGenerator(
                ds = ds, 
                input_dims = input_dims,
                concat_input_dims = False,
                batch_dims = batch_dims,
                input_overlap = input_overlap,
                preload_batch = preload_batch
            )
        else:
            
            gen = xbatcher.BatchGenerator(
                ds = ds, 
                input_dims = input_dims,
                concat_input_dims = concat_input_dims,
                batch_dims = batch_dims,
                input_overlap = input_overlap,
                preload_batch = preload_batch
            )
            
        return gen

class LumpedDataset(Dataset):
    def __init__(
        self,
        xd: torch.Tensor | npt.NDArray,
        y: torch.Tensor | npt.NDArray,
        xs: torch.Tensor | npt.NDArray = None,
        seq_length: int = 60,
        create_seq: bool = False,
    ):
        """Create a dataset for training and validating LSTM-based models

        LSTM assumes shape (B, T, F), with B as basin

        Parameters
        ----------
        xd : torch.Tensor | npt.NDArray
            Dynamic parameter
        y : torch.Tensor | npt.NDArray
            Target
        xs : torch.Tensor | npt.NDArray, optional
            Static parameter, by default None
        seq_length : int, optional
            The hyperparameter of the LSTM represents the time window to gather info to predict, by default 60
        create_seq : bool, optional
            Create n sequences of size seq_length each from time series. Works for small datasets, by default False
        """
        self.seq_len = seq_length
        self.create_seq = create_seq

        num_gridcells, num_samples, num_features = xd.shape

        if isinstance(xs, np.ndarray):
            self.xs = xs.astype(np.float32)
        else:
            self.xs = xs

        if create_seq:
            xd_new = np.zeros(
                (num_gridcells, num_samples - seq_length + 1, seq_length, num_features)
            )
            y_new = np.zeros((num_gridcells, num_samples - seq_length + 1, 1))
            for i in range(0, xd_new.shape[1]):
                xd_new[:, i, :, :num_features] = xd[:, i : i + seq_length, :]
                y_new[:, i, 0] = y[:, i + seq_length - 1, 0]

            self.Xd = torch.from_numpy(xd_new.astype(np.float32))
            self.y = torch.from_numpy(y_new.astype(np.float32))
        else:
            self.Xd = xd.nanmean(0, keepdim=True)  # average over gridcell
            self.y = y.nanmean(0, keepdim=True)

        self.xs = xs.nanmean(0, keepdim=True)

    def __len__(self):
        """Returns number of lumped basins"""
        return self.Xd.shape[0]

    def __getitem__(self, i):
        if self.xs is not None:
            # case when there are static values
            if self.create_seq:
                return self.Xd[i], self.xs, self.y[i]
            else:
                return self.Xd[i], self.xs[i], self.y[i]
        else:
            return self.Xd[i], self.y[i]

class TilesDataset(Dataset):

    def __init__(self, 
                 xs: xr.Dataset = None,  # lat, lon
                 mask = None,
                 downsampler = None,
                 normalizer_static = None,
                 shape:tuple = (), # time ,lat ,lon
                 batch_size:dict = {"xsize":20, "ysize":20},
                 overlap:dict = {"xover":0, "yover":0},
                 missing_policy: str | float = "all",
                 fill_missing = 0, 
                 persist=False, 
                 lstm_1d = False, 
                 static_to_dynamic=False
                 ):

        self.xs = xs

        self.shape = shape

        self.mask = mask

        self.missing_policy = missing_policy

        self.downsampler = downsampler

        KEEP_DEGENERATE_CUBELETS = False # TODO: hardcoded
        
        # compute stuff

        self.cbs_spatial_idxs, self.cbs_missing_idxs, self.cbs_degenerate_idxs, self.cbs_spatial_slices = compute_cubelet_spatial_idxs(shape, 
                                                                                                                   batch_size['xsize'], 
                                                                                                                   batch_size['ysize'], 
                                                                                                                   overlap['xover'], 
                                                                                                                   overlap['yover'], 
                                                                                                                   KEEP_DEGENERATE_CUBELETS,
                                                                                                                   masks = self.mask,
                                                                                                                   missing_policy=self.missing_policy) 
        self.cbs_mapping_idxs = cbs_mapping_idx_slice_notime(self.cbs_spatial_idxs, self.cbs_spatial_slices)
        
        if self.downsampler is not None:
            # DOWNSAMPLE THE REMAINING INDEXES AFTER REMOVING MISSING AND DEGENERATED
            # return a subset of the cbs_mapping_idxs
            #TODO: also self.cbs_time_idxs and self.cbs_spatial_idxs should be updated
            self.cbs_mapping_idxs = self.downsampler.sampling_idx(self.cbs_mapping_idxs)
            
        
        if normalizer_static is not None:
            # this normalize the data corresponding to valid indexes
            
                
            if normalizer_static.stats_iscomputed: # validation or test
                self.xs = normalizer_static.normalize(self.xs)
            else:
                if downsampler:
                    normalizer_static.compute_stats(self.xs)
                else:
                    normalizer_static.compute_stats(self.xs)
                    
                self.xs = normalizer_static.normalize(self.xs)


        self.xs = xs.to_stacked_array( new_dim="feat", sample_dims = ["lat", "lon"]) # H W C
        self.xs = self.xs.transpose("feat", "lat", "lon")
        self.xs = self.xs.astype("float32")
            


        if persist:
            self.xs = self.xs.persist()

        if self.xs is not None:
            self.xs = self.xs.fillna(fill_missing)

        self.lstm_1d = lstm_1d
        self.static_to_dynamic = static_to_dynamic


    def __len__(self):
        return len(self.cbs_mapping_idxs)

    def get_indexes(self):
        return list(range(len(self.cbs_mapping_idxs)))
    
    def __getitem__(self, index):
        
        cubelet_idx = list(self.cbs_mapping_idxs.keys())[index]
        
        #print(index, cubelet_idx)

        lat_slice =  self.cbs_mapping_idxs[cubelet_idx]["lat"]
        lon_slice = self.cbs_mapping_idxs[cubelet_idx]["lon"]

        xs = self.xs[:, lat_slice,lon_slice].values # C H W
        xs = torch.tensor(xs)

            
        if self.lstm_1d:
            xs = xs.squeeze() # C H W => C N  

        return xs

        



class CubeletsDataset(Dataset):

    def __init__(self, 
                 xd: xr.Dataset, # time, lat, lon 
                 y:xr.Dataset, # time, lat, lon
                 xs: xr.Dataset = None,  # lat, lon
                 mask = None,
                 downsampler = None,
                 normalizer_dynamic = None,
                 normalizer_static = None,
                 normalizer_target = None,
                 shape:tuple = (), # time ,lat ,lon
                 batch_size:dict = {"xsize":20, "ysize":20, "tsize":20},
                 overlap:dict = {"xover":0, "yover":0, "tover":0},
                 missing_policy: str | float = "all",
                 fill_missing = 0, 
                 persist=False, 
                 lstm_1d = False, 
                 static_to_dynamic=False
                 ):
        """
        CAN I USE IT FOR PARAM LEARNING?

        IN THE CASE I WANT TO PRECOMPUTE THE DUMMY TIME DIMENSION FOR THE STATIC PARAMETERS I COULD MERGE XS WITH XD AND PASS IT AS XD AND SET XS=NONE
        """
        
        self.xd = xd
        self.y = y
        self.xs = xs

        self.shape = shape

        self.mask = mask

        self.missing_policy = missing_policy

        self.downsampler = downsampler

        KEEP_DEGENERATE_CUBELETS = False # TODO: hardcoded
        
        # compute stuff

        self.cbs_spatial_idxs, self.cbs_missing_idxs, self.cbs_degenerate_idxs, self.cbs_spatial_slices = compute_cubelet_spatial_idxs(shape, 
                                                                                                                   batch_size['xsize'], 
                                                                                                                   batch_size['ysize'], 
                                                                                                                   overlap['xover'], 
                                                                                                                   overlap['yover'], 
                                                                                                                   KEEP_DEGENERATE_CUBELETS,
                                                                                                                   masks = self.mask,
                                                                                                                   missing_policy=self.missing_policy) 
        #print(self.cbs_spatial_idxs)

        self.cbs_time_idxs, self.cbs_degenerate_idxs, self.cbs_time_slices = compute_cubelet_time_idxs(shape, 
                                                                                        batch_size['tsize'],
                                                                                        overlap['tover'], 
                                                                                        KEEP_DEGENERATE_CUBELETS,
                                                                                        masks = self.mask)

        #print(self.cbs_spatial_idxs)
        
        cbs_tuple_idxs = compute_cubelet_tuple_idxs(self.cbs_spatial_idxs, self.cbs_time_idxs)
        cbs_slices = compute_cubelet_slices(self.cbs_spatial_slices, self.cbs_time_slices)

        self.cbs_mapping_idxs = cbs_mapping_idx_slice(cbs_tuple_idxs, cbs_slices)
        
        if self.downsampler is not None:
            # DOWNSAMPLE THE REMAINING INDEXES AFTER REMOVING MISSING AND DEGENERATED
            # return a subset of the cbs_mapping_idxs
            #TODO: also self.cbs_time_idxs and self.cbs_spatial_idxs should be updated
            self.cbs_mapping_idxs = self.downsampler.sampling_idx(self.cbs_mapping_idxs)
            
        
        if normalizer_dynamic is not None:
            # this normalize the data corresponding to valid indexes
            if normalizer_dynamic.stats_iscomputed: # validation or test
                self.xd = normalizer_dynamic.normalize(self.xd)
            else:
                # compute stats for training
                normalizer_dynamic.compute_stats(self.xd)
                self.xd = normalizer_dynamic.normalize(self.xd)

        if normalizer_static is not None:
            #import pdb;pdb.set_trace()
            if normalizer_static.stats_iscomputed: # validation or test
                self.xs = normalizer_static.normalize(self.xs)
            else:
                normalizer_static.compute_stats(self.xs)   
                self.xs = normalizer_static.normalize(self.xs)

        if normalizer_target is not None: 
            if normalizer_target.stats_iscomputed: # validation or test
                self.y = normalizer_target.normalize(self.y)
            else:
                normalizer_target.compute_stats(self.y)
                self.y = normalizer_target.normalize(self.y)

        # either do this here or in the getitem
        # maybe add HERE THE RASHAPE IN CASE IS LSTM 1D
        self.xd = self.xd.to_stacked_array( new_dim="feat", sample_dims = ["time", "lat", "lon"]) # time, lat, lon , feat
        self.xd = self.xd.transpose("time", "feat", "lat" , "lon") # T C H W
        self.xd = self.xd.astype("float32")

        self.y = self.y.to_stacked_array( new_dim="feat", sample_dims = ["time", "lat", "lon"])
        self.y = self.y.transpose("time", "feat", "lat" , "lon") # T C H W
        self.y = self.y.astype("float32")

        if self.xs is not None:
            self.xs = self.xs.to_stacked_array( new_dim="feat", sample_dims = ["lat", "lon"]) # H W C
            self.xs = self.xs.transpose("feat", "lat", "lon")
            self.xs = self.xs.astype("float32")
            


        if persist:
            self.xd = self.xd.compute()
            self.y = self.y.compute()
            if self.xs is not None:
                self.xs = self.xs.compute()

        # The conversion of missing nans should occur at the very end. The nans are used in the compute_cubelets_*_idxs to remove cubelets with all nans
        # should the missing flag not be equal to a potential valid value for that quantity? for example zero may be valid for many geophysical variables
        self.xd = self.xd.fillna(fill_missing)
        
        self.y = self.y.fillna(fill_missing)

        if self.xs is not None:
            self.xs = self.xs.fillna(fill_missing)

        self.lstm_1d = lstm_1d
        self.static_to_dynamic = static_to_dynamic

        # expand static to dynamic 
        #time_da = xr.DataArray(dynamic.time.values, [('time', dynamic.time.values)])
        #static = static.expand_dims({"time":time_da})

    def __len__(self):
        return len(self.cbs_mapping_idxs)

    def get_indexes(self):
        return list(range(len(self.cbs_mapping_idxs)))
    
    def __getitem__(self, index):
        
        cubelet_idx = list(self.cbs_mapping_idxs.keys())[index]
        
        #print(index, cubelet_idx)

        time_slice = self.cbs_mapping_idxs[cubelet_idx]["time"]
        lat_slice =  self.cbs_mapping_idxs[cubelet_idx]["lat"]
        lon_slice = self.cbs_mapping_idxs[cubelet_idx]["lon"]

        # xr.Dataarray to np.ndarray, this triggers loading in memory, in case persist = False
        xd = self.xd[time_slice,:, lat_slice, lon_slice].values # L C H W
        y = self.y[time_slice,:, lat_slice, lon_slice].values # L C H W
        

        # np.ndarray ot torch.tensor
        xd = torch.tensor(xd)
        y = torch.tensor(y)

        if self.xs is not None:
            xs = self.xs[:, lat_slice,lon_slice].values # C H W
            xs = torch.tensor(xs)
        else:
            xs = None
            
        if self.lstm_1d:
            # Super slow when persist == False
            # If True means that the xsize and ysize is equal to 1
            
            #xd = xd.flatten(2,3) # L C H W => L C N
            xd = xd.squeeze() # L C H W, but H W is size 1,1 => L C
            #xd = torch.permute(xd, (2, 0, 1)) # N L C, , but N = 1
            #xd = x.squeeze(0)
            
            #y = y.flatten(2,3) # L C H W => L C N
            y = y.squeeze() # L C H W, but H W is size 1,1 => L C
            #y = torch.permute(y, (2, 0, 1)) # N L C, but N = 1
            #y = y.squeeze(0)
            if self.xs is not None:
                xs = xs.squeeze() # C H W => C N  

        #print(xd.shape,y.shape, xs.shape)
        if self.xs is not None:
            if self.static_to_dynamic:
                if self.lstm_1d:
                    #print(xs.shape)
                    xs = xs.unsqueeze(0).repeat(xd.size(0), 1, )
                else:
                    xs = xs.unsqueeze(0).repeat(xd.size(0), 1, 1, 1)
            return xd, xs, y
        else:
            return xd, torch.tensor([]), y
        


class PyramidDataset(Dataset):

    def __init__(self, 
                 forcing: xr.Dataset, # time, lat, lon 
                 predictor:xr.Dataset, # time, lat, lon
                 target: xr.Dataset = None,  # lat, lon
                 mask = None,
                 downsampler = None,
                 normalizer_forcing = None,
                 normalizer_predictor = None,
                 normalizer_target = None,
                 shape:tuple = (), # time ,lat ,lon
                 batch_size:dict = {"xsize":20, "ysize":20, "tsize":20},
                 overlap:dict = {"xover":0, "yover":0, "tover":0},
                 missing_policy: str | float = "all",
                 fill_missing = 0, 
                 persist=False, 
                 lstm_1d = False, 
                 static_to_dynamic=False
                 ):
        """
        CAN I USE IT FOR PARAM LEARNING?

        IN THE CASE I WANT TO PRECOMPUTE THE DUMMY TIME DIMENSION FOR THE STATIC PARAMETERS I COULD MERGE XS WITH XD AND PASS IT AS XD AND SET XS=NONE
        """
        
        self.forcing = forcing
        self.target = target
        self.predictor = predictor

        self.shape = shape

        self.mask = mask

        self.missing_policy = missing_policy

        self.downsampler = downsampler

        KEEP_DEGENERATE_CUBELETS = False # TODO: hardcoded
        
        # compute stuff

        self.cbs_spatial_idxs, self.cbs_missing_idxs, self.cbs_degenerate_idxs, self.cbs_spatial_slices = compute_cubelet_spatial_idxs(shape, 
                                                                                                                   batch_size['xsize'], 
                                                                                                                   batch_size['ysize'], 
                                                                                                                   overlap['xover'], 
                                                                                                                   overlap['yover'], 
                                                                                                                   KEEP_DEGENERATE_CUBELETS,
                                                                                                                   masks = self.mask,
                                                                                                                   missing_policy=self.missing_policy) 

        self.cbs_time_idxs, self.cbs_degenerate_idxs, self.cbs_time_slices = compute_cubelet_time_idxs(shape, 
                                                                                        batch_size['tsize'],
                                                                                        overlap['tover'], 
                                                                                        KEEP_DEGENERATE_CUBELETS,
                                                                                        masks = self.mask)

        #print(self.cbs_spatial_idxs)
        
        cbs_tuple_idxs = compute_cubelet_tuple_idxs(self.cbs_spatial_idxs, self.cbs_time_idxs)
        cbs_slices = compute_cubelet_slices(self.cbs_spatial_slices, self.cbs_time_slices)

        self.cbs_mapping_idxs = cbs_mapping_idx_slice(cbs_tuple_idxs, cbs_slices)
        
        if self.downsampler is not None:
            # DOWNSAMPLE THE REMAINING INDEXES AFTER REMOVING MISSING AND DEGENERATED
            # return a subset of the cbs_mapping_idxs
            #TODO: also self.cbs_time_idxs and self.cbs_spatial_idxs should be updated
            self.cbs_mapping_idxs = self.downsampler.sampling_idx(self.cbs_mapping_idxs)
            
        
        if normalizer_forcing is not None:
            # this normalize the data corresponding to valid indexes
            if normalizer_forcing.stats_iscomputed: # validation or test
                self.forcing = normalizer_forcing.normalize(self.forcing)
            else:
                # compute stats for training
                normalizer_forcing.compute_stats(self.forcing)
                self.forcing = normalizer_forcing.normalize(self.forcing)

        if normalizer_predictor is not None:
            #import pdb;pdb.set_trace()
            if normalizer_predictor.stats_iscomputed: # validation or test
                self.predictor = normalizer_predictor.normalize(self.predictor)
            else:
                normalizer_predictor.compute_stats(self.predictor)   
                self.predictor = normalizer_predictor.normalize(self.predictor)

        if normalizer_target is not None: 
            if normalizer_target.stats_iscomputed: # validation or test
                self.target = normalizer_target.normalize(self.target)
            else:
                normalizer_target.compute_stats(self.target)
                self.target = normalizer_target.normalize(self.target)

        # either do this here or in the getitem
        # maybe add HERE THE RASHAPE IN CASE IS LSTM 1D
        self.forcing = self.forcing.to_stacked_array( new_dim="feat", sample_dims = ["time", "lat", "lon"]) # time, lat, lon , feat
        self.forcing = self.forcing.transpose("time", "feat", "lat" , "lon") # T C H W
        self.forcing = self.forcing.astype("float32")

        self.target = self.target.to_stacked_array( new_dim="feat", sample_dims = ["time", "lat", "lon"])
        self.target = self.target.transpose("time", "feat", "lat" , "lon") # T C H W
        self.target = self.target.astype("float32")

        self.predictor = self.predictor.to_stacked_array( new_dim="feat", sample_dims = ["lat", "lon"]) # H W C
        self.predictor = self.predictor.transpose("feat", "lat", "lon")
        self.predictor = self.predictor.astype("float32")

        if persist:
            self.forcing = self.forcing.compute()
            self.target = self.target.compute()
            self.predictor = self.predictor.compute()

        # The conversion of missing nans should occur at the very end. The nans are used in the compute_cubelets_*_idxs to remove cubelets with all nans
        # should the missing flag not be equal to a potential valid value for that quantity? for example zero may be valid for many geophysical variables
        self.forcing = self.forcing.fillna(fill_missing)
        
        self.target = self.target.fillna(fill_missing)
    
        self.predictor = self.predictor.fillna(fill_missing)

        self.lstm_1d = lstm_1d
        self.static_to_dynamic = static_to_dynamic

        # top layer resolution
        self.top_layer_res = -1*self.forcing.lat.diff("lat").values[0] / 2

        # expand static to dynamic 
        #time_da = xr.DataArray(dynamic.time.values, [('time', dynamic.time.values)])
        #static = static.expand_dims({"time":time_da})

    def __len__(self):
        return len(self.cbs_mapping_idxs)

    def get_indexes(self):
        return list(range(len(self.cbs_mapping_idxs)))
    
    def __getitem__(self, index):
        
        cubelet_idx = list(self.cbs_mapping_idxs.keys())[index]
        
        #print(index, cubelet_idx)

        time_slice = self.cbs_mapping_idxs[cubelet_idx]["time"]
        lat_slice =  self.cbs_mapping_idxs[cubelet_idx]["lat"]
        lon_slice = self.cbs_mapping_idxs[cubelet_idx]["lon"]

        # xr.Dataarray to np.ndarray, this triggers loading in memory, in case persist = False
        forcing = self.forcing[time_slice,:, lat_slice, lon_slice] #.values # L C H W
        target = self.target[time_slice,:, lat_slice, lon_slice] #.values # L C H W

        latmin = forcing.lat.values.min()
        latmax = forcing.lat.values.max()
        lonmin = forcing.lon.values.min()
        lonmax = forcing.lon.values.max()

        predictor = self.predictor.sel( 
                          lat = slice( latmax + self.top_layer_res, 
                                latmin - self.top_layer_res),
                          lon = slice(lonmin - self.top_layer_res,
                                lonmax + self.top_layer_res)
        ).values 
        
        forcing = forcing.values
        target = target.values

        # np.ndarray ot torch.tensor
        forcing = torch.FloatTensor(forcing)
        target = torch.FloatTensor(target)
        predictor = torch.FloatTensor(predictor)
            
        if self.lstm_1d:
            # Super slow when persist == False
            # If True means that the xsize and ysize is equal to 1
            
            #xd = xd.flatten(2,3) # L C H W => L C N
            xd = xd.squeeze() # L C H W, but H W is size 1,1 => L C
            #xd = torch.permute(xd, (2, 0, 1)) # N L C, , but N = 1
            #xd = x.squeeze(0)
            
            #y = y.flatten(2,3) # L C H W => L C N
            y = y.squeeze() # L C H W, but H W is size 1,1 => L C
            #y = torch.permute(y, (2, 0, 1)) # N L C, but N = 1
            #y = y.squeeze(0)
            if self.xs is not None:
                xs = xs.squeeze() # C H W => C N  

        if self.static_to_dynamic:
            if self.lstm_1d:
                #print(xs.shape)
                predictor = predictor.unsqueeze(0).repeat(forcing.size(0), 1, )
            else:
                predictor = predictor.unsqueeze(0).repeat(forcing.size(0), 1, 1, 1)
        return predictor, forcing, target


DATASETS = {"LSTMDataset": LSTMDataset,
             "XBatchDataset":XBatchDataset,
             "CubeletsDataset":CubeletsDataset}


def get_dataset(dataset):
    return DATASETS.get(dataset)
