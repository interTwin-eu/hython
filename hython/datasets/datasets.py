import torch
import numpy as np
import numpy.typing as npt
from torch.utils.data import Dataset

try:
    import xbatcher
except:
    pass

class LSTMDataset(Dataset):
    def __init__(
        self,
        xd: torch.Tensor | npt.NDArray,
        y: torch.Tensor | npt.NDArray,
        xs: torch.Tensor | npt.NDArray = None,
        seq_length: int = 60,
        create_seq: bool = False,
        convolution: bool = False,
    ):
        """Create a dataset for training and validating LSTM-based models

        LSTM assumes shape (C, T, F), C is the gridcell (is the dimension that is going to be mini-batched)

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
        convolution : bool, optional
            Reshape data sample as LSTM+Convolution requires (B, T, X, Y, F) , by default False
        """
        self.seq_len = seq_length
        self.create_seq = create_seq
        self.convolution = convolution

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
            self.Xd = xd
            self.y = y

    def __len__(self):
        """Returns examples size. If LSTM number of gridcells. If ConvLSTM number of images."""
        return self.Xd.shape[0]

    def __getitem__(self, i):
        if self.xs is not None:
            # case when there are static values
            if self.create_seq:
                return self.Xd[i], self.xs, self.y[i]
            else:
                if self.convolution:
                    return self.Xd[i], self.xs[i], self.y[i]
                else:
                    return self.Xd[i], self.xs[i], self.y[i]
        else:
            return self.Xd[i], self.y[i]

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
                xs = self.xs_gen[index].torch.to_tensor() 
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
class BasinDataset(Dataset):
    def __init__(
        self,
        xd: torch.Tensor | npt.NDArray,
        y_distributed: torch.Tensor | npt.NDArray,  # et, sm, r
        y_lumped: torch.Tensor | npt.NDArray = None,  # discharge
        xs: torch.Tensor | npt.NDArray = None,
    ):
        """Create a dataset for training and validating LSTM-based models

        LSTM assumes shape (C, T, F), C is the gridcell (is the dimension that is going to be mini-batched)

        Parameters
        ----------
        xd : torch.Tensor | npt.NDArray
            Dynamic parameter
        y_lumped : torch.Tensor | npt.NDArray
            Target, represent the river discharge at the basin's outlet. (B, T, F)
        y_distributed : torch.Tensor | npt.NDArray
            Target
        xs : torch.Tensor | npt.NDArray, optional
            Static parameter, by default None
        """

        if y_lumped is None:
            self.islumped = False
        else:
            self.islumped = True

        num_gridcells, num_samples, num_features = xd.shape

        if isinstance(xs, np.ndarray):
            self.xs = xs.astype(np.float32)
        else:
            self.xs = xs

        # integrated
        self.y_lump = y_lumped
        # distributed
        self.Xd = xd
        self.y_distr = y_distributed

    def __len__(self):
        """Returns examples size"""
        return self.Xd.shape[0]

    def get_lumped_target(self):
        return self.y_lump

    def __getitem__(self, i):
        """Returns dataset for cases:
        - islumped
            dynamic, static and distributed targets are (C, T, F)
            lumped targets is (B, T, F), where B = basin and F = 1
        - not lumped
            dynamic, static and distributed targets are (C, T, F)
        """
        if self.xs is not None:
            # case when there are static values
            return self.Xd[i], self.xs[i], self.y_distr[i]
        else:
            return self.Xd[i], self.y[i]


# TODO: extend to handle all basins of the InterTwin domain
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


class GraphDataset(Dataset):
    """
    Generate a graph dataset.
    The getitem returns a snapshot.

    """

    pass


DATASETS = {"LSTMDataset": LSTMDataset,
             "XBatchDataset":XBatchDataset}


def get_dataset(dataset):
    return DATASETS.get(dataset)
