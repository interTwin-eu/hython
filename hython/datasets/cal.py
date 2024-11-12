from . import * 
from hython.preprocessor import reshape

class CalDataset(Dataset):
    def __init__(self, static_predictor, dynamic_input, observation, period, mask, seq_len = 120, downsample = None, frac = 0.01, norm_pred=None, norm_dyn=None, norm_obs=None):
        super().__init__()
        
        self.static = static_predictor
        
        self.dynamic = dynamic_input.sel(time=period)
        
        self.obs = observation.sel(time=period)

        self.mask = mask

        self.seq_len = seq_len
    
        # (1) apply 2d mask
        obs_masked = self.obs.where(mask)
        
        # (2) reshape
        self.obs = reshape(self.obs)
        obs_masked = reshape(obs_masked)
        
        # (3) find indices of valid
        valid_coords = np.argwhere(~(obs_masked.isnull()).values.squeeze(-1))
        
        # (4) avoid hitting bounds
        self.coords = valid_coords[valid_coords[:,1] > seq_len]

        # (5) reduce dataset size
        if downsample == "time":
            self.coords = downsample_time(self.coords, frac)
        elif downsample == "spacetime":
            self.coords = downsample_spacetime(self.coords, frac)

        gridcell_idx = np.unique(self.coords[:,0])
        time_idx = np.unique(self.coords[:,1])
        
        # (6) Normalize
        
        if norm_pred:
            if norm_pred.stats_iscomputed:  # validation or test
                self.static = norm_pred.normalize(self.static)
            else:
                # compute stats for training
                norm_pred.compute_stats(self.static.isel(gridcell=gridcell_idx))
                self.static = norm_pred.normalize(self.static)
        if norm_dyn:
            if norm_dyn.stats_iscomputed:  # validation or test
                self.dynamic = norm_dyn.normalize(self.dynamic)
            else:
                # compute stats for training
                norm_dyn.compute_stats(self.dynamic.isel(gridcell=gridcell_idx, time=time_idx))
                self.dynamic = norm_dyn.normalize(self.dynamic)
                
        if norm_obs:
            if norm_obs.stats_iscomputed:  # validation or test
                self.obs = norm_obs.normalize(self.obs)
            else:
                # compute stats for training
                norm_obs.compute_stats(self.obs.isel(gridcell=gridcell_idx, time=time_idx))
                self.obs = norm_obs.normalize(self.obs)
            
        
    def __len__(self):
        return len(self.coords)

    def __getitem__(self, i):
        
        gridcell, time = self.coords[i]

        time_delta = slice(time - self.seq_len + 1, time + 1)
        
        xs = self.static.isel(gridcell=gridcell).values
        xd = self.dynamic.isel(gridcell=gridcell, time=time_delta).values
        yo = self.obs.isel(gridcell=gridcell, time=time_delta).values

        return xs, xd, yo