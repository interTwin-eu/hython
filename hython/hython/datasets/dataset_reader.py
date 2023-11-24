import xarray as xr
import torch
from torch import nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split

import numpy as np

from .preprocess import xarray_to_array, mask_missing_pixels, normalize_time_series, normalize_static_data



def create_dataset(dyn_vars_ds, static_params_ds, target_ds, time_steps=365,
                   batch_size=8, shuffle=True, normalization =False, torch_ds=True,
                   dyn_vars_names=['precip', 'pet', 'temp'],
                   static_params_names=[ 'M', 'thetaS', 'RootingDepth', 'Kext', 'Sl', 'Swood', 'TT', 'KsatHorFrac'],
                   target_names=['vwc_percroot']):
    
    """THIS Function READ THE DATASETS, TTHEN RETURN THEM IN READABLE FORMAT FOR TIMESERIES MODELS.
    
    Parameters:
        predictors: :class:`xarray.Dataset`     
                   fvfdjvnfkjdv
                   #data reader inherited from torch.utils.data.Dataset for train/validation partitions
        
        static_parameters: :class:`xarray.Dataset`
                          ffdfd
        
        target_ds: :class:`xarray.Dataset`
        
          
    Returns:
        Arrays:   
    """
    datasets = {'Dynamic Variables': dyn_vars_ds, 'Static Parameters': static_params_ds, 'targets': target_ds}
    
    
    # Check the inputs types
    for name, ds in datasets.items():
        assert isinstance(ds, xr.Dataset), f"The input '{name}' is not an xarray.Dataset."
            
    # Read the inputs and extract the data arrays values 
    ## Reshape the target
    
    target_arr = xarray_to_array(target_ds, var_names = target_names, time_steps=time_steps, reshape=False)
    print(target_arr.shape)
    print('Done target parmas')
    
    ## Reshape the dynamics 
    dyn_vars_arr = xarray_to_array(dyn_vars_ds, var_names = dyn_vars_names, time_steps=time_steps)
    print(dyn_vars_arr.shape)
    print('Done dynamics parmas')
    
    ## Reshape the statics
    number_stats_parameters= len(static_params_names) + 14
    static_params_arr = xarray_to_array(static_params_ds, var_names = static_params_names, time_steps=time_steps,number_stats_parameters=number_stats_parameters, dynamic_vars = False)
    print(static_params_arr.shape)
    print('Done static parmas')
    
    
 #   return target_arr, dyn_vars_arr, static_params_arr 
    
    # Mask nans
    target_arr, static_params_arr, dyn_vars_arr = mask_missing_pixels(dyn_vars_arr, static_params_arr, target_arr, time_steps=time_steps)
        
    # Convert the data to PyTorch DataLoader
    print(np.isnan(target_arr).sum())
    print(np.isnan(static_params_arr).sum())
    print(np.isnan(dyn_vars_arr).sum())
    
    # Normalize the data
    if normalization:
        dyn_vars_arr  = normalize_time_series(dyn_vars_arr)
        static_params_arr = normalize_static_data(static_params_arr)
    
    target_arr = torch.Tensor(target_arr) 
    static_params_arr = torch.Tensor(static_params_arr) 
    dyn_vars_arr = torch.Tensor(dyn_vars_arr) 
    #print(dyn_vars_arr.shape)
    
    if torch_ds:
        dataset = TensorDataset(dyn_vars_arr, static_params_arr, target_arr)

        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])


        # PyTorch DataLoader
        #data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        return train_loader, val_loader #data_loader
    
    else:
        return target_arr, static_params_arr, dyn_vars_arr
