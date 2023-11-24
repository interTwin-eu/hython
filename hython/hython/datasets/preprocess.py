import torch
from torch import nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np


# check the longitude/latitude 
def reshape_arr(ds_arr, lat_arr, lon_arr, time_index, number_stats_parameters=None):
    if number_stats_parameters is not None:
        t_index = number_stats_parameters #- 1
    else:
        t_index = time_index 
        
    Lon_Index = lon_arr.shape[0] - 1
    Lat_Index = lat_arr.shape[0] - 1
    Pixels_Index = (Lon_Index * Lat_Index) -1
    
    Input_arr = ds_arr.copy()

    if lat_arr[0]>lat_arr[1]:
        for l in range(t_index):
            for j in range(Lon_Index):
                M=Lat_Index
                for i in range(Lat_Index):
                    Input_arr[l,i,j]=ds_arr[l,M,j]
                    M=M-1
                    
    if lon_arr[0]>lon_arr[1]:
        for l in range(t_index):
            for j in range(Lon_Index):
                M=Lon_Index
                for i in range(Lat_Index):
                    Input_arr[l,i,j]=ds_arr[l,i,M]
                    M=M-1
                    
    Input_arr=Input_arr.reshape(t_index,-1)
    Input_arr=Input_arr.T

    return Input_arr


# Mask the nans values
def mask_missing_pixels(dyn_arr, stats_arr, target_arr, time_steps=365):
    missing_all = []
    for i in range(dyn_arr.shape[0]):
        s = np.all(np.isnan(stats_arr[i]))
        d = np.all(np.isnan(dyn_arr[i]))

        if s or d:
            missing_all.append(i)

    # Now filter the data based on the available indices 
    other_indices = np.setdiff1d(np.arange(len(dyn_arr)), missing_all)

    filtered_target = target_arr[other_indices]
    filtered_stats = stats_arr[other_indices]
    filtered_dyn = dyn_arr[other_indices]
    
    # Now filter based on the target 
    nan_index = []
    index = []
    for i in range(filtered_target.shape[0]):
        #if np.isnan(filtered_target[i]).sum() == time_steps:
        if np.isnan(filtered_target[i]).sum() > 1:
            nan_index.append(i)
        else:
            index.append(i)

            
    filtered_target = filtered_target[index]
    filtered_stats = filtered_stats[index]
    filtered_dyn = filtered_dyn[index]
    
    # fill missing values with mean
    mean_values = np.nanmean(filtered_stats, axis=0, keepdims=True)
    nan_indices = np.isnan(filtered_stats)
    filtered_stats = np.where(nan_indices, mean_values, filtered_stats)
    
    nan_feats = np.all(np.isnan(filtered_stats), axis=0)
    filtered_stats = filtered_stats[:, ~nan_feats]
    
    #filtered_stats[np.isnan(filtered_stats)] = -9999
    #filtered_dyn[np.isnan(filtered_dyn)] = -9999
    
    return filtered_target, filtered_stats, filtered_dyn

# Convert xarray to array
def xarray_to_array(ds, var_names,time_steps=365, number_stats_parameters= None, reshape=True, dynamic_vars = True):
    ds_list = []
    #for varname, var in ds.variables.items():
    for varname in var_names:
        #if str(varname) in var_names:
        var = ds[varname]
        arr = var.values
        if not reshape:
            arr = arr.reshape((time_steps, -1))
            arr = arr.T
        #else:
         #   arr = arr.reshape(-1)
        if number_stats_parameters is not None: 
            if arr.ndim == 3:
                for dim in range(arr.shape[0]):
                    ds_list.append(arr[dim])
            else:
                ds_list.append(arr)
        else:
            ds_list.append(arr)
            
    #Latitude
    lat_var = ds['lat']
    lat_arr=lat_var.values
        
    #Longitude
    lon_var = ds['lon']
    lon_arr=lon_var.values
        
        
    ds_arr = np.stack(ds_list, axis=-1)
    if reshape:
        if dynamic_vars:
            final_arrs = []
            for d in range(ds_arr.shape[-1]):
                c_arr = ds_arr[:,:,:,d]
                reshaped_arr = reshape_arr(c_arr, lat_arr, lon_arr, time_index=time_steps , number_stats_parameters=number_stats_parameters )
                final_arrs.append(reshaped_arr)

            final_arrs = np.stack(final_arrs, axis=-1)
            
        else:
            ds_arr = np.transpose(ds_arr, (2, 0, 1))
            reshaped_arr = reshape_arr(ds_arr, lat_arr, lon_arr, time_index=time_steps , number_stats_parameters=len(ds_list) )
            final_arrs = reshaped_arr.copy()
            
        
    else:
        final_arrs = ds_arr.copy()
        
    return final_arrs #torch.Tensor(final_arrs) 




# Normalize the time series
def normalize_time_series(time_series_data):
    
    # Normalize each feature's time series separately for each sample
    normalized_time_series = np.zeros_like(time_series_data)
    
    
    for sample_idx in range(time_series_data.shape[0]):
        data = time_series_data[sample_idx]
        
        # Compute the mean and standard deviation along the time steps axis
        mean = np.mean(data, axis=0, keepdims=True)
        std = np.std(data, axis=0, keepdims=True)
        
        # Normalize the time series
        normalized_data = (data - mean) / std
    
        normalized_time_series[sample_idx] = normalized_data
    
    return normalized_time_series
    
    
# Normalize the static data
def normalize_static_data(static_data):
    # Compute the mean and standard deviation along the time steps axis
    mean = np.mean(static_data, axis=0, keepdims=True)
    std = np.std(static_data, axis=0, keepdims=True)
        
    # Normalize the time series
    normalized_data = (static_data - mean) / std
    
    return normalized_data    
    
    
    
    
    
    
    
    
    
    