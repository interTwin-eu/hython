""" 
TODO

- Handling the return of sampling metadata in function split_time_and_space 
- The sampling indices are passed to the data loader
This means the training dataset is sampled, which means we need to reshape the whole dataset (only once though)

"""

import matplotlib.pyplot as plt
import numpy as np 
from hython.sampler import RegularIntervalSampler
import torch 
a = np.sin(np.arange(1_500_000).reshape(100,100,50,3))
a.shape

print(a.nbytes / 1e9, "GB")

a_training = a.reshape((10_000, 50, 3))
a_training.shape

sampler = RegularIntervalSampler((10, 10), (3, 3))

a_samp, meta = sampler.sampling(a)

a_samp.shape

meta.idx_sampled_1d


np.array([
    [0, 1, 2, 3, 4],
    [5, 6, 7, 8, 9],
    [10, 11, 12, 13,14],
    [15, 16, 17, 18, 19]
])[np.array([0, 2])[:, None], np.array([1, 3])] # 0, 2 row ; 1 , 3 column

# broadcasting along 3rd axis
np.arange(75).reshape(5,5,3)[np.array([0, 2])[:, None], np.array([1, 3])].shape

# split validation test by n of years 
samp_val = a_samp[:,:,:25]
samp_test = a_samp[:,:,25:]

samp_val.shape 
samp_test.shape

# Pytorch sampler



from torch.utils.data import Dataset, DataLoader



class DemoDataset(Dataset):
    
    def __init__(self, arr):
        self.arr = arr
        
    def __len__(self):
        return len(self.arr)
    
    def __getitem__(self, idx):
        sample = self.arr[idx]
        return sample
    
    
ds = DemoDataset(a_training)

for i in range(10):
    print(ds[i].shape)


    
dl = DataLoader(ds, batch_size=2)


next(iter(dl)).shape

from torch.utils.data.sampler import Sampler, RandomSampler

sampler = RandomSampler(ds, num_samples=4)

dl = DataLoader(ds, batch_size=2, sampler=sampler)


next(iter(dl)).shape

validation_type = "time"

def split_time(a, slice):
    return a[:,:,slice]


def split_space(a, sampler):
    return sampler.sampling(a)



def split_time_and_space(a, 
                         validation_type, 
                         temporal_train_slice = None, 
                         temporal_val_slice = None, 
                         spatial_train_sampler = None,
                         spatial_val_sampler = None):
    # split time
    if validation_type == "space" or validation_type == "spacetime":
        a_train = split_space(a, spatial_train_sampler)[0]
        a_val = split_space(a, spatial_val_sampler)[0]
    
    if validation_type == "time":
        a_train = split_time(a, temporal_train_slice)
        a_val = split_time(a, temporal_val_slice)

    if validation_type == "spacetime":
        a_train = split_time(a_train, temporal_train_slice)
        a_val = split_time(a_val, temporal_val_slice)   

    orig_shape = a.shape
    train_shape = a_train.shape 
    val_shape = a_val.shape
    print(f"""Approach {validation_type}: \n
          Original dataset: (lat {orig_shape[0]}, lon {orig_shape[1]} , time {orig_shape[2]} , feature {orig_shape[3]}) \n
          Train dataset: (lat {train_shape[0]}, lon {train_shape[1]} , time {train_shape[2]} , feature {train_shape[3]}) \n
          Validation dataset: (lat {val_shape[0]}, lon {val_shape[1]} , time {val_shape[2]} , feature {val_shape[3]})""")
        
    return a_train, a_val
    

train, val = split_time_and_space(a, 
                     validation_type = "time",
                     temporal_train_slice=slice(0,35),
                     temporal_val_slice= slice(35,50))

train.shape, val.shape 


train, val = split_time_and_space(a, 
                     validation_type = "space",
                     spatial_train_sampler = RegularIntervalSampler((5,5), (0,0)),
                     spatial_val_sampler = RegularIntervalSampler((5,5), (3,3)) )


train.shape, val.shape 

train, val = split_time_and_space(a, 
                     validation_type = "spacetime",
                     temporal_train_slice=slice(0,35),
                     temporal_val_slice= slice(35,50),
                     spatial_train_sampler = RegularIntervalSampler((5,5), (0,0)),
                     spatial_val_sampler = RegularIntervalSampler((5,5), (3,3)) )

train.shape, val.shape 

# class TimeSampler(Sampler):
    
#     def __init__(self, data_source, num_samples= 10, generator = None):
        
#         self.data_source = data_source
#         self._num_samples = num_samples
#         self.generator = generator
        
#     @property
#     def num_samples(self) -> int:
#         # dataset size might change at runtime
#         if self._num_samples is None:
#             return len(self.data_source)
#         return self._num_samples

#     def __iter__(self):
#         n = len(self.data_source)
#         yield from torch.randint(high=n, size=(self.num_samples % 32,), dtype=torch.int64, generator=generator).tolist()
    
#     def __len__(self) -> int:
#         return self.num_samples
    


# seed = int(torch.empty((), dtype=torch.int64).random_().item())
# generator = torch.Generator()
# generator.manual_seed(seed)


class SpaceSampler(Sampler):
    
    def __init__(self, data_source, num_samples= 10, generator = None, sampling_indices = None):
        
        self.data_source = data_source
        self._num_samples = num_samples
        self.generator = generator
        self.sampling_indices = sampling_indices
        
    @property
    def num_samples(self) -> int:
        # dataset size might change at runtime
        if self._num_samples is None:
            return len(self.data_source)
        return self._num_samples

    def __iter__(self):
        n = len(self.data_source)
        #yield from torch.randint(high=n, size=(self.num_samples % 32,), dtype=torch.int64, generator=generator).tolist()
        yield from self.sampling_indices
    
    def __len__(self) -> int:
        return self.num_samples
    
    

# return sampling indices in 1d (so that I can use in a_trainin shape)
s1 = RegularIntervalSampler((5,5), (2,2))

_, meta = s1.sampling(a)

meta

ds = DemoDataset(a_training)


sampler = SpaceSampler(ds, num_samples=5, sampling_indices = meta.idx_sampled_1d.tolist())

a_training.shape
meta.idx_sampled_1d.shape

dl = DataLoader(ds, batch_size=10, sampler=sampler)

# iterate over 400 (the subset)
c = 0
for i,v in enumerate(dl):
    print(v.shape)
    c += 1*v.shape[0]
print(c)


