import torch 
from torch import nn


class BaseModel(nn.Module):
    def rescale_input(self, param):
        return torch.sigmoid(param)
