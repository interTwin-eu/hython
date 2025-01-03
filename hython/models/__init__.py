import torch
from torch import nn


class BaseModel(nn.Module):
    def rescale_input(self, param):
        return torch.sigmoid(param)


from .cudnnLSTM import CuDNNLSTM

MODELS = {"cudalstm": CuDNNLSTM}


def get_model(model):
    return MODELS.get(model)
