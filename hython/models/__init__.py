import torch
from torch import nn


class BaseModel(nn.Module):
    def rescale_input(self, param):
        return torch.sigmoid(param)


from .cudnnLSTM import CuDNNLSTM
from .hybrid import Hybrid
from .transferNN import TransferNN

MODELS = {
    "cudalstm": CuDNNLSTM,
    "hybrid":Hybrid,
    "transfernn":TransferNN
          }


def get_model(model):
    return MODELS.get(model)
