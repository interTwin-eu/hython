import torch
from torch import nn


class BaseModel(nn.Module):

    def __init__(self, cfg = None):
        super(BaseModel, self).__init__()
        self.cfg = cfg

    def rescale_input(self, param):
        return param

    def rescale_output(self, data):
        return data


from .cudnnLSTM import CuDNNLSTM
from .hybrid import Hybrid
from .transferNN import TransferNN

MODELS = {"cudalstm": CuDNNLSTM, "hybrid": Hybrid, "transfernn": TransferNN}


def get_model(model):
    return MODELS.get(model)
