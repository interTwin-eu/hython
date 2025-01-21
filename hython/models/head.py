from torch import nn
import torch
import torch.nn.functional as F
from .utils import make_mlp


def get_head_layer(layer, input_dim, output_dim, head_activation = "linear", **kwargs):

    if layer == "regression":
        return RegressionHead(input_dim, output_dim, head_activation = head_activation, **kwargs)
    elif layer == "distr_normal":
        return NormalDistrHead(input_dim, output_dim, head_activation = head_activation, **kwargs)


class RegressionHead(nn.Module):
    
    def __init__(self, input_dim, output_dim, hidden_dim = None, n_layers = 1, head_activation = "linear", bias= False):
        super(RegressionHead, self).__init__()

        self.net = make_mlp(input_dim, output_dim, hidden_dim, n_layers, bias, head_activation)

    def forward(self, x):
        return {"y_hat":self.net(x)}
    


class NormalDistrHead(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim = None, n_layers = 1, head_activation = "linear", bias= False):
        super(NormalDistrHead, self).__init__()

        self.mu = make_mlp(input_dim, output_dim, hidden_dim, n_layers, bias, head_activation)
        self.sigma = make_mlp(input_dim, output_dim, hidden_dim, n_layers, bias, head_activation)

    def forward(self, x):
        return {"mu":self.mu(x), "sigma": torch.abs(self.sigma(x))}
