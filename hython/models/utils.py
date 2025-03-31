from torch import nn
from torch.nn import Linear, Sequential, ModuleDict
import torch.nn.functional as F


def get_activation_layer(name):
    if name == "sigmoid":
        return nn.Sigmoid()
    elif name == "leaky_relu":
        return nn.LeakyReLU()
    elif name == "relu":
        return nn.ReLU()


def make_mlp(input_dim, output_dim, hidden_dim, n_layers, bias=False, output_activation_layer = "linear"):
    layers = []
    if n_layers == 1:
        layers.append(Linear(input_dim, output_dim, bias=bias))
    else:
        layers.append(Linear(input_dim, hidden_dim, bias=bias))

        layers = layers + [nn.LeakyReLU()]

        for i in range(n_layers - 2):
            layers.append(Linear(hidden_dim, hidden_dim, bias=bias))
            layers = layers + [nn.LeakyReLU()]

        layers.append(Linear(hidden_dim, output_dim, bias=bias))

    # output activation
    if output_activation_layer != "linear":
        layers = layers + [get_activation_layer(output_activation_layer)]

    mlp = Sequential(*layers)
    return mlp


def stack_mlps(params, input_dim, output_dim, hidden_dim, n_layers, bias=False, output_activation_layer = "linear"):
    dct = {}
    for param in params:
        dct[param] = make_mlp(input_dim, output_dim, hidden_dim, n_layers, bias, output_activation_layer)
    module_dict = ModuleDict(dct)
    return module_dict
