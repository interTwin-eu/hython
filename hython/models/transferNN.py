from torch.nn import Linear, Sequential, ModuleDict
import torch
import torch.nn.functional as F
from torch import nn


def make_mlp(input_dim, output_dim, hidden_dim, n_layers, bias=False):
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

    #layers = layers + [nn.LeakyReLU()]

    mlp = Sequential(*layers)
    return mlp


def stack_mlps(params, input_dim, output_dim, hidden_dim, n_layers, bias=False):
    dct = {}
    for param in params:
        dct[param] = make_mlp(input_dim, output_dim, hidden_dim, n_layers, bias=False)
    module_dict = ModuleDict(dct)
    return module_dict


class TransferNN(nn.Module):
    def __init__(self, params, input_dim, output_dim, hidden_dim, n_layers, bias=False):
        super(TransferNN, self).__init__()

        self.params = params
        # make mlp layers
        self.mlp_dict = stack_mlps(
            params, input_dim, output_dim, hidden_dim, n_layers, bias=False
        )

    def forward(self, x):
        out = []
        for param in self.params:
            out.append(F.sigmoid(self.mlp_dict[param](x)))

        return torch.cat(out, -1).float()


# class TransferNN(nn.Module):
#     def __init__(self, input_dim, output_dim, hidden_dim):
#         super(TransferNN, self).__init__()

#         self.lin = nn.Linear(input_dim, output_dim)
#         self.lin2 = nn.Linear(output_dim, hidden_dim)
#         self.lin3 = nn.Linear(hidden_dim, hidden_dim)
#         self.lin4 = nn.Linear(hidden_dim, output_dim)

#         for p in self.parameters():
#             if isinstance(p, nn.Linear):
#                 torch.nn.init.xavier_uniform_(p)

#     def forward(self, x):
#         l1 = F.leaky_relu(self.lin(x))
#         l2 = F.leaky_relu(self.lin2(l1))
#         l3 = F.leaky_relu(self.lin3(l2))
#         l4 = self.lin4(l3)

#         return l4


# class TransferNN(nn.Module):
#     def __init__(self, input_ch, latent_dims, output_dim, shape_bottom, shape_top):
#         super(TransferNN, self).__init__()
#         self.shape_bottom = shape_bottom
#         self.shape_top = shape_top
#         self.conv1 = nn.Conv2d(input_ch, latent_dims, 5, stride=1, padding=1,padding_mode="reflect") # formula?
#         self.ad1 = nn.AdaptiveAvgPool2d( self.shape_bottom)
#         self.conv2 = nn.Conv2d(latent_dims, output_dim, 5, stride=1, padding=1,padding_mode="reflect")
#         self.ad2 = nn.AdaptiveAvgPool2d( self.shape_top )
#         #self.batch2 = nn.BatchNorm2d(16) # what is this doing?
#         #self.conv3 = nn.Conv2d(16, 32, 5, stride=1, padding=1)
#         # self.linear1 = nn.Linear(3*3*32, 128)
#         # self.linear2 = nn.Linear(128, latent_dims)
#         # self.linear3 = nn.Linear(128, latent_dims)
#         # self.linear4 = nn.Linear(1,1)

#         # self.N = torch.distributions.Normal(0, 1)
#         # self.N.loc = self.N.loc.cuda()
#         # self.N.scale = self.N.scale.cuda()
#         # self.kl = 0

#     def forward(self, x):
#         #x = x.to(device) # why
#         x = F.relu(self.conv1(x))
#         x = self.ad1(x)
#         x = self.conv2(x)
#         x = self.ad2(x)#  F.relu(self.batch2(self.conv2(x)))
#         x = F.relu(x)
#         #x = self.linear4(x.permute(1,2,0))
#         # x = F.relu(self.conv3(x))
#         # x = torch.flatten(x, start_dim=1)
#         # print(x.shape)
#         # x = F.relu(self.linear1(x))
#         # mu = self.linear2(x) # whys
#         # sigma = torch.exp(self.linear3(x)) # why
#         # z = mu + sigma*self.N.sample(mu.shape) # why
#         # self.kl = (sigma**2 -torch.log(sigma) -1/2).sum() # why
#         return x
