import torch 
from torch import nn

import torch.nn.functional as F


class ParamLearner():
    def __init__(self, model):
        self.model = model
    def train(self):
        self["transfer_nn"].train()
    def eval(self):
        self["transfer_nn"].eval()
    def __getitem__(self, key):
        try:
            return self.model[key]
        except KeyError:
            pass
        try:
            return self.model[key]
        except KeyError:
            pass
        raise KeyError(key)

    def __setitem__(self, key, value):
        self.model[key] = value
        
    def to(self, device):
        self["transfer_nn"] = self["transfer_nn"].to(device)
        self["surrogate"] = self["surrogate"].to(device)

    def freeze(self, model_name):
        for param in self[model_name].parameters():
            param.requires_grad = False

    def state_dict(self,model_name = "transfer_nn"):
        return self[model_name].state_dict()


    def load_state_dict(self, fp, model_name = "transfer_nn"):
        self[model_name].load_state_dict(fp)


class TransferNN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(TransferNN, self).__init__()

        self.lin = nn.Linear(input_dim, output_dim)
        self.lin2 = nn.Linear(output_dim, 3)
        self.lin3 = nn.Linear(3, 3)
        self.lin4 = nn.Linear(3, output_dim)
    
        for p in self.parameters():
            if isinstance(p,nn.Linear):
                torch.nn.init.xavier_uniform_(p)
            #print(p)
    def forward(self, x):
        
        l1 = F.leaky_relu(self.lin(x))
        l2 = F.leaky_relu(self.lin2(l1))
        l3 = F.leaky_relu(self.lin3(l2))
        l4 = self.lin4(l3)
        
        return l4

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