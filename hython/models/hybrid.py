import torch
from torch import nn

class Hybrid(nn.Module):

    def __init__(self, transfernn, head_layer, freeze_head=True, head_input_rescaler = None, scale_factor = 5.):
        super(Hybrid, self).__init__()
        
        self.transfernn = transfernn
        self.head_layer = head_layer
        self.rescaler = head_input_rescaler
        self.scale_factor = nn.Parameter(torch.tensor(scale_factor).float())

        # freeze weights
        if freeze_head:
            for weight in self.head_layer.parameters(): 
                weight.requires_grad = False
            
    def forward(self, x_transf, x_head):
        """

        Parameters
        ----------
        x_transf: torch.Tensor
            Tensor of size [batch_size, n_predictor] (N, C) or [batch_size, seq_length, n_predictor] (N, T, C)
        x_head: torch.Tensor
            Tensor of size [batch_size, seq_length, n_param] (N, T, C)

        """
        # run trasnferNN
        param = self.transfernn(x_transf) # output: N T C or N C

        #print(param.min(0)[0],param.max(0)[0])
        # the order of the param does not matter, the rescaling will inform what is what

        # rescale to head_layer
        #print(self.scale_factor)

        if self.rescaler is not None: param = self.rescale(param, self.scale_factor) # output: N T C or N C
        #print("after: ", param.min(0)[0],param.max(0)[0])
        # concat to x_head, as of now add time dimension ot static params
        x_head_concat = torch.concat([
                            x_head,
                            param.unsqueeze(1).repeat(1, x_head.size(1), 1),
                            ], dim=2)
        
        # run head layer
        output = self.head_layer(x_head_concat)
        #print("output: ", output[:,-1])
        return output, param