import torch
from torch import nn

class Hybrid(nn.Module):

    def __init__(self, transfernn, head_layer, rescale_transf_out=True, freeze_head=True):
        super(Hybrid, self).__init__()

        self.rescale_flag = rescale_transf_out
        
        self.transfernn = transfernn
        self.head_layer = head_layer

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

        # the order of the param does not matter, the rescaling will inform what is what

        # rescale to head_layer
        if self.rescale_flag: param = self.head_layer.rescale(param) # output: N T C or N C

        # concat to x_head, as of now add time dimension ot static params
        x_head_concat = torch.concat([
                            param.unsqueeze(1).repeat(1, x_head.size(1), 1),
                            x_head], dim=2)
        
        # run head layer
        output = self.head_layer(x_head_concat)

        return output, param