import torch
from torch import nn
import torch.nn.functional as F
from .head import RegressionHead, NormalDistrHead
from .base import BaseModel


class Hybrid(BaseModel):
    def __init__(
        self,
        transfernn,
        head_layer,
        freeze_head=True,
        scale_head_input_parameter=True,
        scale_head_output=False,
    ):
        super(Hybrid, self).__init__()

        self.transfernn = transfernn
        self.head_layer = head_layer
        self.scale_head_input_parameter = scale_head_input_parameter
        self.scale_head_output = scale_head_output

        # freeze weights
        if freeze_head:
            for weight in self.head_layer.parameters():
                weight.requires_grad = False
            self.head_layer.eval()
        else:
            for weight in self.head_layer.parameters():
                weight.requires_grad = True
            self.head_layer.train()

        if self.scale_head_output:        
            # initialize parameters for scaling output, TODO: move it into BaseModel
            self.scale = nn.Parameter( torch.ones(self.head_layer.output_size).float(), requires_grad=True)
            self.center = nn.Parameter( torch.zeros(self.head_layer.output_size).float() ,requires_grad=True)
            print("Scale and center for output rescaling: ", self.scale, self.center)
    def forward(self, x_predictor, x_head_dynamic, x_head_static = None):
        """
        Parameters
        ----------
        x_predictor: torch.Tensor
            Tensor of size [batch_size, n_predictor] (N, C) or [batch_size, seq_length, n_predictor] (N, T, C)
        x_head: torch.Tensor
            Tensor of size [batch_size, seq_length, n_param] (N, T, C)
        """
        # run trasnferNN
        
        param = self.transfernn(x_predictor)  # output: N T C  or N C

        if self.scale_head_input_parameter:
            param = self.rescale_input(param)  # output: N T C or N C

        # concat to x_head, same sequence of inputs of the surrogate model: concat(dynamic, static)
        x_head_concat = torch.concat(
            [
                x_head_dynamic,
                param.unsqueeze(1).repeat(1, x_head_dynamic.size(1), 1),
                x_head_static.unsqueeze(1).repeat(1, x_head_dynamic.size(1), 1),
            ],
            dim=2,
        )
        # run head layer
        head_output = self.head_layer(x_head_concat)
        output = {}
        if isinstance(self.head_layer.head, RegressionHead):
            output["y_hat"] = head_output["y_hat"]
            if self.scale_head_output:
                output["y_hat"] = self.rescale_output(output["y_hat"])

        elif isinstance(self.head_layer.head, NormalDistrHead):
            output["mu"] = head_output["mu"]
            output["sigma"] = head_output["sigma"]
            if self.scale_head_output:
                output["mu"] = self.rescale_output(output["mu"])
                output["sigma"] = self.rescale_output(output["sigma"])
            
        return {"param": param} | output

    def rescale_output(self, data):
        """ Rescale the output of the head layer. 
        This is useful when the head layer output (e.g. surrogate output) and the calibration target
        are in different ranges. For example, if the head layer output is soil volumetric water content 
        and the calibration target is degree of saturation from satellite estimates. 
        The rescale_output method applies a linear transformation to the output of the head layer
        to bring it to the range of the original data."""
        return data*self.scale + self.center 

    def rescale_input(self, param):
        return F.sigmoid(param)    