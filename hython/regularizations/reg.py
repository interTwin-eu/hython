from typing import Optional, List, Dict
import torch
from torch import nn
from torch.nn.modules.loss import _Loss
import torch.nn.functional as F

__all__ = ["RegCollection", "ParamRuleReg", "RangeBoundReg"]

class RegCollection(nn.Module):
    def __init__(self, regs: List[nn.Module] = None):
        super(RegCollection, self).__init__()

        self.losses = []
        
        if not isinstance(regs, list) and regs is not None:
            regs = [regs]

        if regs is not None:
            self.regs = nn.ModuleDict({l.__name__: l for l in regs})
        else:
            self.regs = {}

    def __getitem__(self, k):
        if k in self.losses:
            return self.losses[k]
        else:
            return return_dict


def return_dict(*args):
    return {}


# class Reg1(nn.Module):
#     __name__ = "PrecipSoilMoisture"

#     def __init__(self):
#         super(Reg1, self).__init__()

#     def forward(self, x, y):
#         N, T, C = x.shape

#         # compute the x and y deltas, and remove the first element from the time vector due to torch.roll logic
#         diff_x = (x - x.roll(1, dims=1))[:, 1:]
#         diff_y = (y - y.roll(1, dims=1))[:, 1:]
#         # positive increments of the x field should produce positive increments of the y field
#         positive_x = diff_x >= 0
#         # positive
#         loss = torch.sum((F.relu(-1 * diff_y[positive_x])) ** 2) / torch.sum(positive_x)

#         return {self.__name__: loss}

RULES = {">=": torch.ge, "<=": torch.le, ">": torch.gt, "<": torch.lt, "==": torch.eq}

class ParamRuleReg(nn.Module):
    """
    # regularization:
    #   _target_: hython.regularizations.ParamConstraintReg
    #   factor: 1
    #   constraints:
    #     - ["thetaS", ">", "thetaR"]
    """
    def __init__(self, parameters: List, rules: List, data_source = "static_inputs", factor: int = 1):
        super(ParamRuleReg, self).__init__()
        self.params = list(parameters) 
        self.factor = factor
        self.rules = rules
        self.data_source = data_source

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: Parameters
        """
        loss = 0
        for c in self.rules:
            pname1 = c[0]
            pname2 = c[2]
            pidx1 = self.params.index(pname1)
            pidx2 = self.params.index(pname2)
            op = RULES[c[1]]
            # If violated should return 1 per example
            violated_bool = torch.logical_not(op(x[pidx1], x[pidx2])) 
            violated_sum = violated_bool.sum()
            loss += (violated_sum * self.factor)
        return loss



class RangeBoundReg(nn.Module):
    def __init__(self, bounds: Dict, factor: int = 1) -> None:
        super(RangeBoundReg, self).__init__()
        self.factor = factor
        lbs = []
        ubs = []
        for k,v in bounds.items():
            lbs.append(v[0]) # min
            ubs.append(v[1]) # max
        self.lbs = torch.tensor(lbs)
        self.ubs = torch.tensor(ubs)


    def forward(self, x: torch.Tensor) -> torch.Tensor:

        loss = 0
        for i in range(x.size(1)):
            lb = self.lbs[i]
            ub = self.ubs[i]
            upper_bound_loss = torch.relu(x[i] - ub)
            lower_bound_loss = torch.relu(lb - x[i])
            mean_loss = self.factor * (upper_bound_loss + lower_bound_loss).mean() / 2.0
            loss = loss + mean_loss
        return loss