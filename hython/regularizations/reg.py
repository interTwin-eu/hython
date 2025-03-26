from typing import Optional, List, Dict
import torch
from torch import nn
from torch.nn.modules.loss import _Loss
import torch.nn.functional as F


class RegCollection(nn.Module):
    def __init__(self, loss: List[nn.Module] = None):
        super(RegCollection, self).__init__()

        if not isinstance(loss, list) and loss is not None:
            loss = [loss]

        if loss is not None:
            self.losses = nn.ModuleDict({l.__name__: l for l in loss})
        else:
            self.losses = {}

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

class ParamConstraintReg(nn.Module):
    def __init__(self, constraints: List, factor: int = 1):
        super(ParamConstraintReg, self).__init__()
        self.fact0r = factor
        self.constraints = constraints

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: Parameters
        """
        for i in range(x.size(1)):
            for c in self.constraints:
                op = RULES[c[1]]
                loss = torch.relu(op(x[c[0]], x[c[0]])).mean()
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