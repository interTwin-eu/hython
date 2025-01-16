from typing import Optional, List
import torch
from torch import nn
from torch.nn.modules.loss import _Loss
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf


class BaseLoss(torch.nn.Module):
    def __init__(self, cfg={}):
        self.cfg = (
            OmegaConf.create(cfg) if isinstance(cfg, dict) else OmegaConf.load(cfg)
        )


class RMSELoss(_Loss):
    def __init__(self):
        """
        Root Mean Squared Error (RMSE) loss for regression task.

         Parameters:
         target_weight: List of targets that contribute in the loss computation, with their associated weights.
                        In the form {target: weight}
        """

        super(RMSELoss, self).__init__()
        self.mseloss = nn.MSELoss()

    def forward(self, y_true, y_pred):
        """
        Calculate the Root Mean Squared Error (RMSE) between two tensors.

        Parameters:
        y_true (torch.Tensor): The true values.
        y_pred (torch.Tensor): The predicted values.
        valid_mask: A boolean mask to pre-filter the y_true and y_pred before they are used in the loss function.

        Shape
        y_true: torch.Tensor of shape (N, C).
        y_pred: torch.Tensor of shape (N, C).
        valid_mask:

        Returns:
        torch.Tensor: The RMSE loss.
        """
        return torch.sqrt(self.mseloss(y_true, y_pred))


class MSELoss(_Loss):
    def __init__(self):
        """
        Mean Squared Error (MSE) loss for regression task.

         Parameters:
         target_weight: List of targets that contribute in the loss computation, with their associated weights.
                        In the form {target: weight}
        """

        super(MSELoss, self).__init__()
        self.mseloss = nn.MSELoss()

    def forward(self, y_true, y_pred):
        return self.mseloss(y_true, y_pred)
