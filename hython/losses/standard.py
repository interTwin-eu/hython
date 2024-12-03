from typing import Optional, List
import torch
from torch import nn
from torch.nn.modules.loss import _Loss
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf


class BaseLoss(torch.nn.Module):

    def __init__(self, cfg = {},):
        self.cfg = OmegaConf.create(cfg) if isinstance(cfg, dict) else OmegaConf.load(cfg)


class RMSELoss(_Loss):
    __name__ = "RMSE"

    def __init__(self):
        """
        Root Mean Squared Error (RMSE) loss for regression task.

         Parameters:
         target_weight: List of targets that contribute in the loss computation, with their associated weights.
                        In the form {target: weight}
        """

        super(RMSELoss, self).__init__()
        self.mseloss = nn.MSELoss()

    def forward(
        self,
        y_true,
        y_pred,
        valid_mask=None,
        scaling_factor=None,
        target_weight: dict = None,
    ):
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
        if target_weight is None:
            total_rmse_loss = torch.sqrt(self.mseloss(y_true, y_pred))
        else:
            if len(target_weight.keys()) > 1:
                total_rmse_loss = 0
                for itarget, target in enumerate(target_weight):
                    iypred = y_pred[:, itarget]
                    iytrue = y_true[:, itarget]
                    if valid_mask is not None:
                        imask = valid_mask[:, itarget]
                        iypred = iypred[imask]
                        iytrue = iytrue[imask]
                    w = target_weight[target]
                    rmse_loss = torch.sqrt(self.mseloss(iytrue, iypred))
                    loss = rmse_loss * w
                    total_rmse_loss += loss
            else:  # case when only one target is available
                if len(y_pred.shape) > 1:
                    iypred = y_pred[:, 0]
                    iytrue = y_true[:, 0]
                else:
                    iypred = y_pred
                    iytrue = y_true
                if valid_mask is not None:
                    imask = valid_mask[:, 0]
                    iypred = iypred[imask]
                    iytrue = iytrue[imask]
                total_rmse_loss = torch.sqrt(self.mseloss(iytrue, iypred))

        if scaling_factor:
            total_rmse_loss = total_rmse_loss * scaling_factor

        return total_rmse_loss


class MSELoss(_Loss):
    __name__ = "MSE"

    def __init__(self):
        """
        Mean Squared Error (MSE) loss for regression task.

         Parameters:
         target_weight: List of targets that contribute in the loss computation, with their associated weights.
                        In the form {target: weight}
        """

        super(MSELoss, self).__init__()
        self.mseloss = nn.MSELoss()

    def forward(
        self,
        y_true,
        y_pred,
        valid_mask=None,
        scaling_factor=None,
        target_weight: dict = None,
    ):
        """
        Calculate the Mean Squared Error (RMSE) between two tensors.

        Parameters:
        y_true (torch.Tensor): The true values.
        y_pred (torch.Tensor): The predicted values.

        Shape
        y_true: torch.Tensor of shape (N, T).
        y_pred: torch.Tensor of shape (N, T).
        (256,3) means 256 samples with 3 targets.

        Returns:
        torch.Tensor: The RMSE loss.
        """
        if target_weight is None:
            loss = self.mseloss(y_true, y_pred)
        else:
            if len(target_weight.keys()) > 1:
                total_loss = 0
                for itarget, target in enumerate(target_weight):
                    iypred = y_pred[:, itarget]
                    iytrue = y_true[:, itarget]
                    if valid_mask is not None:
                        imask = valid_mask[:, itarget]
                        iypred = iypred[imask]
                        iytrue = iytrue[imask]
                    w = target_weight[target]
                    loss = torch.sqrt(self.mseloss(iytrue, iypred))
                    loss = loss * w
                    total_loss += loss
            else:  # case when only one target is available
                if len(y_pred.shape) > 1:
                    iypred = y_pred[:, 0]
                    iytrue = y_true[:, 0]
                else:
                    iypred = y_pred
                    iytrue = y_true
                if valid_mask is not None:
                    imask = valid_mask[:, 0]
                    iypred = iypred[imask]
                    iytrue = iytrue[imask]
                total_loss = self.mseloss(iytrue, iypred)

        if scaling_factor:
            total_loss = total_loss * scaling_factor

        return total_loss
