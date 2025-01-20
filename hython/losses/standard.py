from typing import Optional, List
import torch
from torch import nn
from torch.nn.modules.loss import _Loss
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf
from torch.distributions import Normal

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


class NegLLLoss(_Loss):
    def __init__(
        self
    ):
        """
        Negative log-likelihood (NLL) loss for normal distribution.
         Parameters:
         target_weight: List of targets that contribute in the loss computation, with their associated weights.
                        In the form {target: weight}
        """

        super(NegLLLoss, self).__init__()

    def forward(self, y_true, distr_mean, distr_std):
        """
        Calculate the negative log-likelihood of the underlying normal distribution.
        Parameters:
        y_true (torch.Tensor): The true values.
        distr_mean (torch.Tensor): The predicted mean values. 
        distr_std (torch.Tensor): The predicted std values.
        Shape
        y_true: torch.Tensor of shape (N, T).
        distr_mean: torch.Tensor of shape (N, T).
        distr_std: torch.Tensor of shape (N, T).
        (256,3) means 256 samples with 3 targets.
        Returns:
        torch.Tensor: The NLL loss.
        """
        dist = Normal(distr_mean, distr_std)
        total_nll_loss = -dist.log_prob(y_true).mean()
        return total_nll_loss
    

class PinballLoss(nn.Module):
    def __init__(
        self,
        tau: float
    ):
        """
        Pinball Loss for regression tasks.

        Parameters:
        tau: Quantile level (0 < tau < 1).
        target_weight: Dictionary of targets with associated weights.
                    In the form {target: weight}.
        """
        super(PinballLoss, self).__init__()
        if not 0 < tau < 1:
            raise ValueError("tau must be between 0 and 1")
        self.tau = tau

    def forward(self, y_true, y_pred):
        """
        Calculate the Pinball Loss between two tensors.

        Parameters:
        y_true (torch.Tensor): The true values.
        y_pred (torch.Tensor): The predicted values.

        Shape:
        y_true: torch.Tensor of shape (N, T).
        y_pred: torch.Tensor of shape (N, T).
        (256, 3) means 256 samples with 3 targets.

        Returns:
        torch.Tensor: The Pinball loss.
        """
        error = y_true - y_pred
        pinball_loss = torch.maximum(self.tau * error, (self.tau - 1) * error)
        return pinball_loss.mean()

