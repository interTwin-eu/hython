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

    def forward(self, target, y_pred):
        """
        Calculate the Root Mean Squared Error (RMSE) between two tensors.

        Parameters:
        target (torch.Tensor): The true values.
        y_pred (torch.Tensor): The predicted values.
        valid_mask: A boolean mask to pre-filter the target and pred before they are used in the loss function.

        Shape
        target: torch.Tensor of shape (N, C).
        y_pred: torch.Tensor of shape (N, C).
        valid_mask:

        Returns:
        torch.Tensor: The RMSE loss.
        """
        return torch.sqrt(self.mseloss(target, y_pred))


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

    def forward(self, target, y_pred):
        return self.mseloss(target, y_pred)

class MSEWeightedModeLoss(_Loss):
    def __init__(self, threshold=0.5, high_weight=1.0, low_weight=0.5):
        """
        Mean Squared Error (MSE) loss for bimodal distribution, where one mode is more important than the other.

         Parameters:
         threshold: The threshold value that separates the two modes.
         high_weight: The weight of the high mode.
         low_weight: The weight of the low mode.
        """
        super(MSEWeightedModeLoss, self).__init__()
        self.threshold = threshold
        self.high_weight = high_weight
        self.low_weight = low_weight

    def forward(self, target, y_pred):  

        weights = torch.where(target > self.threshold, self.high_weight, self.low_weight)
        weights = weights / torch.mean(weights)
        loss = torch.mean(weights * (y_pred - target) ** 2)
        return loss
    
class RMSEWeightedModeLoss(_Loss):
    def __init__(self, threshold=0.5, high_weight=1.0, low_weight=0.5):
        """
        Root Mean Squared Error (MSE) loss for bimodal distribution, where one mode is more important than the other.

         Parameters:
         threshold: The threshold value that separates the two modes.
         high_weight: The weight of the high mode.
         low_weight: The weight of the low mode.
        """
        super(RMSEWeightedModeLoss, self).__init__()
        self.threshold = threshold
        self.high_weight = high_weight
        self.low_weight = low_weight

    def forward(self, target, y_pred):  

        weights = torch.where(target > self.threshold, self.high_weight, self.low_weight)
        weights = weights / torch.mean(weights)
        loss = torch.sqrt(torch.mean(weights * (y_pred - target) ** 2))
        return loss
    
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

    def forward(self, target, mu, sigma):
        """
        Calculate the negative log-likelihood of the underlying normal distribution.
        Parameters:
        target (torch.Tensor): The true values.
        distr_mean (torch.Tensor): The predicted mean values. 
        distr_std (torch.Tensor): The predicted std values.
        Shape
        target: torch.Tensor of shape (N, T).
        distr_mean: torch.Tensor of shape (N, T).
        distr_std: torch.Tensor of shape (N, T).
        (256,3) means 256 samples with 3 targets.
        Returns:
        torch.Tensor: The NLL loss.
        """
        dist = Normal(mu, sigma)
        total_nll_loss = -dist.log_prob(target).mean()
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

    def forward(self, target, y_pred):
        """
        Calculate the Pinball Loss between two tensors.

        Parameters:
        target (torch.Tensor): The true values.
        y_pred (torch.Tensor): The predicted values.

        Shape:
        target: torch.Tensor of shape (N, T).
        y_pred: torch.Tensor of shape (N, T).
        (256, 3) means 256 samples with 3 targets.

        Returns:
        torch.Tensor: The Pinball loss.
        """
        error = target - y_pred
        pinball_loss = torch.maximum(self.tau * error, (self.tau - 1) * error)
        return pinball_loss.mean()



# class SPAEFLoss(_Loss):
#     def __init__(self, return_all=False):
#         super(SPAEFLoss, self).__init__()
#         self.return_all = return_all

#     def forward(self, target, y_pred):

#         return spaeff_torch(y_pred, target, self.return_all)


