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

    def forward(self, y_true, mu, sigma):
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
        dist = Normal(mu, sigma)
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



class SPAEFLoss(_Loss):
    def __init__(self, return_all=False):
        super(SPAEFLoss, self).__init__()
        self.return_all = return_all

    def forward(self, y_true, y_pred):

        return spaeff_torch(y_pred, y_true, self.return_all)


def spaeff_torch(sim, obs, return_all = False):
    """
    Compute the SPAEF metric using PyTorch.

    Parameters:
    - sim: torch.Tensor, simulated  (1D or flattened 2D)
    - obs: torch.Tensor, observed (1D or flattened 2D)

    Returns:
    - spaef: float, SPAEF score
    - alpha: float, correlation coefficient
    - beta: float, coefficient of variation ratio
    - gamma: float, histogram intersection
    """
    # Remove NaNs
    mask = ~torch.isnan(sim) & ~torch.isnan(obs)
    sim, obs = sim[mask], obs[mask]

    # Compute correlation coefficient (alpha)
    alpha = torch.corrcoef(torch.stack((sim, obs)))[0, 1]

    # Compute coefficient of variation ratio (beta)
    beta = (torch.std(sim) / torch.mean(sim)) / (torch.std(obs) / torch.mean(obs))

    # Compute histogram intersection (gamma)
    bins = int(torch.sqrt(torch.tensor(len(obs), dtype=torch.float32)))
    hist_sim = torch.histc(sim, bins=bins, min=sim.min().item(), max=sim.max().item())
    hist_obs = torch.histc(obs, bins=bins, min=obs.min().item(), max=obs.max().item())

    gamma = torch.sum(torch.min(hist_sim, hist_obs)) / torch.sum(hist_obs)

    # Compute SPAEF
    spaef = 1 - torch.sqrt((alpha - 1) ** 2 + (beta - 1) ** 2 + (gamma - 1) ** 2)
    if return_all:
        return spaef, alpha, beta, gamma
    else:
        return spaef
