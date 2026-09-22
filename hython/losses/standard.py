from typing import Optional, List
import torch
from torch import nn
from torch.nn.modules.loss import _Loss
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf
from torch.distributions import Normal
from functorch import vmap

class BaseLoss(torch.nn.Module):
    def __init__(self, cfg={}):
        self.cfg = (
            OmegaConf.create(cfg) if isinstance(cfg, dict) else OmegaConf.load(cfg)
        )

def compute_spaef(observed: torch.Tensor, simulated: torch.Tensor) -> torch.Tensor:
    """
    Compute the Spatial Efficiency (SPAEF) between an observed and simulated feature set.
    :param observed: 1D torch tensor (true feature vector)
    :param simulated: 1D torch tensor (simulated feature vector)
    :return: SPAEF score (tensor)
    """
    r = torch.corrcoef(torch.stack([observed, simulated]))[0, 1]
    alpha = torch.std(simulated) / torch.std(observed)
    
    hist_obs = torch.histc(observed, bins=100, min=0, max=1) #min=observed.min(), max=observed.max())
    hist_sim = torch.histc(simulated, bins=100, min=0, max=1) #min=simulated.min(), max=simulated.max())
    hist_obs /= hist_obs.sum()
    hist_sim /= hist_sim.sum()
    beta = torch.min(hist_obs, hist_sim).sum()
    
    return 1 - torch.sqrt((r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2)

def spaef_temporal(obs_series: torch.Tensor, sim_series: torch.Tensor, method: str = 'mean') -> torch.Tensor:
    """
    Compute the aggregated SPAEF score over a time series of feature vectors.
    :param obs_series: 3D torch tensor (B, T, F) - observed feature vectors over time
    :param sim_series: 3D torch tensor (B, T, F) - simulated feature vectors over time
    :param method: Aggregation method ('mean', 'median', or 'percentile_X')
    :return: Aggregated SPAEF score (tensor)
    """
    
    spaef_fn = vmap(vmap(compute_spaef, in_dims=(1, 1)), in_dims=(0, 0))  # Apply across time and batch
    spaef_values = spaef_fn(obs_series, sim_series)
    
    if method == 'mean':
        return torch.mean(spaef_values) # dim=1 # Aggregate over time for each batch
    elif method == 'median':
        return torch.median(spaef_values, dim=1).values
    elif method.startswith('percentile_'):
        percentile = float(method.split('_')[1])
        return torch.quantile(spaef_values, percentile / 100, dim=1)
    else:
        raise ValueError("Invalid aggregation method. Choose 'mean', 'median', or 'percentile_X'")

def compute_kge_torch(true, pred, eps=1e-8):
    # Remove invalid (NaN) entries
    mask = (~torch.isnan(true)) & (~torch.isnan(pred))
    true = true[mask]
    pred = pred[mask]

    # Pearson correlation
    true_mean = torch.mean(true)
    pred_mean = torch.mean(pred)
    r_num = torch.sum((true - true_mean) * (pred - pred_mean))
    r_den = torch.sqrt(torch.sum((true - true_mean) ** 2) * torch.sum((pred - pred_mean) ** 2) + eps)
    r = r_num / r_den

    # Standard deviation ratio (α)
    alpha = torch.std(pred, unbiased=True) / (torch.std(true, unbiased=True) + eps)

    # Mean ratio (β)
    beta = pred_mean / (true_mean + eps)

    # Kling-Gupta Efficiency
    kge = 1 - torch.sqrt((r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2)

    return kge

def compute_kge_per_cell(true, pred, eps=1e-8, use_beta=True, weights=None):
    """
    Compute the KGE of each cell over its own time series.

    :param true: torch tensor (N, T). NaN marks a missing observation.
    :param pred: torch tensor (N, T).
    :param use_beta: if False, the KGE has no mean-ratio term:
        1 - sqrt((r-1)^2 + (alpha-1)^2).
    :param weights: optional (w_r, w_alpha, w_beta), the scaled KGE
        1 - sqrt((w_r(r-1))^2 + (w_alpha(alpha-1))^2 + (w_beta(beta-1))^2).
        None means (1, 1, 1).
    :return: tuple (kge, count). kge is (M,) and holds only the cells with
        2 or more observations, because the standard deviation needs 2.
        count is (M,), the number of observations of each of these cells.
    """
    mask = (~torch.isnan(true)) & (~torch.isnan(pred))
    keep = mask.sum(-1) >= 2
    mask = mask[keep].float()
    # Set the missing values to 0 before the arithmetic. A NaN that is only
    # multiplied by 0 still makes the gradient NaN.
    true = torch.where(mask.bool(), true[keep], 0.0)
    pred = torch.where(mask.bool(), pred[keep], 0.0)

    n = mask.sum(-1)
    true_mean = (true * mask).sum(-1) / n
    pred_mean = (pred * mask).sum(-1) / n
    dtrue = (true - true_mean[:, None]) * mask
    dpred = (pred - pred_mean[:, None]) * mask
    ss_true = (dtrue**2).sum(-1)
    ss_pred = (dpred**2).sum(-1)

    # Pearson correlation
    r = (dtrue * dpred).sum(-1) / torch.sqrt(ss_true * ss_pred + eps)

    # Standard deviation ratio (α). eps in the sqrt keeps the gradient finite
    # when a cell's prediction is constant.
    alpha = torch.sqrt(ss_pred / (n - 1) + eps) / (torch.sqrt(ss_true / (n - 1)) + eps)

    # Mean ratio (β)
    beta = pred_mean / (true_mean + eps)

    w_r, w_a, w_b = weights if weights is not None else (1.0, 1.0, 1.0)
    terms = (w_r * (r - 1)) ** 2 + (w_a * (alpha - 1)) ** 2
    if use_beta:
        terms = terms + (w_b * (beta - 1)) ** 2
    kge = 1 - torch.sqrt(terms)

    return kge, n

class CellKGELoss(_Loss):
    """
    Negative KGE, computed per cell and then averaged over the cells.

    Each cell's KGE is weighted by its number of observations: a cell with
    few observations represents its time series less well, so it has less
    effect on the loss. Cells with fewer than 2 observations get no weight.
    ``use_beta: false`` drops the mean-ratio term (H12). ``weights``
    (w_r, w_alpha, w_beta) scales the three terms; None means (1, 1, 1).

    The trainer gives this loss the (N, T) tensors, not the flattened valid
    values (see ``per_cell``).
    """

    # Tells the trainer to keep the (N, T) shape and set the invalid values to NaN
    per_cell = True

    def __init__(self, use_beta: bool = True, weights=None):
        super(CellKGELoss, self).__init__()
        self.use_beta = use_beta
        self.weights = None if weights is None else tuple(float(w) for w in weights)

    def forward(self, target, y_pred):
        kge, n = compute_kge_per_cell(target, y_pred, use_beta=self.use_beta, weights=self.weights)
        if kge.numel() == 0:
            # No cell with enough observations: a zero loss that keeps the graph
            return y_pred.nan_to_num().sum() * 0.0
        return -1 * (kge * n).sum() / n.sum()

class SPAEFLoss(_Loss):
    def __init__(self, method: str = 'mean'):
        super(SPAEFLoss, self).__init__()
        self.method = method
    def forward(self, target, y_pred):
        return spaef_temporal(target, y_pred, method=self.method)

class KGELoss(_Loss):
    def __init__(self):

        super(KGELoss, self).__init__()

    def forward(self, target, y_pred):
        #  kge best is 1, reverse the sign to make it a loss
        return -1*compute_kge_torch(target, y_pred)

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
    

class PearsonLoss(_Loss):
    def __init__(self):
        super(PearsonLoss, self).__init__()


    def forward(self, target, y_pred):  
        
        target_centered = target - target.mean() 
        pred_centered =  y_pred - y_pred.mean()

        num = torch.sum(target_centered*pred_centered)
        den = torch.sqrt( torch.sum(target_centered**2)*torch.sum(pred_centered**2) )
        
        loss = num/den

        return -loss
    

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





