import torch
from torch import nn

class RMSELoss(nn.Module):
    def __init__(self):
        super(RMSELoss, self).__init__()
        
        self.mseloss = nn.MSELoss()

    def forward(self, y_true, y_pred):
        """
        Calculate the Root Mean Squared Error (RMSE) between two tensors.

        Parameters:
        y_true (torch.Tensor): The true values.
        y_pred (torch.Tensor): The predicted values.

        Returns:
        torch.Tensor: The RMSE loss.
        """
        rmse_loss = torch.sqrt(self.mseloss(y_true, y_pred))

        return rmse_loss