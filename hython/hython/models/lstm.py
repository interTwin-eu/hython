import torch
from torch import nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset



class CustomLSTMModel(nn.Module):
    def __init__(self, model_params):
        
        input_size  = model_params["input_size"]
        hidden_size = model_params["hidden_size"]
        output_size = model_params["output_size"]
        number_static_predictors = model_params["number_static_predictors"]
        
        super(CustomLSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        
        self.fc1 = nn.Linear(hidden_size + number_static_predictors, 32)  # Concatenating with static parameters
        
        self.fc2 = nn.Linear(32, output_size)

    def forward(self, x, static_params):
        lstm_output, _ = self.lstm(x)
        
        # Concatenate LSTM output with static parameters
        combined_output = torch.cat((lstm_output, static_params.unsqueeze(1).repeat(1, lstm_output.size(1), 1)), dim=-1)
        
        out = torch.relu(self.fc1(combined_output))
        
        out = self.fc2(out)
        return out
    
    
    