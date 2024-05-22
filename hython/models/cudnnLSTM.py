import torch
from torch import nn


class CuDNNLSTM(nn.Module):
    def __init__(self, input_size = 3, hidden_size = 34, output_size = 2, number_static_predictors = 5, dropout = 0.1):
        
        super(CuDNNLSTM, self).__init__()

        self.fc0 = nn.Linear(input_size + number_static_predictors, hidden_size)

        self.lstm = nn.LSTM(hidden_size , hidden_size, batch_first=True)

        self.fc1 = nn.Linear(hidden_size, output_size)
        
    def forward(self, x, static_params):

        s = static_params.unsqueeze(1).repeat(1, x.size(1), 1)
        
        x_ds = torch.cat(
             (x,
              s),
              dim=-1,
         )
        
        l1 = self.fc0(x_ds)

        lstm_output, (h_n, c_n) = self.lstm(l1)

        out =  self.fc1(lstm_output)

        return out