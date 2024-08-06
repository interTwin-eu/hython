import torch
from torch import nn


class CuDNNLSTM(nn.Module):
    def __init__(
        self,
        hidden_size: int = 34,
        dynamic_input_size: int = 3,
        static_input_size: int = 5,
        output_size: int = 2,
        static_to_dynamic: bool = True,
        num_layers:int = 1,
        dropout:float = 0.0,
    ):
        super(CuDNNLSTM, self).__init__()

        self.static_to_dynamic = static_to_dynamic

        self.fc0 = nn.Linear(dynamic_input_size + static_input_size, hidden_size)

        self.lstm = nn.LSTM(hidden_size, hidden_size, num_layers = num_layers, batch_first=True, dropout=dropout)

        self.fc1 = nn.Linear(hidden_size, output_size)

    def forward(self, x):

        l1 = self.fc0(x)

        lstm_output, (h_n, c_n) = self.lstm(l1)
    
        out = self.fc1(lstm_output)

        return out
