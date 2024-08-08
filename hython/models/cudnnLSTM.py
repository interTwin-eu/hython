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



class LSTMModule(nn.Module):
    def __init__(
        self,
        hidden_size: int = 34,
        input_size: int = 3,
        num_layers:int = 1,
        dropout:float = 0.0,
    ):
        super(LSTMModule, self).__init__()

        self.fc0 = nn.Linear(input_size, hidden_size)

        self.lstm = nn.LSTM(hidden_size, hidden_size, num_layers = num_layers, batch_first=True, dropout=dropout)

    def forward(self, x):

        l1 = self.fc0(x)

        lstm_output, (h_n, c_n) = self.lstm(l1)

        return lstm_output
    

class LandSurfaceLSTM(nn.Module):

    def __init__(self, 
                 module_dict,
                 output_size,
                 device):
        super(LandSurfaceLSTM, self).__init__()
        
        self.modules = {k:LSTMModule(**v).to(device) for k,v in module_dict.items()}

        total_hidden_size = sum([v["hidden_size"] for v in module_dict.values()])

        self.fc0 = nn.Linear(total_hidden_size, output_size)

        self.relu = nn.ReLU()

    def forward(self, x):

        distr_out = [] 
        for variable in self.modules:
            distr_out.append(
                self.modules[variable](x)
            )
            
        out = torch.cat(distr_out, dim=-1)

        # Interaction
        out = self.relu(self.fc0(out))
        
        return out