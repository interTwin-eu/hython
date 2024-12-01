import torch
from torch import nn
import torch.nn.functional as F
from . import BaseModel

class CuDNNLSTM(BaseModel):
    def __init__(
        self,
        hidden_size: int = 34,
        dynamic_input_size: int = 3,
        static_input_size: int = 5,
        output_size: int = 2,
        static_to_dynamic: bool = True,
        num_layers: int = 1,
        dropout: float = 0.0,
        batch_norm: bool = False,
        output_activation: str = "linear"
    ):
        super(CuDNNLSTM, self).__init__()

        self.static_to_dynamic = static_to_dynamic
        self.output_activation = output_activation

        self.bn_flag = batch_norm
        if batch_norm: 
            self.bn_layer = nn.BatchNorm1d(dynamic_input_size + static_input_size) # expects N C T

        self.dropout = nn.Dropout(p=dropout)

        self.fc0 = nn.Linear(dynamic_input_size + static_input_size, hidden_size)

        self.lstm = nn.LSTM(
            hidden_size,
            hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )

        self.fc1 = nn.Linear(hidden_size, output_size)

    def forward(self, x):

        if self.bn_flag:
            x  = self.bn_layer(x.permute(0,2,1)).permute(0,2,1)

        l1 = self.fc0(x)

        lstm_output, (h_n, c_n) = self.lstm(l1)

        lstm_output = self.dropout(lstm_output)
        
        if self.output_activation == "linear":
            out = self.fc1(lstm_output)
        else:
            out = getattr(F, self.output_activation)(self.fc1(lstm_output))

        pred = {"y_hat":out, "h_n":h_n, "c_n":c_n}
            
        return pred


class LSTMModule(nn.Module):
    def __init__(
        self,
        hidden_size: int = 34,
        input_size: int = 3,
        num_layers: int = 1,
        dropout: float = 0.0,
    ):
        super(LSTMModule, self).__init__()

        self.fc0 = nn.Linear(input_size, hidden_size)

        self.lstm = nn.LSTM(
            hidden_size,
            hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
        )

    def forward(self, x):
        l1 = self.fc0(x)

        lstm_output, (h_n, c_n) = self.lstm(l1)

        return lstm_output


class LandSurfaceLSTM(nn.Module):
    def __init__(self, module_dict, output_size, device=None):
        super(LandSurfaceLSTM, self).__init__()

        self.model_modules = nn.ModuleDict(
            {k: LSTMModule(**v) for k, v in module_dict.items()}
        )
        if device:
            self.model_modules = self.model_modules.to(device)
        total_hidden_size = sum([v["hidden_size"] for v in module_dict.values()])

        self.fc0 = nn.Linear(total_hidden_size, output_size)

        self.relu = nn.ReLU()

    def forward(self, x):
        distr_out = []
        for variable in self.model_modules:
            distr_out.append(self.model_modules[variable](x))

        out = torch.cat(distr_out, dim=-1)

        # Interaction
        out = self.relu(self.fc0(out))

        return out
