import torch
from torch import nn
import torch.nn.functional as F
from .base import BaseModel
from .head import get_head_layer


class CudaLSTM(BaseModel):
    def __init__(
        self,
        cfg):
        super(CudaLSTM, self).__init__(cfg=cfg)

        self.output_size = len(cfg.target_variables)
        self.dynamic_input_size = len(cfg.dynamic_inputs)
        self.static_input_size = len(cfg.static_inputs)
        self.bn_flag = cfg.lstm_batch_norm

        if cfg.lstm_batch_norm:
            self.bn_layer = nn.BatchNorm1d(
                self.dynamic_input_size + self.static_input_size
            )  # expects N C T

        self.dropout = nn.Dropout(p=cfg.dropout)

        self.fc0 = nn.Linear(self.dynamic_input_size + self.static_input_size, cfg.hidden_size)

        self.lstm = nn.LSTM(
            cfg.hidden_size,
            cfg.hidden_size,
            num_layers=cfg.lstm_layers,
            batch_first=True,
        )

        self.head = get_head_layer(cfg.model_head_layer, 
                                   input_dim= cfg.hidden_size, 
                                   output_dim= self.output_size, 
                                   head_activation= cfg.model_head_activation,
                                       **cfg.model_head_kwargs)


    def forward(self, x):
        if self.bn_flag:
            x = self.bn_layer(x.permute(0, 2, 1)).permute(0, 2, 1)

        l1 = self.fc0(x)

        lstm_output, (h_n, c_n) = self.lstm(l1)

        lstm_output = self.dropout(lstm_output)

        head_out = self.head(lstm_output) # regression -> y_hat, normal --> mu, sigma
        
        pred = {"h_n": h_n, "c_n": c_n} | head_out 

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
