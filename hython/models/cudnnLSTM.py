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
        self.lstm_layers = cfg.lstm_layers
        self.hidden_size = cfg.hidden_size


        if cfg.lstm_batch_norm:
            self.bn_layer = nn.BatchNorm1d(
                self.dynamic_input_size + self.static_input_size
            )  # expects N C T

        self.dropout = nn.Dropout(p=cfg.dropout)

        self.fc0 = nn.Linear(self.dynamic_input_size + self.static_input_size, self.hidden_size)

        self.lstm = nn.LSTM(
            self.hidden_size,
            self.hidden_size,
            num_layers=self.lstm_layers,
            batch_first=True,
        )

        self.head = get_head_layer(cfg.model_head_layer, 
                                   input_dim= self.hidden_size, 
                                   output_dim= self.output_size, 
                                   head_activation= cfg.model_head_activation,
                                       **cfg.model_head_kwargs)

    def forward(self, x, h_0=None, c_0=None):
        if self.bn_flag:
            x = self.bn_layer(x.permute(0, 2, 1)).permute(0, 2, 1)

        l1 = self.fc0(x)

        if h_0 is None or c_0 is None:
            lstm_output, (h_n, c_n) = self.lstm(l1)
        else:
            lstm_output, (h_n, c_n) = self.lstm(l1, (h_0, c_0))

        lstm_output = self.dropout(lstm_output)

        head_out = self.head(lstm_output) # regression -> y_hat, normal --> mu, sigma
        
        pred = {"h_n": h_n, "c_n": c_n} | head_out 

        return pred



