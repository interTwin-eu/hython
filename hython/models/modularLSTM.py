import torch
from torch import nn
import torch.nn.functional as F
from .base import BaseModel
from .head import get_head_layer
from .utils import make_mlp

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
            batch_first=True
        )

    def forward(self, x):
        l1 = self.fc0(x)

        lstm_output, (h_n, c_n) = self.lstm(l1)

        return lstm_output


class ModularLSTM(nn.Module):
    def __init__(self, cfg):
        super(ModularLSTM, self).__init__()
        """
        Parameters:
        cfg: Configuration object.
        
        Modular LSTM model with multiple LSTM modules.
        freeze_module: List of modules to freeze.
        output_size: Number of output features.
        dynamic_input_size: Number of dynamic input features.
        static_input_size: Number of static input features.
        modules: Dictionary of modules with their respective configurations.
        """

        self.freeze_module = cfg.freeze_module
        self.output_size = len(cfg.target_variables)
        self.dynamic_input_size = len(cfg.dynamic_inputs)
        self.static_input_size = len(cfg.static_inputs)
        self.modules = cfg.modules
        self.len_mod = len(self.modules)

        for k,v in self.modules.items():
            self.modules[k]["input_size"] = self.dynamic_input_size + self.static_input_size
           
        self.model_modules = nn.ModuleDict(
            {k: LSTMModule(**v) for k, v in self.modules.items()}
        )

        if self.freeze_module:
            for module in self.freeze_module:
                for weight in self.model_modules[module].parameters():
                    weight.requires_grad = False
                
            
        total_hidden_size = sum([v["hidden_size"] for v in self.modules.values()])

        self.interaction_layer = make_mlp(
            total_hidden_size, 
            int(total_hidden_size/3),
            int(total_hidden_size/self.len_mod), 
            3, 
            False, 
            "linear"
        )

        self.head = get_head_layer(cfg.model_head_layer, 
                                   input_dim= int(total_hidden_size/3), 
                                   output_dim= self.output_size, 
                                   head_activation= cfg.model_head_activation,
                                       **cfg.model_head_kwargs)


    def forward(self, x):
        distr_out = []
        for variable in self.model_modules:
            distr_out.append(
                self.model_modules[variable](x)
                )

        out = torch.cat(distr_out, dim=-1)

        # Interaction
        out = self.interaction_layer(out)

        out = self.head(out)

        return out