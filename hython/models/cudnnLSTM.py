import torch
from torch import nn


class CuDNNLSTM(nn.Module):
    def __init__(
        self,
        hidden_size: int = 34,
        dynamic_input_size: int = 3,
        static_input_size: int = 5,
        output_size: int = 2,
        dropout=0.1,
    ):
        super(CuDNNLSTM, self).__init__()

        self.fc0 = nn.Linear(dynamic_input_size + static_input_size, hidden_size)

        self.lstm = nn.LSTM(hidden_size, hidden_size, batch_first=True)

        self.fc1 = nn.Linear(hidden_size, output_size)

    def forward(self, x, static_params):
        s = static_params.unsqueeze(1).repeat(1, x.size(1), 1)

        x_ds = torch.cat(
            (x, s),
            dim=-1,
        )

        l1 = self.fc0(x_ds)

        lstm_output, (h_n, c_n) = self.lstm(l1)

        out = self.fc1(lstm_output)

        return out


class LSTMCell(nn.Module):
    def __init__(
        self,
        hidden_size: int = 34,
        dynamic_input_size: int = 3,
        static_input_size: int = 5,
        output_size: int = 1,
        dropout=0.1,
    ):
        super(LSTMCell, self).__init__()

        self.fc0 = nn.Linear(dynamic_input_size + static_input_size, hidden_size)

        self.lstm = nn.LSTM(hidden_size, hidden_size, batch_first=True)

        self.fc1 = nn.Linear(hidden_size, output_size)

    def forward(self, x_ds):


        l1 = self.fc0(x_ds)

        lstm_output, (h_n, c_n) = self.lstm(l1)

        #out = self.fc1(lstm_output)

        return lstm_output
    


class LandSurfaceModel(nn.Module):

    def __init__(self, module_dict, hidden_size, output_size):
        super(LandSurfaceModel, self).__init__()
        
        self.module_dict = module_dict

        self.fc0 = nn.Linear(hidden_size, output_size)

        self.relu = nn.ReLU()

    def forward(self, x, static_params):

        s = static_params.unsqueeze(1).repeat(1, x.size(1), 1)

        x_ds = torch.cat(
            (x, s),
            dim=-1,
        )
        #import pdb;pdb.set_trace()
        # distributed
        distr_out = [] 
        for variable in self.module_dict:
            distr_out.append(
                self.module_dict[variable](x_ds)
            )
            

        out = torch.cat(distr_out, dim=-1)
        
        out = self.fc0(out)
        
        return out
    

class GIUH(nn.Module):
    def __init__(self, 
                 hidden_size,
                 dynamic_input_size,
                 static_input_size, 
                 output_size
                 ):
        super(GIUH, self).__init__()   

        self.lstm_cell = LSTMCell(
            hidden_size = hidden_size,
            dynamic_input_size = dynamic_input_size, # forcings + state_and_flux
            static_input_size=static_input_size,
            output_size=output_size
        )

    def forward(self, forcing, state_and_flux, static):
        # Find a mapping between each hru (gridcell) 
        # of the basin and the outlet
        # use forcings + states and fluxes + static inputs
        
        input = forcing + state_and_flux + static # N L C[n]
        output = self.lstm_cell(input) # N L C[n] => N L C[discharge]

        return output
