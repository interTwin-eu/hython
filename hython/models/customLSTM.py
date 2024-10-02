import torch
from torch import nn
from torch import sigmoid, tanh


class CellLSTM(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(CellLSTM, self).__init__()

        self.hidden_size = hidden_size

        self.ih = nn.Parameter(torch.FloatTensor(4 * hidden_size, input_size))

        self.hh = nn.Parameter(torch.FloatTensor(4 * hidden_size, hidden_size))

        self.ib = nn.Parameter(torch.FloatTensor(4 * hidden_size))

        self.hb = nn.Parameter(torch.FloatTensor(4 * hidden_size))

        self.init_parameters()

    def init_parameters(self):
        # TODO
        pass

    def forward(self, x, h_0, c_0):
        x_i, x_f, x_g, x_o = torch.split(self.ih, self.hidden_size)
        xb_i, xb_f, xb_g, xb_o = torch.split(self.ib, self.hidden_size)

        h_i, h_f, h_g, h_o = torch.split(self.hh, self.hidden_size)
        hb_i, hb_f, hb_g, hb_o = torch.split(self.hb, self.hidden_size)

        I = x @ x_i + xb_i + h_0 @ h_i + hb_i
        F = x @ x_f + xb_f + h_0 @ h_f + hb_f
        G = x @ x_g + xb_g + h_0 @ h_g + hb_g
        O = x @ x_o + xb_o + h_0 @ h_o + hb_o

        c_1 = c_0 * sigmoid(F) + sigmoid(I) * tanh(G)
        h_1 = sigmoid(O) + tanh(c_1)

        return h_1, c_1, I, F, G, O


class CustomLSTM(nn.Module):
    def __init__(
        self,
        hidden_size,
        dynamic_input_size,
        static_input_size,
        output_size,
        dropout=0.01,
    ):
        super(CustomLSTM, self).__init__()

        self.hidden_size = hidden_size

        self.embedding = nn.Linear(dynamic_input_size + static_input_size, hidden_size)

        self.lstm_cell = CellLSTM(hidden_size, hidden_size)

        self.dropout = nn.Dropout(p=dropout)

        self.head = nn.Linear(hidden_size, output_size)

    def forward(self, x, h_0=None, c_0=None):
        emb = self.embedding(x)

        batch_size, seq_len, _ = emb.size()

        if h_0 is None:
            h_0 = emb.data.new(batch_size, self.hidden_size).zero_()
        if c_0 is None:
            c_0 = emb.data.new(batch_size, self.hidden_size).zero_()

        h_n, c_n = [], []
        h_x = (h_0, c_0)

        for t in range(seq_len):
            h_0, c_0 = h_x

            cell_out = self.lstm_cell(emb[:, t, :], h_0, c_0)

            h_x = (cell_out[0], cell_out[1])

            h_n.append(cell_out[0])
            c_n.append(cell_out[1])

        h_n = torch.stack(h_n, 1)
        c_n = torch.stack(c_n, 1)

        out = self.head(self.dropout(h_n))

        return out
