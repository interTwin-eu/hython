import torch
from torch import nn
from typing import List


class ConvLSTMCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size, bias):
        """
        The ConvLSTM cell operates at each element of the sequence.

        Parameters
        ----------
        input_dim: int
            Number of channels of input tensor.
        hidden_dim: int
            Number of channels of hidden state.
        kernel_size: (int, int)
            Size of the convolutional kernel.
        bias: bool
            Whether or not to add the bias.
        """

        super(ConvLSTMCell, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.kernel_size = kernel_size
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2
        self.bias = bias

        # input layer convolution
        self.conv_x = nn.Conv2d(
            in_channels=self.input_dim,
            out_channels=self.hidden_dim * 4,  # will split in 4 input weight matrices
            kernel_size=self.kernel_size,
            dilation=1,
            padding=self.padding,
            bias=self.bias,
        )
        # hidden layer convolution
        self.conv_h = nn.Conv2d(
            in_channels=self.hidden_dim,
            out_channels=self.hidden_dim * 4,  # will split in 4 hidden weight matrices
            kernel_size=self.kernel_size,
            dilation=1,
            padding=self.padding,
            bias=self.bias,
        )

        self.conv = nn.Conv2d(
            in_channels=self.input_dim + self.hidden_dim,
            out_channels=4 * self.hidden_dim,
            dilation=1,  # FIXME: HARDCODED
            kernel_size=self.kernel_size,
            padding=self.padding,
            bias=self.bias,
        )

    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state

        x_concat = self.conv_x(input_tensor)
        h_concat = self.conv_h(h_cur)

        i_x, f_x, o_x, g_x = torch.split(x_concat, self.hidden_dim, dim=1)  # split
        i_h, f_h, o_h, g_h = torch.split(h_concat, self.hidden_dim, dim=1)  # split

        i = torch.sigmoid(i_x + i_h)
        f = torch.sigmoid(f_x + f_h)
        o = torch.sigmoid(o_x + o_h)

        g = torch.tanh(g_x + g_h)

        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)

        return h_next, c_next

    def init_hidden(self, batch_size, image_size):
        height, width = image_size
        return (
            torch.zeros(
                batch_size,
                self.hidden_dim,
                height,
                width,
                device=self.conv.weight.device,
            ),
            torch.zeros(
                batch_size,
                self.hidden_dim,
                height,
                width,
                device=self.conv.weight.device,
            ),
        )


class ConvLSTM(nn.Module):
    def __init__(
        self,
        input_dim: int = 2,
        output_dim: int = 2,
        hidden_dim: int = 24,
        kernel_size: list[int] | tuple[int] = [3, 3],
        num_layers: int = 1,
        batch_first: bool = False,
        bias: bool = True,
        return_all_layers: bool = False,
    ):
        """_summary_

        Parameters
        ----------
        input_dim : int
            Input dimensions, by default 2
        output_dim : int
            Output dimensions, by default 2
        hidden_dim : int, optional
            Hidden or filter dimensions, by default 24
        kernel_size : list[int] | tuple[int], optional
            Kernel size, by default [3,3]
        num_layers : int, optional
            Number of convolutional layers, by default 1
        batch_first : bool, optional
            Decide whether the batch dimension should be first or second, by default False
        bias : bool, optional
            Add bias, by default True
        return_all_layers : bool, optional
            Whether returning all the stacked convolutional layers or only the last one, by default False

        """
        super(ConvLSTM, self).__init__()

        kernel_size = self._extend_for_multilayer(kernel_size, num_layers)
        hidden_dim = self._extend_for_multilayer(hidden_dim, num_layers)

        if not len(kernel_size) == len(hidden_dim) == num_layers:
            raise ValueError("Inconsistent list length.")

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.bias = bias
        self.return_all_layers = return_all_layers

        cell_list = []
        for i in range(0, self.num_layers):
            cur_input_dim = self.input_dim if i == 0 else self.hidden_dim[i - 1]

            cell_list.append(
                ConvLSTMCell(
                    input_dim=cur_input_dim,
                    hidden_dim=self.hidden_dim[i],
                    kernel_size=self.kernel_size[i],
                    bias=self.bias,
                )
            )

        self.cell_list = nn.ModuleList(cell_list)

        # output

        self.fc1 = nn.Linear(hidden_dim[-1], output_dim)

    def forward(self, input_tensor, hidden_state=None):
        """
        Parameters
        ----------
        input_tensor:
            5-D Tensor either of shape (t, n, c, h, w) or (n, t, c, h, w)
        hidden_state:
            None.

        Returns
        -------
        last_state_list, layer_output
        """
        if not self.batch_first:
            # (t, b, c, h, w) -> (b, t, c, h, w)
            input_tensor = input_tensor.permute(1, 0, 2, 3, 4)

        b, it, _, h, w = input_tensor.size()

        # Implement stateful ConvLSTM
        if hidden_state is not None:
            raise NotImplementedError()
        else:
            # Since the init is done in forward. Can send image size here
            hidden_state = self._init_hidden(batch_size=b, image_size=(h, w))

        layer_output_list = []
        last_state_list = []

        seq_len = it
        cur_layer_input = input_tensor

        for layer_idx in range(self.num_layers):
            h, c = hidden_state[layer_idx]
            layer_idx_hidden_sequence = []
            for t in range(seq_len):
                h, c = self.cell_list[layer_idx](
                    input_tensor=cur_layer_input[:, t, :, :, :], cur_state=[h, c]
                )
                layer_idx_hidden_sequence.append(h)

            layer_output = torch.stack(layer_idx_hidden_sequence, dim=1)
            cur_layer_input = layer_output

            layer_output_list.append(layer_output)
            last_state_list.append([h, c])

        if not self.return_all_layers:
            layer_output_list = layer_output_list[-1:]  # N L C H W
            last_state_list = last_state_list[-1:]

        # FC Head
        out = torch.permute(layer_output_list[0], (0, 1, 3, 4, 2))  # N L H W Ch

        out = self.fc1(torch.relu(out))  # N L H W Cout

        return out, last_state_list

    def _init_hidden(self, batch_size, image_size):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.cell_list[i].init_hidden(batch_size, image_size))
        return init_states

    @staticmethod
    def _extend_for_multilayer(param, num_layers):
        if not isinstance(param, list) or not isinstance(param, tuple):
            param = [param] * num_layers
        return param
