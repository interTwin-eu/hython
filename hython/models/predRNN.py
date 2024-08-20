import torch
from torch import nn
import torch.nn.functional as F


class SpatioTemporalLSTMCell(nn.Module):
    def __init__(self, in_channel, num_hidden, width, filter_size, stride, layer_norm):
        super(SpatioTemporalLSTMCell, self).__init__()

        self.num_hidden = num_hidden
        self.padding = filter_size // 2
        self._forget_bias = 1.0
        if layer_norm:
            self.conv_x = nn.Sequential(
                nn.Conv2d(in_channel, num_hidden * 7, kernel_size=filter_size, stride=stride, padding=self.padding),
                nn.LayerNorm([num_hidden * 7, width, width])
            )
            self.conv_h = nn.Sequential(
                nn.Conv2d(num_hidden, num_hidden * 4, kernel_size=filter_size, stride=stride, padding=self.padding),
                nn.LayerNorm([num_hidden * 4, width, width])
            )
            self.conv_a = nn.Sequential(
                nn.Conv2d(num_hidden, num_hidden * 4, kernel_size=filter_size, stride=stride, padding=self.padding),
                nn.LayerNorm([num_hidden * 4, width, width])
            )
            self.conv_m = nn.Sequential(
                nn.Conv2d(num_hidden, num_hidden * 3, kernel_size=filter_size, stride=stride, padding=self.padding),
                nn.LayerNorm([num_hidden * 3, width, width])
            )
            self.conv_o = nn.Sequential(
                nn.Conv2d(num_hidden * 2, num_hidden, kernel_size=filter_size, stride=stride, padding=self.padding),
                nn.LayerNorm([num_hidden, width, width])
            )
        else:
            self.conv_x = nn.Sequential(
                nn.Conv2d(in_channel, num_hidden * 7, kernel_size=filter_size, stride=stride, padding=self.padding),
            )
            self.conv_h = nn.Sequential(
                nn.Conv2d(num_hidden, num_hidden * 4, kernel_size=filter_size, stride=stride, padding=self.padding),
            )
            self.conv_a = nn.Sequential(
                nn.Conv2d(num_hidden, num_hidden * 4, kernel_size=filter_size, stride=stride, padding=self.padding),
            )
            self.conv_m = nn.Sequential(
                nn.Conv2d(num_hidden, num_hidden * 3, kernel_size=filter_size, stride=stride, padding=self.padding),
            )
            self.conv_o = nn.Sequential(
                nn.Conv2d(num_hidden * 2, num_hidden, kernel_size=filter_size, stride=stride, padding=self.padding),
            )
        self.conv_last = nn.Conv2d(num_hidden * 2, num_hidden, kernel_size=1, stride=1, padding=0)

    def forward(self, x_t, h_t, c_t, m_t, a_t):
        x_concat = self.conv_x(x_t)
        h_concat = self.conv_h(h_t)
        a_concat = self.conv_a(a_t)
        m_concat = self.conv_m(m_t)
        i_x, f_x, g_x, i_x_prime, f_x_prime, g_x_prime, o_x = torch.split(x_concat, self.num_hidden, dim=1)
        i_h, f_h, g_h, o_h = torch.split(h_concat * a_concat, self.num_hidden, dim=1)
        i_m, f_m, g_m = torch.split(m_concat, self.num_hidden, dim=1)

        i_t = torch.sigmoid(i_x + i_h)
        f_t = torch.sigmoid(f_x + f_h + self._forget_bias)
        g_t = torch.tanh(g_x + g_h)

        delta_c = i_t * g_t
        c_new = f_t * c_t + delta_c

        i_t_prime = torch.sigmoid(i_x_prime + i_m)
        f_t_prime = torch.sigmoid(f_x_prime + f_m + self._forget_bias)
        g_t_prime = torch.tanh(g_x_prime + g_m)

        delta_m = i_t_prime * g_t_prime
        m_new = f_t_prime * m_t + delta_m

        mem = torch.cat((c_new, m_new), 1)
        o_t = torch.sigmoid(o_x + o_h + self.conv_o(mem))
        h_new = o_t * torch.tanh(self.conv_last(mem))

        return h_new, c_new, m_new, delta_c, delta_m





class PredRNN(nn.Module):
    def __init__(self, num_layers, num_hidden, configs):
        super(PredRNN, self).__init__()

        self.configs = configs
        self.conv_on_input = self.configs.conv_on_input
        self.res_on_conv = self.configs.res_on_conv
        self.patch_height = configs.img_width // configs.patch_size
        self.patch_width = configs.img_width // configs.patch_size
        self.patch_ch = configs.img_channel * (configs.patch_size ** 2)
        self.action_ch = configs.num_action_ch
        self.rnn_height = self.patch_height
        self.rnn_width = self.patch_width

        if self.configs.conv_on_input == 1:
            self.rnn_height = self.patch_height // 4
            self.rnn_width = self.patch_width // 4
            self.conv_input1 = nn.Conv2d(self.patch_ch, num_hidden[0] // 2,
                                         configs.filter_size,
                                         stride=2, padding=configs.filter_size // 2, bias=False)
            self.conv_input2 = nn.Conv2d(num_hidden[0] // 2, num_hidden[0], configs.filter_size, stride=2,
                                         padding=configs.filter_size // 2, bias=False)
            self.action_conv_input1 = nn.Conv2d(self.action_ch, num_hidden[0] // 2,
                                                configs.filter_size,
                                                stride=2, padding=configs.filter_size // 2, bias=False)
            self.action_conv_input2 = nn.Conv2d(num_hidden[0] // 2, num_hidden[0], configs.filter_size, stride=2,
                                                padding=configs.filter_size // 2, bias=False)
            self.deconv_output1 = nn.ConvTranspose2d(num_hidden[num_layers - 1], num_hidden[num_layers - 1] // 2,
                                                     configs.filter_size, stride=2, padding=configs.filter_size // 2,
                                                     bias=False)
            self.deconv_output2 = nn.ConvTranspose2d(num_hidden[num_layers - 1] // 2, self.patch_ch,
                                                     configs.filter_size, stride=2, padding=configs.filter_size // 2,
                                                     bias=False)
        self.num_layers = num_layers
        self.num_hidden = num_hidden
        cell_list = []
        self.beta = configs.decouple_beta
        self.MSE_criterion = nn.MSELoss().cuda()
        self.norm_criterion = nn.SmoothL1Loss().cuda()

        for i in range(num_layers):
            if i == 0:
                in_channel = self.patch_ch + self.action_ch if self.configs.conv_on_input == 0 else num_hidden[0]
            else:
                in_channel = num_hidden[i - 1]
            cell_list.append(
                SpatioTemporalLSTMCell(in_channel, num_hidden[i], self.rnn_width,
                                       configs.filter_size, configs.stride, configs.layer_norm)
            )
        self.cell_list = nn.ModuleList(cell_list)
        if self.configs.conv_on_input == 0:
            self.conv_last = nn.Conv2d(num_hidden[num_layers - 1], self.patch_ch + self.action_ch, 1, stride=1,
                                       padding=0, bias=False)
        self.adapter = nn.Conv2d(num_hidden[num_layers - 1], num_hidden[num_layers - 1], 1, stride=1, padding=0,
                                 bias=False)

    def forward(self, all_frames, mask_true):
        # [batch, length, height, width, channel] -> [batch, length, channel, height, width]
        frames = all_frames.permute(0, 1, 4, 2, 3).contiguous()
        input_frames = frames[:, :, :self.patch_ch, :, :]
        input_actions = frames[:, :, self.patch_ch:, :, :]
        mask_true = mask_true.permute(0, 1, 4, 2, 3).contiguous()

        next_frames = []
        h_t = []
        c_t = []
        delta_c_list = []
        delta_m_list = []

        for i in range(self.num_layers):
            zeros = torch.zeros(
                [self.configs.batch_size, self.num_hidden[i], self.rnn_height, self.rnn_width]).cuda()
            h_t.append(zeros)
            c_t.append(zeros)
            delta_c_list.append(zeros)
            delta_m_list.append(zeros)

        decouple_loss = []
        memory = torch.zeros([self.configs.batch_size, self.num_hidden[0], self.rnn_height, self.rnn_width]).cuda()

        for t in range(self.configs.total_length - 1):
            if t == 0:
                net = input_frames[:, t]
            else:
                net = mask_true[:, t - 1] * input_frames[:, t] + \
                      (1 - mask_true[:, t - 1]) * x_gen
            action = input_actions[:, t]

            if self.conv_on_input == 1:
                net_shape1 = net.size()
                net = self.conv_input1(net)
                if self.res_on_conv == 1:
                    input_net1 = net
                net_shape2 = net.size()
                net = self.conv_input2(net)
                if self.res_on_conv == 1:
                    input_net2 = net
                action = self.action_conv_input1(action)
                action = self.action_conv_input2(action)

            h_t[0], c_t[0], memory, delta_c, delta_m = self.cell_list[0](net, h_t[0], c_t[0], memory, action)
            delta_c_list[0] = F.normalize(self.adapter(delta_c).view(delta_c.shape[0], delta_c.shape[1], -1), dim=2)
            delta_m_list[0] = F.normalize(self.adapter(delta_m).view(delta_m.shape[0], delta_m.shape[1], -1), dim=2)

            for i in range(1, self.num_layers):
                h_t[i], c_t[i], memory, delta_c, delta_m = self.cell_list[i](h_t[i - 1], h_t[i], c_t[i], memory, action)
                delta_c_list[i] = F.normalize(self.adapter(delta_c).view(delta_c.shape[0], delta_c.shape[1], -1), dim=2)
                delta_m_list[i] = F.normalize(self.adapter(delta_m).view(delta_m.shape[0], delta_m.shape[1], -1), dim=2)

            for i in range(0, self.num_layers):
                decouple_loss.append(torch.mean(torch.abs(
                    torch.cosine_similarity(delta_c_list[i], delta_m_list[i], dim=2))))
            if self.conv_on_input == 1:
                if self.res_on_conv == 1:
                    x_gen = self.deconv_output1(h_t[self.num_layers - 1] + input_net2, output_size=net_shape2)
                    x_gen = self.deconv_output2(x_gen + input_net1, output_size=net_shape1)
                else:
                    x_gen = self.deconv_output1(h_t[self.num_layers - 1], output_size=net_shape2)
                    x_gen = self.deconv_output2(x_gen, output_size=net_shape1)
            else:
                x_gen = self.conv_last(h_t[self.num_layers - 1])
            next_frames.append(x_gen)

        decouple_loss = torch.mean(torch.stack(decouple_loss, dim=0))
        # [length, batch, channel, height, width] -> [batch, length, height, width, channel]
        next_frames = torch.stack(next_frames, dim=0).permute(1, 0, 3, 4, 2).contiguous()
        loss = self.MSE_criterion(next_frames,
                                  all_frames[:, 1:, :, :, :next_frames.shape[4]]) + self.beta * decouple_loss
        next_frames = next_frames[:, :, :, :, :self.patch_ch]
        return next_frames, loss
    



class LayerNorm2D(nn.LayerNorm):
    def __init__(self, num_channels, eps=1e-6, affine=True):
        super().__init__(num_channels, eps=eps, elementwise_affine=affine)

    def forward(self, x):
        return F.layer_norm(
            x.permute(0, 2, 3, 1), self.normalized_shape, self.weight, self.bias, self.eps
        ).permute(0, 3, 1, 2)

class ActionSTLSTMCell(nn.Module):
    def __init__(self, in_channel, action_channel, num_hidden, filter_size, stride):
        super().__init__()

        self.num_hidden = num_hidden
        self.padding = filter_size // 2
        self._forget_bias = 1.0
        self.conv_x = nn.Sequential(
            nn.Conv2d(in_channel, num_hidden * 7, filter_size, stride, self.padding, padding_mode='reflect'),
            LayerNorm2D(num_hidden * 7)
        )
        self.conv_a = nn.Sequential(
            nn.Conv2d(action_channel, num_hidden * 4, filter_size, stride, self.padding, padding_mode='reflect'),
            LayerNorm2D(num_hidden * 4)
        )
        self.conv_h = nn.Sequential(
            nn.Conv2d(num_hidden, num_hidden * 4, filter_size, stride, self.padding, padding_mode='reflect'),
            LayerNorm2D(num_hidden * 4)
        )
        self.conv_m = nn.Sequential(
            nn.Conv2d(num_hidden, num_hidden * 3, filter_size, stride, self.padding, padding_mode='reflect'),
            LayerNorm2D(num_hidden * 3)
        )
        self.conv_o = nn.Sequential(
            nn.Conv2d(num_hidden * 2, num_hidden,     filter_size, stride, self.padding, padding_mode='reflect'),
            LayerNorm2D(num_hidden)
        )
        self.conv_last = nn.Conv2d(num_hidden * 2, num_hidden, kernel_size=1, stride=1, padding=0)

    def forward(self, x_t, a_t, h_t, c_t, m_t):
        x_conv = self.conv_x(x_t)
        a_conv = self.conv_a(a_t)
        h_conv = self.conv_h(h_t)
        m_conv = self.conv_m(m_t)
        i_x, f_x, g_x, i_xp, f_xp, g_xp, o_x = torch.split(x_conv, self.num_hidden, dim=1)
        i_h, f_h, g_h, o_h = torch.split(h_conv * a_conv, self.num_hidden, dim=1)
        i_m, f_m, g_m = torch.split(m_conv, self.num_hidden, dim=1)

        i_t = torch.sigmoid(i_x + i_h)
        f_t = torch.sigmoid(f_x + f_h + self._forget_bias)
        g_t = torch.tanh(g_x + g_h)

        i_tp = torch.sigmoid(i_xp + i_m)
        f_tp = torch.sigmoid(f_xp + f_m + self._forget_bias)
        g_tp = torch.tanh(g_xp + g_m)

        delta_c = i_t * g_t
        c_new = f_t * c_t + delta_c
        delta_m = i_tp * g_tp
        m_new = f_tp * m_t + delta_m

        mem = torch.cat((c_new, m_new), 1)
        o_t = torch.sigmoid(o_x + o_h + self.conv_o(mem))
        h_new = o_t * torch.tanh(self.conv_last(mem))
        return h_new, c_new, m_new, delta_c, delta_m





class ForcedSTRNN(nn.Module):
    def __init__(
        self,
        num_layers,
        num_hidden,
        img_channel, # states? inputs....
        act_channel, # dynamic/forcings
        init_cond_channel, # states?
        static_channel, # static
        out_channel,
        filter_size=5,
        stride=1,
    ):
        super().__init__()

        self.input_channel = img_channel
        self.action_channel = act_channel
        self.init_cond_channel = init_cond_channel
        self.static_channel = static_channel
        self.frame_channel = img_channel # inputs..
        self.num_layers = num_layers
        self.num_hidden = num_hidden
        self.out_channel = out_channel
        self.decouple_loss = None
        cell_list = []

        for i in range(num_layers):
            in_channel = self.frame_channel if i == 0 else num_hidden[i - 1]
            cell_list.append(ActionSTLSTMCell(
                in_channel, act_channel, num_hidden[i], filter_size, stride,
            ))
        self.cell_list = nn.ModuleList(cell_list)
        # Convolution on outputs
        self.conv_last = nn.Conv2d(
            num_hidden[num_layers - 1],
            self.out_channel,
            kernel_size=1,
            bias=False
        )
        # convolution on inputs
        adapter_num_hidden = num_hidden[0]
        self.adapter = nn.Conv2d(
            adapter_num_hidden,
            adapter_num_hidden,
            kernel_size=1,
            bias=False
        )

        self.memory_encoder = nn.Conv2d(
            self.init_cond_channel, num_hidden[0], kernel_size=1, bias=True
        )
        self.cell_encoder = nn.Conv2d(
            self.static_channel, sum(num_hidden), kernel_size=1, bias=True
        )

    def update_state(self, state):
        out_shape = (state.shape[0], state.shape[1], -1)
        return F.normalize(self.adapter(state).view(out_shape), dim=2)

    def calc_decouple_loss(self, c, m):
        return torch.mean(torch.abs(torch.cosine_similarity(c, m, dim=2)))

    def forward(self, forcings, init_cond, static_inputs):
        # Input shape:
        #   (batch, length, channel, height, width)
        batch, timesteps, channels, height, width = forcings.shape

        # Initialize list of states
        next_frames = []
        h_t = []
        c_t = []
        delta_c_list = []
        delta_m_list = []
        decouple_loss = []
        for i in range(self.num_layers):
            zeros = torch.zeros([batch, self.num_hidden[i], height, width]).to(self.device)
            h_t.append(zeros)
            c_t.append(zeros)
            delta_c_list.append(zeros)
            delta_m_list.append(zeros)

        # Initialize memory state and first layer cell state
        # Note: static_inputs and init_cond should have length=1
        # TODO: Fix this requirement
        memory = self.memory_encoder(init_cond[:, 0])
        # TODO: Should we encode all layer cell states?
        c_t = list(torch.split(
            self.cell_encoder(static_inputs[:, 0]),
            self.num_hidden, dim=1
        ))

        # First input is the initial condition
        x = init_cond[:, 0]
        for t in range(timesteps):
            a = forcings[:, t] # FORCINGS ARE CONSIDERED ACTIONS
            h_t[0], c_t[0], memory, dc, dm = self.cell_list[0](x, a, h_t[0], c_t[0], memory)
            delta_c_list[0] = self.update_state(dc)
            delta_m_list[0] = self.update_state(dm)

            for i in range(1, self.num_layers): # LOOP OVER OTHER LAYERS
                h_t[i], c_t[i], memory, dc, dm = self.cell_list[i](h_t[i - 1], a, h_t[i], c_t[i], memory)
                delta_c_list[i] = self.update_state(dc)
                delta_m_list[i] = self.update_state(dm)

            x = self.conv_last(h_t[-1]) + x
            next_frames.append(x)

            # decoupling loss
            for i in range(self.num_layers):
                decouple_loss.append(self.calc_decouple_loss(delta_c_list[i], delta_m_list[i]))

        self.decouple_loss = torch.mean(torch.stack(decouple_loss, dim=0))
        # Stack to: [batch, length, channel, height, width]
        next_frames = torch.stack(next_frames, dim=1)
        #next_frames = torch.clamp(
        #    torch.stack(next_frames, dim=1)
        #    -10, 10
        #)
        return next_frames

    def training_step(self, train_batch, train_batch_idx):
        forcing, state, params, target = train_batch
        y_hat = self(forcing, state, params).squeeze()
        loss = self.loss_fun(y_hat, target)
        self.log('train_loss', loss)
        return loss




