import torch
import torch.nn as nn
import torch.nn.functional as F


def get_norm(name, num_channels, dim=None):
    if name == 'bn':
        return nn.BatchNorm1d(num_channels)
    elif 'gn' in name:
        num_groups = name[2:]
        if num_groups == '': num_groups = 8
        num_groups = int(num_groups)
        return nn.GroupNorm(num_groups, num_channels)
    elif name == 'in':
        return nn.GroupNorm(num_channels, num_channels)
    elif name == 'ln':
        return nn.GroupNorm(1, num_channels)

def get_non_lin(name):
    if name == 'relu':
        return nn.ReLU()
    if name == 'leaky_relu':
        return nn.LeakyReLU()
    if name == 'gelu':
        return nn.GELU()
    if name == 'elu':
        return nn.ELU()

def get_conv(dim, *args, **kwargs):
    return nn.Conv1d(*args, **kwargs)

def get_conv_block(dim, in_channels, out_channels, norm, non_lin, kernel_size=3, bias=True, padding='same',
                   padding_mode='zeros', dropout_rate=0.):
    if padding == 'same':
        padding = kernel_size // 2
    layers = [
        get_conv(dim, in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding=padding,
                 bias=bias, padding_mode=padding_mode)]
    if norm is not None:
        layers.append(get_norm(norm, num_channels=out_channels, dim=dim))
    if non_lin is not None:
        layers.append(get_non_lin(non_lin))
    if dropout_rate > 0:
        layers.append(nn.Dropout(dropout_rate))
    return nn.Sequential(*layers)


class UNetEncoder(nn.Module):
    def __init__(self,
                 dim,
                 in_channels,
                 num_stages,
                 initial_num_channels,
                 norm=None,
                 non_lin='relu',
                 kernel_size=3,
                 pooling='max',
                 bias=True,
                 padding='same',
                 padding_mode='zeros',
                 stride_sequence=None,
                 skip_connections=True,
                 dropout_rate=0
                 ):
        super().__init__()

        assert pooling in ['avg', 'max'], f"pooling can be 'avg' or 'max'"
        self.skip_connections = skip_connections

        if pooling == 'avg':
            pooling_class = nn.AvgPool1d
        else:
            pooling_class = nn.MaxPool1d

        if stride_sequence is None:
            stride_sequence = [2] * (num_stages - 1)

        self.module_list = nn.ModuleList()
        self.pooling_layers = nn.ModuleList()

        for i in range(num_stages):
            block_1_in_channels = in_channels if i == 0 else (2 ** i) * initial_num_channels
            block_1_out_channels = (2 ** i) * initial_num_channels
            block_2_in_channels = block_1_out_channels
            block_2_out_channels = (2 ** (i + 1)) * initial_num_channels
            m = nn.Sequential(
                get_conv_block(
                    dim=dim,
                    in_channels=block_1_in_channels,
                    out_channels=block_1_out_channels,
                    kernel_size=kernel_size,
                    norm=norm,
                    non_lin=non_lin,
                    bias=bias,
                    padding=padding,
                    padding_mode=padding_mode,
                    dropout_rate=dropout_rate
                ),
                get_conv_block(
                    dim=dim,
                    in_channels=block_2_in_channels,
                    out_channels=block_2_out_channels,
                    kernel_size=kernel_size,
                    norm=norm,
                    non_lin=non_lin,
                    bias=bias,
                    padding=padding,
                    padding_mode=padding_mode,
                    dropout_rate=dropout_rate
                )
            )
            self.module_list.append(m)
            if i < num_stages - 1:
                self.pooling_layers.append(pooling_class(3, stride_sequence[i], padding=1))

    def forward(self, x, print_shapes=False):

        acts = []
        for i, (m, p) in enumerate(zip(self.module_list[:-1], self.pooling_layers)):
            x = m(x)
            acts.append(x)
            x = p(x)
        x = self.module_list[-1](x)
        return x, acts


class UNetDecoder(nn.Module):
    def __init__(self,
                 dim,
                 out_channels,
                 num_stages,
                 initial_num_channels,
                 norm=None,
                 non_lin='relu',
                 kernel_size=3,
                 bias=True,
                 padding='same',
                 padding_mode='zeros',
                 skip_connections=False,
                 dropout_rate=0,
                 ):
        super().__init__()

        self.module_list = nn.ModuleList()

        for i in range(num_stages - 1):
            block_in_channels = (2 ** (i + 1) + (2 ** (i + 2))) * initial_num_channels
            block_out_channels = (2 ** (i + 1)) * initial_num_channels
            m = nn.Sequential(
                get_conv_block(
                    dim=dim,
                    in_channels=block_in_channels,
                    out_channels=block_out_channels,
                    kernel_size=kernel_size,
                    norm=norm,
                    non_lin=non_lin,
                    bias=bias,
                    padding=padding,
                    padding_mode=padding_mode,
                    dropout_rate=dropout_rate
                ),
                get_conv_block(
                    dim=dim,
                    in_channels=block_out_channels,
                    out_channels=block_out_channels,
                    kernel_size=kernel_size,
                    norm=norm,
                    non_lin=non_lin,
                    bias=bias,
                    padding=padding,
                    padding_mode=padding_mode,
                    dropout_rate=dropout_rate
                )
            )
            self.module_list.append(m)

        self.final_conv = get_conv(dim, 2 * initial_num_channels, out_channels, 1, bias=bias, padding=0,
                                   padding_mode=padding_mode)

    def forward(self, x, acts, print_shapes=False):

        interpolation = 'linear'
        for i, (y, m) in enumerate(zip(reversed(acts), reversed(self.module_list))):
            x = F.interpolate(x, y.shape[2:], mode=interpolation, align_corners=True)
            x = m(torch.cat([y, x], 1))

        x = self.final_conv(x)
        return x


class UNet(nn.Module):
    def __init__(self,
                 in_channels=1,
                 out_channels=1,
                 num_stages=3,
                 initial_num_channels=32,
                 n_freqs=100,
                 dim=1,
                 n_class=5,
                 seq_len=30,
                 norm=None,
                 non_lin='relu',
                 kernel_size=11,
                 pooling='max',
                 bias=True,
                 padding='same',
                 padding_mode='zeros',
                 stride_sequence=None,
                 skip_connections=True,
                 dropout_rate=0,
                 ):
        super().__init__()

        self.encoder = UNetEncoder(
            dim=dim,
            in_channels=in_channels,
            num_stages=num_stages,
            initial_num_channels=initial_num_channels,
            norm=norm,
            non_lin=non_lin,
            kernel_size=kernel_size,
            pooling=pooling,
            bias=bias,
            padding=padding,
            padding_mode=padding_mode,
            stride_sequence=stride_sequence,
            skip_connections=skip_connections,
            dropout_rate=dropout_rate
        )
        self.decoder = UNetDecoder(
            dim=dim,
            out_channels=out_channels,
            num_stages=num_stages,
            initial_num_channels=initial_num_channels,
            norm=norm,
            non_lin=non_lin,
            kernel_size=kernel_size,
            bias=bias,
            padding=padding,
            padding_mode=padding_mode,
            skip_connections=skip_connections,
            dropout_rate=dropout_rate
        )
        self.classifier = nn.Linear(seq_len*n_freqs, n_class)

    def forward(self, batch, print_shapes=False, classification=True):

        if isinstance(batch, dict):
            x = torch.cat([batch['respiration'].unsqueeze(1), batch['RRI'].unsqueeze(1)], dim=1)
        else:
            x = batch

        if print_shapes:
            print('Input shape:', x.shape)

        x, acts = self.encoder(x, print_shapes=print_shapes)
        x = self.decoder(x, acts, print_shapes=print_shapes)

        if print_shapes:
            print('Output shape:', x.shape)
        x = x.reshape(x.shape[0], -1)
        if classification:
            return self.classifier(x)
        else:
            return x
    def get_loss(self, batch):
        bcg = batch["BCG"].unsqueeze(1).to("cuda")
        ecg = batch["ECG"].to("cuda")
        rec = self(bcg, classification=False)
        return F.mse_loss(rec, ecg)