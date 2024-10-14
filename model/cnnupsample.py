import torch
import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.nn as nn
import typing as t

class CNNUpSampling(nn.Module):
    def __init__(self, **kwargs) -> None:
        super(CNNUpSampling, self).__init__()
        self.in_channels = kwargs["in_channels"]
        self.trans_feature_extractors = nn.Sequential(
            nn.ConvTranspose1d(self.in_channels, 256, 4, 1, 0, bias=False), # 1 -> 4
            nn.BatchNorm1d(256),
            nn.ReLU(True),
            nn.ConvTranspose1d(256, 128, 3, 2, 1, bias=False), # 4 -> 7
            nn.BatchNorm1d(128),
            nn.ReLU(True),
            nn.ConvTranspose1d(128, 64, 4, 2, 1, bias=False), # 7 -> 14
            nn.BatchNorm1d(64),
            nn.ReLU(True),
            nn.ConvTranspose1d(64, 32, 4, 2, 0, bias=False), # 14 -> 30
            nn.BatchNorm1d(32),
            nn.ReLU(True),
            nn.ConvTranspose1d(32, 1, 4, 2, 1, bias=False), # 30 -> 60
        )
        for m in self.modules():
            if isinstance(m, nn.ConvTranspose1d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm1d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x is of shape (batch, length, embedding_dim)
        bsz = x.size(0)
        x = x.reshape(x.size(0) * x.size(1), x.size(2)).unsqueeze(-1)
        # print(x.shape, self.in_channels)

        # x is of shape (batch * length, embedding_dim)
        x = self.trans_feature_extractors(x).squeeze(-2)
        # print(x.shape)
        # x is of shape (batch * length, stride)
        x = x.reshape(bsz, -1, x.size(1))
        return x