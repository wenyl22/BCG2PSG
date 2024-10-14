import torch
import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.nn as nn
import typing as t
from wuji_dl.ops.res_block1d import ResBlock1d


class ResNetFeatureExtractor(nn.Module):
    def __init__(
        self,
        channels: int,
        layers: list[int],
        cnn_strides: list[int],
        embedding_dim: int = 128,
        norm_layer: t.Callable[[int], nn.Module] = nn.BatchNorm1d,
    ) -> None:
        super().__init__()
        self._norm_layer = norm_layer
        self.inplanes = embedding_dim // 8
        self.layer0 = nn.Sequential(
            nn.Conv1d(channels, self.inplanes, 7, stride=1, padding=3, bias=False),
            norm_layer(self.inplanes),
            nn.ReLU(),
        )
        self.layer1 = self._make_layer(embedding_dim // 8, layers[0])
        self.layer2 = self._make_layer(embedding_dim // 4, layers[1], stride=cnn_strides[0])
        self.layer3 = self._make_layer(embedding_dim // 2, layers[2], stride=cnn_strides[1])
        self.layer4 = self._make_layer(embedding_dim, layers[3], stride=cnn_strides[2])
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm1d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, planes: int, blocks: int, stride: int = 1) -> nn.Sequential:
        downsample = None
        if stride != 1 or self.inplanes != planes:
            downsample = nn.Sequential(
                nn.Conv1d(self.inplanes, planes, 1, stride=stride, bias=False),
                self._norm_layer(planes),
            )
        layers = [ResBlock1d(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes
        for _ in range(1, blocks):
            layers.append(ResBlock1d(planes, planes, 1, norm_layer=self._norm_layer))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x

class CNNDownSampling(nn.Module):
    def __init__(self, **kwargs) -> None:
        super(CNNDownSampling, self).__init__()
        self.feature_extractor = ResNetFeatureExtractor(kwargs["in_channels"], [2, 2, 2, 2], kwargs["cnn_strides"], kwargs["embedding_dim"])

    def forward(self, **kwargs) -> torch.Tensor:
        #print("DEBUG: CNNDownSampling.forward")
        # x: (batch_size, channels, seq_len)
        x = kwargs["input"]
        if len(x.size()) == 2:
            x = x.unsqueeze(1)
        x = self.feature_extractor(x)
        return x