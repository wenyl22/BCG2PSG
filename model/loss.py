import torch
import torch.nn as nn
from torchaudio.transforms import Spectrogram
import typing as t


class SpecAmpLoss(nn.Module):
    def __init__(
        self,
        field: int,
        stride: int,
        power: float = 2,
        window: str = "ones",
        reduction: t.Literal["none", "mean", "sum"] = "mean",
    ) -> None:
        super().__init__()
        scale = torch.ones(field // 2 + 1) * (2 ** 0.5)
        scale[0] = 1
        if field % 2 == 0:
            scale[field // 2] = 1
        self.register_buffer("scale", scale[:, None])
        self.spec = Spectrogram(
            n_fft=field,
            hop_length=stride,
            window_fn=getattr(torch, window),
            power=1,
            center=False,
            normalized=True,
        )
        self.power = power
        self.reduction = reduction

    def forward(self, x: torch.Tensor, x_hat: torch.Tensor) -> torch.Tensor:
        x = self.spec(x) * self.scale
        x_hat = self.spec(x_hat) * self.scale
        loss = torch.sum((x - x_hat).abs() ** self.power, dim=1)
        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss
