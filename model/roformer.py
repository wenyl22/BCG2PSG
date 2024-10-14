import lightning as L
import torch
import torch.nn as nn
from transformers import RoFormerConfig, RoFormerModel
import typing as t
from wuji_dl.factory import get_class
from wuji_dl.ops.functional import length_mask
from model.cnndownsample import CNNDownSampling
from model.cnnupsample import CNNUpSampling
from model.flow_matching import FlowMatchingLoss

class RoformerBackBone(nn.Module):
    def __init__(self, **kwargs) -> None:
        super().__init__()
        self.feature_extractors = nn.ModuleList(
            [CNNDownSampling(**kwargs)]
        )
        self.stride = kwargs["stride"]
        config = RoFormerConfig(vocab_size=1, embedding_size=kwargs["embedding_dim"], **kwargs)
        print(config)
        self.roformer = RoFormerModel(config)
        self.hidden_size = config.hidden_size

    def _forward(self, length: torch.Tensor, **kwargs):
        #print("DEBUG: RoformerBackBone._forward")
        feats = [head(**kwargs) for head in self.feature_extractors]
        x = torch.stack(feats).sum(dim=0).transpose(1, 2)
        assert torch.all(length // self.stride * self.stride == length).item(), "Length must be divisible by stride"
        mask = length_mask(x.size(1), length//self.stride)
        return self.roformer(
            inputs_embeds=x, attention_mask=mask, output_hidden_states=True
        )

    def get_hidden_states(self, **kwargs):
        return self._forward(**kwargs).hidden_states

    def forward(self, **kwargs):
        return self._forward(**kwargs).last_hidden_state

class Roformer4SignalFiltering(nn.Module):
    def __init__(self, **kwargs) -> None:
        super().__init__()
        self.model = kwargs["model"]
        self.backbone = RoformerBackBone(**kwargs)
        self.out_proj = FlowMatchingLoss(model = {"x_dim": kwargs["stride"], "z_dim": kwargs["embedding_dim"], "hidden_dim": kwargs["embedding_dim"], "num_layers": kwargs["num_layers"]}, T = 50, shape = [kwargs["stride"]], name = self.model)
        self.device = kwargs["device"]
        self.stride = kwargs["stride"]
        self.embedding_dim = kwargs["embedding_dim"]

    def get_hidden(self, **kwargs):
        input = kwargs["input"].to(self.device)
        length = kwargs["length"].to(self.device)
        hidden = self.backbone(input = input, length = length)
        return hidden

    def get_rec(self, **kwargs):
        bsz = kwargs["input"].size(0)
        hidden = self.get_hidden(**kwargs).view(-1, self.embedding_dim)
        rec = self.out_proj.sample(hidden).view(bsz, -1)
        return rec
    
    def get_loss(self, **kwargs):
        hidden = self.get_hidden(**kwargs)
        hidden = hidden.view(-1, self.embedding_dim)
        rec = self.out_proj(hidden, kwargs["tar"].to(self.device).view(-1, self.stride))
        return rec
    
    def get_mse_loss(self, **kwargs):
        bsz = kwargs["input"].size(0)
        rec = self.get_rec(**kwargs).view(bsz, -1)
        return torch.mean((rec - kwargs["tar"].to(self.device))**2)

