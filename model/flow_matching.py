import torch
import torch.nn as nn
import torch.nn.functional as F
import typing as t
from diffusers.models.embeddings import get_timestep_embedding, TimestepEmbedding
from timm.models.vision_transformer import Mlp
def modulate(x, shift, scale):
    return x * (1 + scale) + shift

class Spo2FlowMLP(nn.Module):
    def __init__(self, x_dim: int, z_dim: int, hidden_dim: int, num_layers: int):
        super().__init__()
        self.dim = hidden_dim
        self.x_emb = nn.Linear(x_dim, hidden_dim)
        self.z_emb = nn.Linear(z_dim, hidden_dim)
        self.t_emb = TimestepEmbedding(hidden_dim, hidden_dim)
        self.layers = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for i in range(1, num_layers)])
        self.x_out = nn.Linear(hidden_dim, x_dim)

    def forward(self, x: torch.Tensor, z: torch.Tensor, t: torch.Tensor):
        x = self.x_emb(x)
        z = self.z_emb(z)
        t = get_timestep_embedding(t, self.dim, scale=10000)
        t = self.t_emb(t)
        x = F.gelu(x + z + t)
        for f in self.layers:
            x = F.gelu(f(x))
        x = self.x_out(x)
        return x

class DiTFlow(nn.Module):
    def __init__(self, x_dim: int, z_dim: int, hidden_dim: int, num_layers: int):
        super().__init__()
        self.dim = hidden_dim
        self.x_emb = nn.Linear(x_dim, hidden_dim)
        self.z_emb = nn.Linear(z_dim, hidden_dim)
        self.t_emb = TimestepEmbedding(hidden_dim, hidden_dim)
        self.norm1 = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_dim, 6 * hidden_dim, bias=True)
        )
        # N x H -> N x 6H
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(in_features=hidden_dim, hidden_features=4 * hidden_dim, act_layer=approx_gelu, drop=0)
        self.layers = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for i in range(1, num_layers)])
        self.LN = nn.ModuleList([nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6) for i in range(1, num_layers)])
        self.x_out = nn.Linear(hidden_dim, x_dim)

    def forward(self, x: torch.Tensor, z: torch.Tensor, t: torch.Tensor):
        z = self.z_emb(z)
        t = self.t_emb(get_timestep_embedding(t, self.dim, scale=10000))
        c = z + t
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
        x = self.x_emb(x)
        x = x + gate_msa * modulate(self.norm1(x), shift_msa, scale_msa)
        x = x + gate_mlp * self.mlp(modulate(self.norm2(x), shift_mlp, 
        scale_mlp))
        for i in range(len(self.layers)):
            x = self.LN[i](x)
            x = F.gelu(self.layers[i](x) + x)

        x = self.x_out(x)
        return x

def extend(x: torch.Tensor, ndim: int):
    ext = [1] * (ndim - x.ndim)
    return x.view(*x.shape, *ext)

class FlowMatchingLoss(nn.Module):
    def __init__(self, model: dict, T: int, shape: list[int], name: str):
        super().__init__()
        if name == "roformer":
            self.model = Spo2FlowMLP(**model)
        elif name  == "dit":
            self.model = DiTFlow(**model)

        self.T = T
        self.shape = shape
        self.model: t.Callable[..., torch.Tensor]

    def forward(self, z: torch.Tensor, x1: torch.Tensor):
        #print("DEBUG: FlowMatchingLoss.forward")
        x0 = torch.randn_like(x1)
        t = torch.rand((len(x1),), device=x1.device)
        u = x1 - x0
        xt = x0 + extend(t, u.ndim) * u
        v = self.model(xt, z, t)
        #print("DEBUG: FlowMatchingLoss.forward end")
        return ((u - v) ** 2).mean()

    def sample(self, z: torch.Tensor):
        # use a deterministic prior
        x = torch.zeros(len(z), *self.shape, device=z.device)
        for i in range(self.T):
            t = i / self.T
            t = torch.full((len(z),), t, device=z.device)
            v = self.model(x, z, t)
            x = x + v / self.T
        return x