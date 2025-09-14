from einops import rearrange
import math
import torch
from torch import nn
from nnssl.architectures.vocoV2_architecture import VoCoV2Architecture, VoCoV2EvaArchitecture


class TransformerProjectionHead(nn.Module):
    def __init__(self, total_channels: int, hidden_dim: int, output_dim: int, num_layers=1, num_heads=16):
        super(TransformerProjectionHead, self).__init__()
        self.hidden_dim = hidden_dim
        self.in_proj = nn.Linear(total_channels, hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim,
            dropout=0.1,
            activation='relu',
            batch_first=True,
            norm_first=False,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer=encoder_layer, num_layers=num_layers)
        self.out_proj = nn.Linear(hidden_dim, output_dim)

    def pe_sincos_2d(self, coords_2d, dim, T=10000.0):
        """
        coords_2d: [B, N, 2]
        dim: PE dim, divisible by 4 (sin & cos, coord_x & coord_y)
        returns: [B, N, dim]
        """
        assert dim % 4 == 0
        dim_sub = dim // 4
        freqs = 1.0 / (T ** (torch.arange(dim_sub, device=coords_2d.device) / dim_sub))  # [dim_sub]

        def enc(v):  # v: [B, N, 1]
            v = v * freqs  # broadcast
            return torch.cat([torch.sin(2 * math.pi * v), torch.cos(2 * math.pi * v)], dim=-1)  # [B, N, 2 * dim_sub]

        x, y = coords_2d[..., 0:1], coords_2d[..., 1:2]
        pe = torch.cat([enc(x), enc(y)], dim=-1)
        return pe

    def forward(self, x: list[torch.Tensor], coords_2d) -> torch.Tensor:
        """
        x: [B, N, D]
        coords_2d: [B, N, 2]
        """
        x = self.in_proj(x)
        pe = self.pe_sincos_2d(coords_2d, self.hidden_dim)
        x = x + pe
        x = self.transformer(x)
        x = self.out_proj(x)
        return x


class VoCoV2TfProjArchitecture(VoCoV2Architecture):
    def __init__(self, encoder: nn.Module,  features: list[int]):
        super().__init__(encoder=encoder, features=features)
        total_features = sum(features)
        self.projector_tea = TransformerProjectionHead(total_features, 2048, 2048)
        self.projector_stu = TransformerProjectionHead(total_features, 2048, 2048)
        for param_tea, param_stu in zip(self.projector_tea.parameters(), self.projector_stu.parameters()):
            param_tea.data.copy_(param_stu.data)

    def forward(self, x, top_left_xy, NBASE, batch_size):
        # top_left_xy: [B, n_base + n_target, 2]
        out = self.encoder(x)
        flat_out = torch.concat([self.adaptive_pool(o) for o in out], dim=1)
        flat_out = torch.reshape(flat_out, (flat_out.shape[0], -1))

        _base_out = rearrange(flat_out[:NBASE], "(b NBASE) c -> b NBASE c", b=batch_size)
        _target_out = rearrange(flat_out[NBASE:], "(b nTARGET) c -> b nTARGET c", b=batch_size)
        flat_out = torch.cat([_base_out, _target_out], dim=1) # [B, n_base + n_target, in_dim]

        self._ema_update_teacher()
        with torch.no_grad():
            x_tea = self.projector_tea(flat_out, top_left_xy)
        x_stu = self.projector_stu(self.dropout(flat_out), top_left_xy)
        return x_tea, x_stu # [B, n_base + n_target, out_dim]


class VoCoV2TfProjEvaArchitecture(VoCoV2EvaArchitecture):
    """
    We don't have multiple CNN stages that we can take the features from and concatenate them, so for the transformer
    we only use the features from the (last) output layer.
    """
    def __init__(self, encoder: nn.Module, embed_dim: int):
        super().__init__(encoder=encoder, embed_dim=embed_dim)
        self.projector_tea = TransformerProjectionHead(embed_dim, 2048, 2048)
        self.projector_stu = TransformerProjectionHead(embed_dim, 2048, 2048)
        for param_tea, param_stu in zip(self.projector_tea.parameters(), self.projector_stu.parameters()):
            param_tea.data.copy_(param_stu.data)

    def forward(self, x, top_left_xy, NBASE, batch_size):
        # top_left_xy: [B, n_base + n_target, 2]
        out = self.encoder(x)
        flat_out = torch.concat([self.adaptive_pool(o) for o in out], dim=1)
        flat_out = torch.reshape(flat_out, (flat_out.shape[0], -1))

        _base_out = rearrange(flat_out[:NBASE], "(b NBASE) c -> b NBASE c", b=batch_size)
        _target_out = rearrange(flat_out[NBASE:], "(b nTARGET) c -> b nTARGET c", b=batch_size)
        flat_out = torch.cat([_base_out, _target_out], dim=1) # [B, n_base + n_target, in_dim]

        self._ema_update_teacher()
        with torch.no_grad():
            x_tea = self.projector_tea(flat_out, top_left_xy)
        x_stu = self.projector_stu(self.dropout(flat_out), top_left_xy)
        return x_tea, x_stu # [B, n_base + n_target, out_dim]
