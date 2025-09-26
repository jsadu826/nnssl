import math

import torch
from einops import rearrange
from torch import nn

from nnssl.architectures.vocoV2TfProj_architecture import VoCoV2TfProjArchitecture


class VoCoV2TfProjReconArchitecture(VoCoV2TfProjArchitecture):
    def __init__(self, encoder: nn.Module, decoder: nn.Module, features: list[int]):
        super().__init__(encoder=encoder, features=features)
        self.decoder = decoder

    def forward(self, x, mixed_image, top_left_xy, NBASE, batch_size):
        out = self.encoder(x)
        flat_out = torch.concat([self.adaptive_pool(o) for o in out], dim=1)
        flat_out = torch.reshape(flat_out, (flat_out.shape[0], -1))

        _base_out = rearrange(flat_out[:NBASE], "(b NBASE) c -> b NBASE c", b=batch_size)
        _target_out = rearrange(flat_out[NBASE:], "(b nTARGET) c -> b nTARGET c", b=batch_size)
        flat_out = torch.cat([_base_out, _target_out], dim=1)  # [B, n_base + n_target, in_dim]

        self._ema_update_teacher()
        with torch.no_grad():
            x_tea = self.projector_tea(flat_out, top_left_xy)
        x_stu = self.projector_stu(self.dropout(flat_out), top_left_xy)

        recon_image = self.decoder(self.encoder(mixed_image))

        return x_tea, x_stu, recon_image
