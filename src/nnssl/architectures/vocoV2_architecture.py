import torch
from nnssl.architectures.voco_architecture import VocoProjectionHead
from torch import nn


class VoCoV2Architecture(nn.Module):
    def __init__(self, encoder: nn.Module,  features: list[int]):
        super(VoCoV2Architecture, self).__init__()
        self.encoder = encoder
        self.adaptive_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.dropout = nn.Dropout1d(p=0.2, inplace=False)

        total_features = sum(features)
        self.projector_tea = VocoProjectionHead(total_features, 2048, 2048, norm_op=nn.InstanceNorm1d)
        self.projector_stu = VocoProjectionHead(total_features, 2048, 2048, norm_op=nn.InstanceNorm1d)
        for param_tea, param_stu in zip(self.projector_tea.parameters(), self.projector_stu.parameters()):
            param_tea.data.copy_(param_stu.data)

    @torch.no_grad()
    def _ema_update_teacher(self):
        momentum = 0.9
        for param_tea, param_stu in zip(self.projector_tea.parameters(), self.projector_stu.parameters()):
            param_tea.data = momentum * param_tea.data + (1.0 - momentum) * param_stu.data

    def forward(self, x):
        out = self.encoder(x)
        flat_out = torch.concat([self.adaptive_pool(o) for o in out], dim=1)
        flat_out = torch.reshape(flat_out, (flat_out.shape[0], -1))
        self._ema_update_teacher()
        with torch.no_grad():
            x_tea = self.projector_tea(flat_out)
        x_stu = self.projector_stu(self.dropout(flat_out))
        return x_tea, x_stu


class VoCoV2EvaArchitecture(nn.Module):
    """
    We don't have multiple CNN stages that we can take the features from and concatenate them, so for the transformer
    we only use the features from the (last) output layer.
    """
    def __init__(self, encoder: nn.Module, embed_dim: int):
        super(VoCoV2EvaArchitecture, self).__init__()
        self.encoder = encoder
        self.adaptive_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.dropout = nn.Dropout1d(0.2, inplace=False)

        self.projector_tea = VocoProjectionHead(embed_dim, 2048, 2048, norm_op=nn.InstanceNorm1d)
        self.projector_stu = VocoProjectionHead(embed_dim, 2048, 2048, norm_op=nn.InstanceNorm1d)
        for param_tea, param_stu in zip(self.projector_tea.parameters(), self.projector_stu.parameters()):
            param_tea.data.copy_(param_stu.data)

    @torch.no_grad()
    def _ema_update_teacher(self):
        momentum = 0.9
        for param_tea, param_stu in zip(self.projector_tea.parameters(), self.projector_stu.parameters()):
            param_tea.data = momentum * param_tea.data + (1.0 - momentum) * param_stu.data

    def forward(self, x):
        out = self.encoder(x)
        flat_out = self.adaptive_pool(out)
        flat_out = torch.reshape(flat_out, (flat_out.shape[0], -1))
        self._ema_update_teacher()
        with torch.no_grad():
            x_tea = self.projector_tea(flat_out)
        x_stu = self.projector_stu(self.dropout(flat_out))
        return x_tea, x_stu
