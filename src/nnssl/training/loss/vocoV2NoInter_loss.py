import torch
from torch import nn
import torch.nn.functional as F


class VoCoV2NoInterLoss(nn.Module):
    """
    input must be logits, not probabilities!

    https://arxiv.org/pdf/2410.09890

    https://github.com/Luffy03/Large-Scale-Medical/blob/ccae1410f97d2c1a196abb4ae7ee96a482215ea8/Self-supervised/models/voco_head.py
    """

    def __init__(self, pred_weight=1, reg_weight=1):
        super(VoCoV2NoInterLoss, self).__init__()
        self.pred_weight = pred_weight
        self.reg_weight = reg_weight

    def prediction_loss(self, base_embeddings_tea, target_embeddings_stu, gt_overlaps) -> float:
        # We don't backprop through the base embeddings only the target embeddings.
        base_embeddings_tea_de = base_embeddings_tea.detach()
        pred_similarity = F.cosine_similarity(
            base_embeddings_tea_de[:, None, :, :], target_embeddings_stu[:, :, None, :], dim=-1
        )
        logits = F.relu(pred_similarity)

        # This would have been the code if it wasn't wrongly descibed in the paper...
        # sim_dist = torch.abs(gt_overlaps - logits)
        # N = sim_dist.shape[-1] * sim_dist.shape[-2]
        # l_pred = - torch.sum(torch.log(1 - sim_dist), dim=(1, 2)) / N
        pos_dist = torch.abs(gt_overlaps - logits)
        neg_pos = torch.where(gt_overlaps == 0, torch.ones_like(gt_overlaps), torch.zeros_like(gt_overlaps))
        pos_loss = ((-torch.log(1 - pos_dist + 1e-6)) * gt_overlaps).sum() / (gt_overlaps.sum() + 1e-6)
        neg_loss = ((-torch.log(1 - logits + 1e-6)) * neg_pos).sum() / (neg_pos.sum() + 1e-6)

        l_pred = pos_loss + neg_loss
        return l_pred

    def regularization_loss(self, base_embeddings_stu: torch.Tensor) -> float:
        inter_crop_similarity = F.cosine_similarity(
            base_embeddings_stu[:, None, :, :],
            base_embeddings_stu[:, :, None, :],
            dim=-1,
        )
        inter_crop_sim_relu = F.relu(inter_crop_similarity)

        up_tri = torch.ones(
            inter_crop_sim_relu.shape[-2], inter_crop_sim_relu.shape[-1], device=inter_crop_sim_relu.device
        ).triu(diagonal=1)[None, ...]

        upper_triangular = up_tri * inter_crop_sim_relu
        N = upper_triangular.shape[-1]
        # Aggregate per image cluster then average across batch samples.
        l_reg = torch.mean(torch.sum(upper_triangular ** 2, dim=(-2, -1)) * 2 / (N * (N - 1)))
        return l_reg

    def forward(
        self,
        base_embeddings_tea: torch.Tensor,
        base_embeddings_stu: torch.Tensor,
        target_embeddings_stu: torch.Tensor,
        gt_overlaps: torch.Tensor,
    ):
        pred_loss = self.prediction_loss(base_embeddings_tea, target_embeddings_stu, gt_overlaps)
        reg_loss = self.regularization_loss(base_embeddings_stu)
        final_loss = self.pred_weight * pred_loss + self.reg_weight * reg_loss
        return final_loss
