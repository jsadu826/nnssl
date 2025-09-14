import torch
from nnssl.training.loss.vocoV2_loss import VoCoV2Loss


class VoCoV2NoInterLoss(VoCoV2Loss):

    def __init__(self, pred_weight=1, reg_weight=1):
        super().__init__(
            pred_weight=pred_weight,
            reg_weight=reg_weight,
            inter_weight=0.0, # place-holder
        )

    def forward(
        self,
        base_embeddings_tea: torch.Tensor,
        base_embeddings_stu: torch.Tensor,
        target_embeddings_tea: torch.Tensor, # place-holder
        target_embeddings_stu: torch.Tensor,
        gt_overlaps: torch.Tensor,
    ):
        pred_loss = self.prediction_loss(base_embeddings_tea, target_embeddings_stu, gt_overlaps)
        reg_loss = self.regularization_loss(base_embeddings_stu)
        final_loss = self.pred_weight * pred_loss + self.reg_weight * reg_loss
        return final_loss
