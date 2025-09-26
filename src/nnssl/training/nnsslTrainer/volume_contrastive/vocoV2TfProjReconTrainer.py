import os
from copy import deepcopy
from typing import Tuple, Union

import numpy as np
import torch
from batchgenerators.transforms.abstract_transforms import AbstractTransform, Compose
from batchgenerators.transforms.color_transforms import (
    BrightnessMultiplicativeTransform,
    ContrastAugmentationTransform,
    GammaTransform,
)
from batchgenerators.transforms.noise_transforms import (
    GaussianBlurTransform,
    GaussianNoiseTransform,
)
from batchgenerators.transforms.resample_transforms import (
    SimulateLowResolutionTransform,
)
from batchgenerators.transforms.utility_transforms import NumpyToTensor
from torch import autocast, nn

from nnssl.adaptation_planning.adaptation_plan import AdaptationPlan, ArchitecturePlans
from nnssl.architectures.get_network_by_name import get_network_by_name
from nnssl.architectures.vocoV2TfProjRecon_architecture import (
    VoCoV2TfProjReconArchitecture,
)
from nnssl.experiment_planning.experiment_planners.plan import ConfigurationPlan, Plan
from nnssl.ssl_data.dataloading.patchmix_transform import PatchMixTransform
from nnssl.ssl_data.dataloading.voco_transform import VocoTransform
from nnssl.training.loss.vocoV2Recon_loss import VoCoV2ReconLoss
from nnssl.training.nnsslTrainer.volume_contrastive.vocoV2Trainer import VoCoV2Trainer
from nnssl.utilities.helpers import dummy_context


class VoCoV2TfProjReconTrainer(VoCoV2Trainer):

    def __init__(
        self,
        plan: Plan,
        configuration_name: str,
        fold: int,
        pretrain_json: dict,
        device: torch.device = torch.device("cuda"),
        patch_size: tuple = (256, 256, 64),
        base_crop_count: tuple = (4, 4, 1),
        target_crop_count: int = 4,
    ):
        super().__init__(
            plan,
            configuration_name,
            fold,
            pretrain_json,
            device,
            patch_size,
            base_crop_count,
            target_crop_count,
        )

    def build_loss(self) -> nn.Module:
        return VoCoV2ReconLoss()

    def get_training_transforms(
        self,
        patch_size: Union[np.ndarray, Tuple[int]],
        rotation_for_DA: dict,
        mirror_axes: Tuple[int, ...],
        do_dummy_2d_data_aug: bool,
        order_resampling_data: int = 3,
        order_resampling_seg: int = 1,
        border_val_seg: int = -1,
    ) -> AbstractTransform:
        tr_transforms = []

        if do_dummy_2d_data_aug:
            raise NotImplementedError("We don't do dummy 2d aug here anymore. Data should be isotropic!")

        tr_transforms.append(
            PatchMixTransform(
                patch_xy_size=self.voco_crop_size[:2],
                num_patch_groups=self.voco_base_crop_count[0],
                mix_across_gpus=False,  # DO NOT MIX ACROSS GPUS
                mix_prob=1.0,
                data_key="data",
            )
        )
        # Just enhance the mixed images
        tr_transforms.append(
            Compose(
                [
                    GaussianNoiseTransform(p_per_sample=0.1, data_key="mixed_images"),
                    GaussianBlurTransform(
                        (0.5, 1.0),
                        different_sigma_per_channel=True,
                        p_per_sample=0.2,
                        p_per_channel=0.5,
                        data_key="mixed_images",
                    ),
                    BrightnessMultiplicativeTransform(multiplier_range=(0.75, 1.25), p_per_sample=0.15, data_key="mixed_images"),
                    ContrastAugmentationTransform(p_per_sample=0.15),
                    SimulateLowResolutionTransform(
                        zoom_range=(0.5, 1),
                        per_channel=True,
                        p_per_channel=0.5,
                        order_downsample=0,
                        order_upsample=3,
                        p_per_sample=0.1,
                        ignore_axes=None,
                        data_key="mixed_images",
                    ),
                    GammaTransform((0.7, 1.5), True, True, retain_stats=True, p_per_sample=0.1, data_key="mixed_images"),
                    GammaTransform((0.7, 1.5), False, True, retain_stats=True, p_per_sample=0.3, data_key="mixed_images"),
                ]
            )
        )
        tr_transforms.append(
            VocoTransform(
                voco_base_crop_count=self.voco_base_crop_count,
                voco_crop_size=self.voco_crop_size,
                aug="train",
                voco_target_crop_count=self.voco_target_crop_count,
                data_key="data",
            )
        )
        tr_transforms.append(
            NumpyToTensor(
                [
                    "all_crops",
                    "base_target_crop_overlaps",
                    "all_crops_top_left_xy",
                    "data",
                    "mixed_images",
                    "m2o_simi_labels",
                    "o2m_simi_labels",
                    "m2m_simi_labels",
                    "forward_indices",
                    "backward_indices",
                ],
                "float",
            )
        )
        tr_transforms = Compose(tr_transforms)
        return tr_transforms

    def get_validation_transforms(self) -> AbstractTransform:
        val_transforms = []

        # --------------------------- VoCo Transformation --------------------------- #
        val_transforms.append(
            PatchMixTransform(
                patch_xy_size=self.voco_crop_size[:2],
                num_patch_groups=self.voco_base_crop_count[0],
                mix_across_gpus=False,
                mix_prob=1.0,
                data_key="data",
            )
        )
        val_transforms.append(
            VocoTransform(
                voco_base_crop_count=self.voco_base_crop_count,
                voco_crop_size=self.voco_crop_size,
                aug="none",
                voco_target_crop_count=self.voco_target_crop_count,
                data_key="data",
            )
        )
        val_transforms.append(
            NumpyToTensor(
                [
                    "all_crops",
                    "base_target_crop_overlaps",
                    "all_crops_top_left_xy",
                    "data",
                    "mixed_images",
                    "m2o_simi_labels",
                    "o2m_simi_labels",
                    "m2m_simi_labels",
                    "forward_indices",
                    "backward_indices",
                ],
                "float",
            )
        )
        val_transforms = Compose(val_transforms)
        return val_transforms

    def build_architecture_and_adaptation_plan(self, config_plan: ConfigurationPlan, num_input_channels: int, num_output_channels: int) -> nn.Module:
        network = get_network_by_name(
            config_plan,
            "ResEncL",
            num_input_channels,
            num_output_channels,
            encoder_only=False,  # WE NEED THE DECODER AS WELL !!!
        )
        encoder = network.encoder
        decoder = network.decoder
        architecture = VoCoV2TfProjReconArchitecture(encoder, decoder, encoder.output_channels)

        # We need to set the patch size to the one the model saw during training
        plan = deepcopy(self.plan)
        plan.configurations[self.configuration_name].patch_size = self.voco_crop_size

        adapt_plan = AdaptationPlan(
            architecture_plans=ArchitecturePlans("ResEncL"),
            pretrain_plan=plan,
            recommended_downstream_patchsize=self.recommended_downstream_patchsize,
            pretrain_num_input_channels=num_input_channels,
            key_to_encoder="encoder.stages",
            key_to_stem="encoder.stem",
            keys_to_in_proj=("encoder.stem.convs.0.conv", "encoder.stem.convs.0.all_modules.0"),
        )
        return architecture, adapt_plan

    def train_step(self, batch: dict) -> dict:
        all_crops = batch["all_crops"]
        NBASE = batch["base_crop_index"]
        gt_overlaps = batch["base_target_crop_overlaps"]
        all_crops_top_left_xy = batch["all_crops_top_left_xy"]  # [B, n_base + n_target, 2]
        orig_image = batch["data"]
        mixed_image = batch["mixed_images"]

        # ====================== Save as PNG ======================
        # FIXME - Un-comment to debug
        # # Save middle slice of original and mixed images as PNG
        # import matplotlib.pyplot as plt

        # # Create directory for saving images if it doesn't exist
        # save_dir = "/mnt/nfs_share/zheyu/official_nnssl/tmp"
        # os.makedirs(save_dir, exist_ok=True)

        # # Get middle slice index
        # middle_slice = orig_image.shape[-1] // 2

        # # Convert to numpy and get middle slice
        # orig_slice = orig_image[0, 0, :, :, middle_slice].detach().cpu().numpy()
        # mixed_slice = mixed_image[0, 0, :, :, middle_slice].detach().cpu().numpy()

        # plt.figure(figsize=(12, 5))
        # plt.subplot(1, 2, 1)
        # plt.imshow(orig_slice, cmap="gray")
        # plt.title("Original Image - Middle Slice")
        # plt.axis("off")

        # plt.subplot(1, 2, 2)
        # plt.imshow(mixed_slice, cmap="gray")
        # plt.title("Mixed Image - Middle Slice")
        # plt.axis("off")

        # plt.tight_layout()
        # plt.savefig(f"{save_dir}/batch_{self.current_epoch if hasattr(self, 'current_epoch') else 0}_orig_mixed_slices.png", dpi=150, bbox_inches="tight")
        # plt.close()
        # =========================================================

        all_crops = all_crops.to(self.device, non_blocking=True)
        gt_overlaps = gt_overlaps.to(self.device, non_blocking=True)
        all_crops_top_left_xy = all_crops_top_left_xy.to(self.device, non_blocking=True)
        orig_image = orig_image.to(self.device, non_blocking=True)
        mixed_image = mixed_image.to(self.device, non_blocking=True)

        span_x = self.voco_crop_size[0] * (self.voco_base_crop_count[0] - 1)
        span_y = self.voco_crop_size[1] * (self.voco_base_crop_count[1] - 1)
        all_crops_top_left_xy /= torch.tensor([span_x, span_y], dtype=torch.float, device=self.device)  # normalize to the range [0, 1]

        self.optimizer.zero_grad(set_to_none=True)
        # Autocast is a little bitch.
        # If the device_type is 'cpu' then it's slow as heck and needs to be disabled.
        # If the device_type is 'mps' then it will complain that mps is not implemented, even if enabled=False is set. Whyyyyyyy. (this is why we don't make use of enabled=False)
        # So autocast will only be active if we have a cuda device.
        with autocast(self.device.type, enabled=True) if self.device.type == "cuda" else dummy_context():
            embeddings_tea, embeddings_stu, recon_image = self.network(
                all_crops,
                mixed_image,
                all_crops_top_left_xy,
                NBASE,
                self.batch_size,
            )  # [B, n_base + n_target, out_dim]
            base_embeddings_tea = embeddings_tea[:, : NBASE // self.batch_size, :]  # [B, n_base, out_dim]
            target_embeddings_tea = embeddings_tea[:, NBASE // self.batch_size :, :]  # [B, n_target, out_dim]
            base_embeddings_stu = embeddings_stu[:, : NBASE // self.batch_size, :]  # [B, n_base, out_dim]
            target_embeddings_stu = embeddings_stu[:, NBASE // self.batch_size :, :]  # [B, n_target, out_dim]

            l = self.loss(base_embeddings_tea, base_embeddings_stu, target_embeddings_tea, target_embeddings_stu, gt_overlaps, recon_image, orig_image)

        if self.grad_scaler is not None:
            self.grad_scaler.scale(l).backward()
            self.grad_scaler.unscale_(self.optimizer)
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
        else:
            l.backward()
            self.optimizer.step()
        return {"loss": l.detach().cpu().numpy()}

    def validation_step(self, batch: dict) -> dict:
        return {"loss": np.array([0.0])} # FIXME - Currently we don't do validation during pretraining for efficiency
        all_crops = batch["all_crops"]
        NBASE = batch["base_crop_index"]
        gt_overlaps = batch["base_target_crop_overlaps"]
        all_crops_top_left_xy = batch["all_crops_top_left_xy"]  # [B, n_base + n_target, 2]
        orig_image = batch["data"]
        mixed_image = batch["mixed_images"]

        all_crops = all_crops.to(self.device, non_blocking=True)
        gt_overlaps = gt_overlaps.to(self.device, non_blocking=True)
        all_crops_top_left_xy = all_crops_top_left_xy.to(self.device, non_blocking=True)
        orig_image = orig_image.to(self.device, non_blocking=True)
        mixed_image = mixed_image.to(self.device, non_blocking=True)

        span_x = self.voco_crop_size[0] * (self.voco_base_crop_count[0] - 1)
        span_y = self.voco_crop_size[1] * (self.voco_base_crop_count[1] - 1)
        all_crops_top_left_xy /= torch.tensor([span_x, span_y], dtype=torch.float, device=self.device)  # normalize to the range [0, 1]

        # Autocast is a little bitch.
        # If the device_type is 'cpu' then it's slow as heck and needs to be disabled.
        # If the device_type is 'mps' then it will complain that mps is not implemented, even if enabled=False is set. Whyyyyyyy. (this is why we don't make use of enabled=False)
        # So autocast will only be active if we have a cuda device.
        with torch.no_grad():
            with autocast(self.device.type, enabled=True) if self.device.type == "cuda" else dummy_context():
                embeddings_tea, embeddings_stu, recon_image = self.network(
                    all_crops,
                    mixed_image,
                    all_crops_top_left_xy,
                    NBASE,
                    self.batch_size,
                )  # [B, n_base + n_target, out_dim]
                base_embeddings_tea = embeddings_tea[:, : NBASE // self.batch_size, :]  # [B, n_base, out_dim]
                target_embeddings_tea = embeddings_tea[:, NBASE // self.batch_size :, :]  # [B, n_target, out_dim]
                base_embeddings_stu = embeddings_stu[:, : NBASE // self.batch_size, :]  # [B, n_base, out_dim]
                target_embeddings_stu = embeddings_stu[:, NBASE // self.batch_size :, :]  # [B, n_target, out_dim]

                l = self.loss(base_embeddings_tea, base_embeddings_stu, target_embeddings_tea, target_embeddings_stu, gt_overlaps, recon_image, orig_image)

        return {"loss": l.detach().cpu().numpy()}


####################################################################
############################# VARIANTS #############################
####################################################################


class VoCoV2TfProjReconTrainer_BS8_lr_2e5(VoCoV2TfProjReconTrainer):
    def __init__(
        self,
        plan: Plan,
        configuration_name: str,
        fold: int,
        pretrain_json: dict,
        device: torch.device = torch.device("cuda"),
    ):
        super().__init__(plan, configuration_name, fold, pretrain_json, device)
        self.total_batch_size = 8
        self.initial_lr = 2e-5
