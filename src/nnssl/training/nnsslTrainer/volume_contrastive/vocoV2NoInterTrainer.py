from copy import deepcopy
from typing import Union, Tuple, override

import numpy as np
import torch
from torch import nn
from torch.optim.adamw import AdamW
from batchgenerators.dataloading.single_threaded_augmenter import SingleThreadedAugmenter
from batchgenerators.transforms.abstract_transforms import AbstractTransform, Compose
from batchgenerators.transforms.utility_transforms import NumpyToTensor

from torch import autocast
from nnssl.adaptation_planning.adaptation_plan import AdaptationPlan, ArchitecturePlans
from nnssl.architectures.get_network_by_name import get_network_by_name
from nnssl.architectures.vocoV2NoInter_architecture import VoCoV2NoInterArchitecture
from nnssl.training.loss.vocoV2NoInter_loss import VoCoV2NoInterLoss
from nnssl.training.nnsslTrainer.volume_contrastive.vocoTrainer import VoCoTrainer
from nnssl.utilities.helpers import dummy_context
from batchgenerators.utilities.file_and_folder_operations import save_json


from einops import rearrange


from nnssl.experiment_planning.experiment_planners.plan import ConfigurationPlan, Plan
from nnssl.ssl_data.configure_basic_dummyDA import configure_rotation_dummyDA_mirroring_and_inital_patch_size
from nnssl.ssl_data.dataloading.voco_transform import VocoTransform
from nnssl.ssl_data.limited_len_wrapper import LimitedLenWrapper


from nnssl.training.nnsslTrainer.AbstractTrainer import AbstractBaseTrainer

from nnssl.utilities.default_n_proc_DA import get_allowed_n_proc_DA


class VoCoV2NoInterTrainer(VoCoTrainer):

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
        return VoCoV2NoInterLoss(pred_weight=self.pred_loss_weight, reg_weight=self.reg_loss_weight)

    def build_architecture_and_adaptation_plan(
        self, config_plan: ConfigurationPlan, num_input_channels: int, num_output_channels: int
    ) -> nn.Module:
        encoder = get_network_by_name(
            config_plan,
            "ResEncL",
            num_input_channels,
            num_output_channels,
            encoder_only=True,
        )
        architecture = VoCoV2NoInterArchitecture(encoder, encoder.output_channels)

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

        all_crops = all_crops.to(self.device, non_blocking=True)
        gt_overlaps = gt_overlaps.to(self.device, non_blocking=True)

        self.optimizer.zero_grad(set_to_none=True)
        # Autocast is a little bitch.
        # If the device_type is 'cpu' then it's slow as heck and needs to be disabled.
        # If the device_type is 'mps' then it will complain that mps is not implemented, even if enabled=False is set. Whyyyyyyy. (this is why we don't make use of enabled=False)
        # So autocast will only be active if we have a cuda device.
        with autocast(self.device.type, enabled=True) if self.device.type == "cuda" else dummy_context():
            embeddings_tea, embeddings_stu = self.network(all_crops)
            base_embeddings_tea = rearrange(embeddings_tea[:NBASE], "(b NBASE) c -> b NBASE c", b=self.batch_size)
            # target_embeddings_tea = rearrange(embeddings_tea[NBASE:], "(b nTARGET) c -> b nTARGET c", b=self.batch_size)
            base_embeddings_stu = rearrange(embeddings_stu[:NBASE], "(b NBASE) c -> b NBASE c", b=self.batch_size)
            target_embeddings_stu = rearrange(embeddings_stu[NBASE:], "(b nTARGET) c -> b nTARGET c", b=self.batch_size)

            # del data
            l = self.loss(base_embeddings_tea, base_embeddings_stu, target_embeddings_stu, gt_overlaps)

        if self.grad_scaler is not None:
            self.grad_scaler.scale(l).backward()
            self.grad_scaler.unscale_(self.optimizer)
            # torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
        else:
            l.backward()
            # torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.optimizer.step()
        return {"loss": l.detach().cpu().numpy()}

    def validation_step(self, batch: dict) -> dict:
        all_crops = batch["all_crops"]
        NBASE = batch["base_crop_index"]
        gt_overlaps = batch["base_target_crop_overlaps"]

        all_crops = all_crops.to(self.device, non_blocking=True)
        gt_overlaps = gt_overlaps.to(self.device, non_blocking=True)

        # Autocast is a little bitch.
        # If the device_type is 'cpu' then it's slow as heck and needs to be disabled.
        # If the device_type is 'mps' then it will complain that mps is not implemented, even if enabled=False is set. Whyyyyyyy. (this is why we don't make use of enabled=False)
        # So autocast will only be active if we have a cuda device.
        with torch.no_grad():
            with autocast(self.device.type, enabled=True) if self.device.type == "cuda" else dummy_context():
                embeddings_tea, embeddings_stu = self.network(all_crops)
                base_embeddings_tea = rearrange(embeddings_tea[:NBASE], "(b NBASE) c -> b NBASE c", b=self.batch_size)
                # target_embeddings_tea = rearrange(embeddings_tea[NBASE:], "(b nTARGET) c -> b nTARGET c", b=self.batch_size)
                base_embeddings_stu = rearrange(embeddings_stu[:NBASE], "(b NBASE) c -> b NBASE c", b=self.batch_size)
                target_embeddings_stu = rearrange(embeddings_stu[NBASE:], "(b nTARGET) c -> b nTARGET c", b=self.batch_size)

                # del data
                l = self.loss(base_embeddings_tea, base_embeddings_stu, target_embeddings_stu, gt_overlaps)

        return {"loss": l.detach().cpu().numpy()}


####################################################################
############################# VARIANTS #############################
####################################################################


class VoCoV2NoInterTrainer_test(VoCoV2NoInterTrainer):
    def __init__(
        self,
        plan: Plan,
        configuration_name: str,
        fold: int,
        pretrain_json: dict,
        device: torch.device = torch.device("cuda"),
    ):
        super().__init__(
            plan,
            configuration_name,
            fold,
            pretrain_json,
            device,
            patch_size=(128, 128, 64),
            base_crop_count=(2, 2, 1),
        )
        self.total_batch_size = 1


############################# LEARNING RATE #############################


class VoCoV2NoInterTrainer_BS8_lr_1e2(VoCoV2NoInterTrainer):
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
        self.initial_lr = 1e-2


class VoCoV2NoInterTrainer_BS8_lr_1e3(VoCoV2NoInterTrainer):
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
        self.initial_lr = 1e-3


class VoCoV2NoInterTrainer_BS8_lr_1e4(VoCoV2NoInterTrainer):
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
        self.initial_lr = 1e-4


############################# WEIGHT DECAY #############################


class VoCoV2NoInterTrainer_BS8_lr_1e2_wd_3e4(VoCoV2NoInterTrainer):
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
        self.initial_lr = 1e-2
        self.weight_decay = 3e-4


class VoCoV2NoInterTrainer_BS8_lr_1e2_wd_3e6(VoCoV2NoInterTrainer):
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
        self.initial_lr = 1e-2
        self.weight_decay = 3e-6


class VoCoV2NoInterTrainer_BS8_lr_1e2_wd_3e2(VoCoV2NoInterTrainer):
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
        self.initial_lr = 1e-2
        self.weight_decay = 3e-2


############################# BASES & PATCH SIZE #############################


class VoCoV2NoInterTrainer_BS8_lr_1e2_wd_3e5_2x2x1_PS96(VoCoV2NoInterTrainer):
    def __init__(
        self,
        plan: Plan,
        configuration_name: str,
        fold: int,
        pretrain_json: dict,
        device: torch.device = torch.device("cuda"),
    ):
        super().__init__(
            plan,
            configuration_name,
            fold,
            pretrain_json,
            device,
            patch_size=(192, 192, 96),
            base_crop_count=(2, 2, 1),
        )
        self.total_batch_size = 8


class VoCoV2NoInterTrainer_BS8_lr_1e2_wd_3e5_2x2x2_PS96(VoCoV2NoInterTrainer):
    def __init__(
        self,
        plan: Plan,
        configuration_name: str,
        fold: int,
        pretrain_json: dict,
        device: torch.device = torch.device("cuda"),
    ):
        super().__init__(
            plan,
            configuration_name,
            fold,
            pretrain_json,
            device,
            patch_size=(192, 192, 192),
            base_crop_count=(2, 2, 2),
        )
        self.total_batch_size = 8


class VoCoV2NoInterTrainer_BS8_lr_1e2_wd_3e5_3x3x1_PS64(VoCoV2NoInterTrainer):
    def __init__(
        self,
        plan: Plan,
        configuration_name: str,
        fold: int,
        pretrain_json: dict,
        device: torch.device = torch.device("cuda"),
    ):
        super().__init__(
            plan,
            configuration_name,
            fold,
            pretrain_json,
            device,
            patch_size=(192, 192, 64),
            base_crop_count=(3, 3, 1),
        )
        self.total_batch_size = 8


class VoCoV2NoInterTrainer_BS8_lr_1e2_wd_3e5_3x3x2_PS64(VoCoV2NoInterTrainer):
    def __init__(
        self,
        plan: Plan,
        configuration_name: str,
        fold: int,
        pretrain_json: dict,
        device: torch.device = torch.device("cuda"),
    ):
        super().__init__(
            plan,
            configuration_name,
            fold,
            pretrain_json,
            device,
            patch_size=(192, 192, 128),
            base_crop_count=(3, 3, 2),
        )
        self.total_batch_size = 8


class VoCoV2NoInterTrainer_BS8_lr_1e2_wd_3e5_4x4x2_PS64(VoCoV2NoInterTrainer):
    def __init__(
        self,
        plan: Plan,
        configuration_name: str,
        fold: int,
        pretrain_json: dict,
        device: torch.device = torch.device("cuda"),
    ):
        super().__init__(
            plan,
            configuration_name,
            fold,
            pretrain_json,
            device,
            patch_size=(256, 256, 128),
            base_crop_count=(4, 4, 2),
        )
        self.total_batch_size = 8


############################# NUMBER OF TARGET CROPS #############################


class VoCoV2NoInterTrainer_BS8_lr_1e2_wd_3e5_4x4x1_PS64_N2(VoCoV2NoInterTrainer):
    def __init__(
        self,
        plan: Plan,
        configuration_name: str,
        fold: int,
        pretrain_json: dict,
        device: torch.device = torch.device("cuda"),
    ):
        super().__init__(
            plan,
            configuration_name,
            fold,
            pretrain_json,
            device,
            patch_size=(256, 256, 64),
            base_crop_count=(4, 4, 1),
            target_crop_count=2,
        )
        self.total_batch_size = 8


class VoCoV2NoInterTrainer_BS8_lr_1e2_wd_3e5_4x4x1_PS64_N8(VoCoV2NoInterTrainer):
    def __init__(
        self,
        plan: Plan,
        configuration_name: str,
        fold: int,
        pretrain_json: dict,
        device: torch.device = torch.device("cuda"),
    ):
        super().__init__(
            plan,
            configuration_name,
            fold,
            pretrain_json,
            device,
            patch_size=(256, 256, 64),
            base_crop_count=(4, 4, 1),
            target_crop_count=8,
        )
        self.total_batch_size = 8
