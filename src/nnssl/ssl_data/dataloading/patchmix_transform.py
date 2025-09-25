from copy import deepcopy
from typing import Tuple

import numpy as np
import torch
import torch.distributed
from batchgenerators.transforms.abstract_transforms import AbstractTransform
from einops import rearrange
from numpy import ndarray


class PatchMixTransform(AbstractTransform):
    def __init__(
        self,
        patch_xy_size: Tuple[int, int],
        num_patch_groups: int,
        mix_across_gpus: bool = True,
        mix_prob: float = 1.0,
        data_key: str = "data",
    ):
        """
        Implementation of https://arxiv.org/pdf/2306.12243

        **Args**
        - `patch_xy_size`: Spatial size of each patch along the first two axes (xy-plane).
        - `num_patch_groups`: PatchMix contains 4 steps:
        (1) Each image is converted into a patch sequence.
        (2) In the sequence, patches are shuffled. The shuffle order is the same for all images in the batch.
        (3) The patch sequence is evenly divided into several patch groups.
        (4) Patches are mixed group-wise.
        - `mix_across_gpus`: If True, mix patches over all GPUs, else only within each GPU.
        - `mix_prob`: Probability of applying PatchMix, synchronized across GPUs.
        If not applied, the original images are returned.
        - `data_key`: Key to access images in the input data dictionary.

        **Notes**
        - Denote `b` the batch size on a single GPU and `b_total` the summed batch size across all GPUs.
        If single GPU, `b` == `b_total`.
        - Denote `B` either `b` or `b_total`.
        - Leave assertions to users.
        - Also, users should ensure that `num_patch_groups` <= (`b_total` if `mix_across_gpus` else `b`) #TODO - Fix this limitation
        """
        self.patch_xy_size = patch_xy_size
        self.num_patch_groups = num_patch_groups
        self.mix_across_gpus = mix_across_gpus
        self.mix_prob = mix_prob
        self.data_key = data_key
        self.distributed = torch.distributed.is_initialized() and torch.distributed.get_world_size() > 1
        self.world_size = torch.distributed.get_world_size() if self.distributed else 1
        self.rank = torch.distributed.get_rank() if self.distributed else 0

    def __call__(self, **data_dict):
        data = data_dict.get(self.data_key)
        if data is None:
            raise ValueError(f"No data found for key {self.data_key}")

        orig_images = deepcopy(data_dict[self.data_key])

        # ==========================================================
        # ==========================================================
        # ==========================================================
        # ==========================================================
        if isinstance(orig_images, ndarray):
            orig_images = torch.from_numpy(orig_images)

        b = orig_images.shape[0]
        b_total = b * self.world_size

        shall_mix = self.num_patch_groups > 1 and np.random.rand() < self.mix_prob

        # Synchronize PatchMix decision across GPUs
        if self.distributed and not self.mix_across_gpus:
            broadcasted_shall_mix = torch.tensor(int(shall_mix)).cuda()
            torch.distributed.broadcast(broadcasted_shall_mix, src=0)
            shall_mix = bool(broadcasted_shall_mix.cpu().item())
            del broadcasted_shall_mix

        # ==========================================================
        #                            Mix!
        # ==========================================================
        if shall_mix:
            # Gather images from all GPUs
            if self.distributed and self.mix_across_gpus:
                gathered_orig_images = [torch.zeros_like(orig_images).cuda() for _ in range(self.world_size)]
                torch.distributed.all_gather(gathered_orig_images, orig_images.cuda())
                orig_images = torch.cat(gathered_orig_images, dim=0).cpu()
                del gathered_orig_images

            B, c, x, y, z = orig_images.shape

            num_patches_x = x // self.patch_xy_size[0]
            num_patches_y = y // self.patch_xy_size[1]
            num_patches = num_patches_x * num_patches_y

            # ============ Step 1: Patchify
            patches = rearrange(
                orig_images,
                "B c (npx x_new) (npy y_new) z -> B (npx npy) c x_new y_new z",
                npx=num_patches_x,
                npy=num_patches_y,
            )

            # ============ Step 2: Generate shuffle indices
            forward_indices = torch.from_numpy(np.random.permutation(num_patches))  # [num_patches]

            # Sort the forward indices by patch group by patch group,
            # not affecting result but adding readability
            forward_indices = forward_indices.view(-1, self.num_patch_groups)
            forward_indices = torch.sort(forward_indices, dim=1)[0]
            forward_indices = forward_indices.view(-1)

            backward_indices = torch.argsort(forward_indices)  # [num_patches]

            # ============ Step 3: Shuffle patches. The shuffle order is the same for all images in the batch.
            shuffled_patches = patches[:, forward_indices]

            # ============ Step 4: Mix patches
            grouped_patches = rearrange(shuffled_patches, "B (npg lpg) ... -> (B npg) lpg ...", npg=self.num_patch_groups)
            NPG = grouped_patches.shape[0]
            mixed_patches = grouped_patches[(torch.arange(NPG) + torch.arange(NPG) % self.num_patch_groups * self.num_patch_groups) % NPG]
            mixed_patches = rearrange(mixed_patches, "(B npg) lpg ... -> B (npg lpg) ...", npg=self.num_patch_groups)

            # ============ Step 5: Unshuffle patches
            unshuffled_patches = mixed_patches[:, backward_indices]

            # ============ Step 6: Unpatchify
            mixed_images = rearrange(
                unshuffled_patches,
                "B (npx npy) c x_new y_new z -> B c (npx x_new) (npy y_new) z",
                npx=num_patches_x,
            ).contiguous()

            # ============ Step 7: Get similarity labels
            base_indices = torch.arange(B).view(-1, 1)
            m2o_indices = (base_indices + torch.arange(start=0, end=self.num_patch_groups)) % B  # [B, num_patch_groups]
            o2m_indices = (base_indices - torch.arange(start=0, end=self.num_patch_groups)) % B  # [B, num_patch_groups]
            m2m_indices = (base_indices + torch.arange(start=-self.num_patch_groups + 1, end=self.num_patch_groups) + B) % B  # [B, 2 * num_patch_groups - 1]

            if self.distributed and not self.mix_across_gpus:
                offset = b * self.rank
                m2o_indices += offset
                o2m_indices += offset
                m2m_indices += offset

            m2o_simi_labels = torch.full((B, b_total), 0.0).scatter_(1, m2o_indices, 1.0 / self.num_patch_groups)  # [B, b_total]
            o2m_simi_labels = torch.full((B, b_total), 0.0).scatter_(1, o2m_indices, 1.0 / self.num_patch_groups)  # [B, b_total]
            m2m_simi_labels = torch.full((B, b_total), 0.0).scatter_(1, m2m_indices, (1.0 - torch.abs(self.num_patch_groups - torch.arange(2 * self.num_patch_groups - 1) - 1) / self.num_patch_groups).expand(B, -1))  # [B, b_total]

            # ============ Step 5: Re-distribute to each GPU if needed
            if self.distributed and self.mix_across_gpus:
                mixed_images = mixed_images.cuda()
                torch.distributed.broadcast(mixed_images, src=0)
                mixed_images = mixed_images.cpu()
                mixed_images = mixed_images[b * self.rank : b * (self.rank + 1)]

                for labels in [m2o_simi_labels, o2m_simi_labels, m2m_simi_labels]:
                    labels = labels.cuda()
                    torch.distributed.broadcast(labels, src=0)
                    labels = labels.cpu()
                    labels = labels[b * self.rank : b * (self.rank + 1)]
        # ==========================================================
        #                         Don't mix!
        # ==========================================================
        else:
            mixed_images = orig_images
            m2o_simi_labels = torch.eye(b, b_total)
            o2m_simi_labels = torch.eye(b, b_total)
            m2m_simi_labels = torch.eye(b, b_total)
            forward_indices = torch.tensor([0])
            backward_indices = torch.tensor([0])
        # ==========================================================
        # ==========================================================
        # ==========================================================
        # ==========================================================

        data_dict["mixed_images"] = mixed_images.float()  # [b, c, x, y, z]
        data_dict["m2o_simi_labels"] = m2o_simi_labels.float()  # [b, b_total]
        data_dict["o2m_simi_labels"] = o2m_simi_labels.float()  # [b, b_total]
        data_dict["m2m_simi_labels"] = m2m_simi_labels.float()  # [b, b_total]
        data_dict["forward_indices"] = forward_indices.float()  # [num_patches]
        data_dict["backward_indices"] = backward_indices.float()  # [num_patches]

        return data_dict
