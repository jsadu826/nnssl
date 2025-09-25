if __name__ == "__main__":
    import argparse
    import os
    import shutil

    import matplotlib.pyplot as plt
    import monai.transforms as mtf
    import numpy as np
    import torch
    from monai.data import list_data_collate

    from nnssl.ssl_data.dataloading.patchmix_transform import PatchMixTransform

    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    # -------------------------------------------------
    # Define hyper-parameters
    # srun -p h800 -n 1 --gres=gpu:1 python test_mixing.py -npg 1 -tbs 4
    # srun -p h800 -n 1 --gres=gpu:1 python test_mixing.py -npg 3 -tbs 8
    # srun -p h800 -n 1 --gres=gpu:1 python test_mixing.py -npg 3 -tbs 8 -mag
    # srun -p h800 -n 2 --gres=gpu:2 python test_mixing.py -npg 3 -tbs 8
    # srun -p h800 -n 2 --gres=gpu:2 python test_mixing.py -npg 3 -tbs 8 -mag
    # srun -p h800 -n 4 --gres=gpu:4 python test_mixing.py -npg 2 -tbs 8
    # srun -p h800 -n 4 --gres=gpu:4 python test_mixing.py -npg 8 -tbs 8 -mag
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_patch_groups", "-npg", type=int, default=4)
    parser.add_argument("--mix_across_gpus", "-mag", action="store_true")
    parser.add_argument("--total_batch_size", "-tbs", type=int, default=8)
    args = parser.parse_args()

    FULL_XY_SIZE = (192, 192)

    NUM_PATCH_GROUPS = args.num_patch_groups
    PATCH_XY_SIZE = (FULL_XY_SIZE[0] // NUM_PATCH_GROUPS, FULL_XY_SIZE[1] // NUM_PATCH_GROUPS)
    MIX_ACROSS_GPUS = args.mix_across_gpus
    MIX_PROB = 1.0
    DATA_KEY = "data"

    PATCH_Z_SIZE = 32
    SAVE_DIR = "./test_output"
    WORLD_SIZE = int(os.environ["SLURM_NTASKS"])
    RANK = int(os.environ["SLURM_PROCID"])
    BATCH_SIZE = args.total_batch_size // WORLD_SIZE
    DATA_PATHS = [ent.path for ent in os.scandir("samples")][BATCH_SIZE * RANK : BATCH_SIZE * (RANK + 1)]
    # -------------------------------------------------

    torch.cuda.set_device(f"cuda:{RANK}")
    if WORLD_SIZE > 1:
        torch.distributed.init_process_group(backend="nccl", init_method="tcp://localhost:20000", rank=RANK, world_size=WORLD_SIZE)

    pre_transforms = mtf.Compose(
        [
            mtf.LoadImage(image_only=True),
            mtf.EnsureChannelFirst(),
            mtf.Orientation(axcodes="RAS"),
            mtf.NormalizeIntensity(nonzero=True),
            mtf.CropForeground(allow_smaller=True),
            mtf.Resize(
                mode="trilinear",
                align_corners=True,
                spatial_size=[FULL_XY_SIZE[0], FULL_XY_SIZE[1], PATCH_Z_SIZE],
            ),
            mtf.ToTensor(track_meta=False),
        ]
    )

    patchmix_transform = PatchMixTransform(
        PATCH_XY_SIZE,
        NUM_PATCH_GROUPS,
        MIX_ACROSS_GPUS,
        MIX_PROB,
        DATA_KEY,
    )

    min_max_norm = lambda x: (x - x.min()) / (x.max() - x.min())
    pre_transformed_images = list_data_collate([pre_transforms(data) for data in DATA_PATHS])  # [b, 1, x, y, z]
    pre_transformed_images = min_max_norm(pre_transformed_images)

    # -----------------------------------------------------------------
    # Add a half-transparent random color mask to each grayscale image

    # Gather images from all GPUs if needed
    if torch.distributed.is_initialized() and MIX_ACROSS_GPUS:
        _b = BATCH_SIZE * WORLD_SIZE
        gathered_pre_transformed_images = [torch.zeros_like(pre_transformed_images).cuda() for _ in range(WORLD_SIZE)]
        torch.distributed.all_gather(gathered_pre_transformed_images, pre_transformed_images.cuda())
        pre_transformed_images = torch.cat(gathered_pre_transformed_images, dim=0).cpu()
        del gathered_pre_transformed_images
    else:
        _b = BATCH_SIZE

    _, _, x, y, z = pre_transformed_images.shape
    torch.random.manual_seed(50)
    colors = torch.rand(_b, 3, dtype=pre_transformed_images.dtype)
    colored_images = torch.zeros(_b, 3, x, y, z, dtype=pre_transformed_images.dtype)

    for i in range(_b):
        gray_image = pre_transformed_images[i, 0]  # [x, y, z]
        alpha = 0.5  # Transparency factor
        for c in range(3):
            colored_images[i, c] = gray_image * alpha + colors[i, c] * (1 - alpha)

    # Re-distribute to each GPU if needed
    if torch.distributed.is_initialized() and MIX_ACROSS_GPUS:
        colored_images = colored_images.cuda()
        torch.distributed.broadcast(colored_images, src=0)
        colored_images = colored_images.cpu()
        colored_images = colored_images[BATCH_SIZE * RANK : BATCH_SIZE * (RANK + 1)]
    # -----------------------------------------------------------------

    output = patchmix_transform(**{DATA_KEY: colored_images})
    orig_images = output[DATA_KEY]
    mixed_images = output["mixed_images"]
    m2o_simi_labels = output["m2o_simi_labels"]
    o2m_simi_labels = output["o2m_simi_labels"]
    m2m_simi_labels = output["m2m_simi_labels"]
    forward_indices = output["forward_indices"]
    backward_indices = output["backward_indices"]

    # Sequential print for each rank
    for r in range(WORLD_SIZE):
        if RANK == r:
            print(f"------------------------ rank {r} ------------------------")
            print("orig_images:", list(orig_images.shape))
            print("mixed_images:", list(mixed_images.shape))
            print("m2o_simi_labels:", list(m2o_simi_labels.shape), "\n", m2o_simi_labels)
            print("o2m_simi_labels:", list(o2m_simi_labels.shape), "\n", o2m_simi_labels)
            print("m2m_simi_labels:", list(m2m_simi_labels.shape), "\n", m2m_simi_labels)
            print("forward_indices:", list(forward_indices.shape), "\n", forward_indices)
            print("backward_indices:", list(backward_indices.shape), "\n", backward_indices)
        if torch.distributed.is_initialized():
            torch.distributed.barrier()

    # Visualization
    shutil.rmtree(SAVE_DIR, ignore_errors=True)
    os.makedirs(SAVE_DIR, exist_ok=True)

    orig_images = orig_images.numpy()
    mixed_images = mixed_images.numpy()

    orig_slices = []
    mixed_slices = []
    margin_width = 3

    for i in range(len(DATA_PATHS)):
        input_slice = orig_images[i, :, :, :, int(PATCH_Z_SIZE * 0.7)].transpose(1, 2, 0)  # [c, x, y] -> [x, y, c]
        output_slice = mixed_images[i, :, :, :, int(PATCH_Z_SIZE * 0.7)].transpose(1, 2, 0)  # [c, x, y] -> [x, y, c]
        orig_slices.append(input_slice)
        mixed_slices.append(output_slice)

    def add_margins(slices, margin_width):
        if len(slices) == 0:
            return slices
        h, w, c = slices[0].shape
        margin = np.ones((h, margin_width, c), dtype=slices[0].dtype)
        slices_with_margins = []
        for i, slice_img in enumerate(slices):
            slices_with_margins.append(slice_img)
            if i < len(slices) - 1:
                slices_with_margins.append(margin)
        return slices_with_margins

    # Add margins to both rows
    orig_slices_with_margins = add_margins(orig_slices, margin_width)
    mixed_slices_with_margins = add_margins(mixed_slices, margin_width)

    # Combine images into rows
    orig_row = np.concatenate(orig_slices_with_margins, axis=1)  # Concatenate horizontally
    mixed_row = np.concatenate(mixed_slices_with_margins, axis=1)  # Concatenate horizontally

    # Add horizontal margin between the two rows
    if len(orig_slices) > 0:
        row_margin = np.ones((margin_width, orig_row.shape[1], orig_row.shape[2]), dtype=orig_row.dtype)
        combined_image = np.concatenate([orig_row, row_margin, mixed_row], axis=0)  # Stack vertically with margin
    else:
        combined_image = np.concatenate([orig_row, mixed_row], axis=0)  # Stack vertically

    plt.imsave(os.path.join(SAVE_DIR, f"rank_{RANK}.png"), combined_image)

    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()
