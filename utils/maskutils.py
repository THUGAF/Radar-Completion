from typing import List, Tuple
import random
import torch


def gen_random_blockage_mask(ref: torch.Tensor, azimuth_blockage_range: List = [5, 10], random_seed: int = 0) \
    -> Tuple[torch.Tensor, torch.Tensor, int, int]:
    random.seed(random_seed)
    azimuth_size = ref.size(2)
    blockage_len = random.randint(min(azimuth_blockage_range), max(azimuth_blockage_range))
    anchor = random.randint(0, azimuth_size - blockage_len - 1)
    mask = torch.ones_like(ref).type_as(ref)
    mask[:, 0, anchor: anchor + blockage_len] = 0.0
    masked_tensor = ref * mask
    return masked_tensor, mask, anchor, blockage_len


def gen_fixed_blockage_mask(ref: torch.Tensor, azimuth_start_point: int, anchor: int, blockage_len: int) \
    -> Tuple[torch.Tensor, torch.Tensor, int, int]:
    mask = torch.ones_like(ref).type_as(ref)
    mask[:, 0, anchor - azimuth_start_point: anchor + blockage_len - azimuth_start_point] = 0.0
    masked_tensor = ref * mask
    return masked_tensor, mask, anchor, blockage_len


def crop(x: torch.Tensor, xmin: int, ymin: int, w: int, h: int) -> torch.Tensor:
    return x[:, :, ymin: ymin + h, xmin: xmin + w]
