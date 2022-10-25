import random
from typing import List
import numpy as np
import torch


def gen_blockage_mask(tensor: torch.Tensor, azimuth_blockage_range: List = [5, 10], random_seed: int = 0):
    random.seed(random_seed)
    azimuth_size = tensor.size(2)
    blockage_len = random.randint(min(azimuth_blockage_range), max(azimuth_blockage_range))
    anchor = random.randint(0, azimuth_size - blockage_len - 1)
    mask = torch.ones_like(tensor).type_as(tensor).to(tensor.device)
    mask[:, 0, anchor: anchor + blockage_len] = 0.0
    masked_tensor = tensor * mask
    return masked_tensor, mask, anchor, blockage_len
    