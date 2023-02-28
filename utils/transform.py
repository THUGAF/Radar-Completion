import torch


def minmax_norm(tensor: torch.Tensor, vmax: float, vmin: float) -> torch.Tensor:
    tensor = torch.clip(tensor, vmin, vmax)
    tensor = ((tensor - vmin) / (vmax - vmin))
    return tensor


def reverse_minmax_norm(tensor: torch.Tensor, vmax: float, vmin: float) -> torch.Tensor:
    tensor = torch.clip(tensor, 0.0, 1.0)
    tensor = tensor * (vmax - vmin) + vmin
    return tensor


def convert_to_gray(tensor: torch.Tensor) -> torch.Tensor:
    tensor = tensor * 255
    tensor = tensor.to(torch.uint8)
    return tensor
