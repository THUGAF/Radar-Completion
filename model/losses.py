import torch
import torch.nn as nn
import torch.nn.functional as F
import utils.scaler as scalar


def biased_mse_loss(pred: torch.Tensor, truth: torch.Tensor, vmax: float, vmin: float) -> torch.Tensor:
    points = scalar.minmax_norm(torch.tensor([10.0, 20.0, 30.0, 40.0]), vmax, vmin)
    weight = (truth < points[0]) * 1 \
        + (torch.logical_and(truth >= points[0], truth < points[1])) * 2 \
        + (torch.logical_and(truth >= points[1], truth < points[2])) * 5 \
        + (torch.logical_and(truth >= points[2], truth < points[3])) * 10 \
        + (truth >= points[3]) * 30
    return torch.mean(weight * (pred - truth) ** 2)


def biased_mae_loss(pred: torch.Tensor, truth: torch.Tensor, vmax: float, vmin: float) -> torch.Tensor:
    points = scalar.minmax_norm(torch.tensor([10.0, 20.0, 30.0, 40.0]), vmax, vmin)
    weight = (truth < points[0]) * 1 \
        + (torch.logical_and(truth >= points[0], truth < points[1])) * 2 \
        + (torch.logical_and(truth >= points[1], truth < points[2])) * 5 \
        + (torch.logical_and(truth >= points[2], truth < points[3])) * 10 \
        + (truth >= points[3]) * 30
    return torch.mean(weight * torch.abs(pred - truth))