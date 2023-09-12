from typing import Tuple
import torch


def get_sector_weights(radial_size: int, azimuthal_size: int):
    sector_weights = []
    for r in range(radial_size):
        weight = ((r + 1) ** 2 - r ** 2) / (radial_size ** 2 * azimuthal_size)
        sector_weight = torch.Tensor([weight]).repeat(azimuthal_size)
        sector_weights.append(sector_weight)
    sector_weights = torch.cat(sector_weights)
    return sector_weights


def evaluate_wmae(pred: torch.Tensor, truth: torch.Tensor, mask: torch.Tensor) -> float:
    assert pred.size() == truth.size()
    pred, truth = pred.cpu(), truth.cpu()
    pred = torch.clip(pred, min=0, max=70)
    truth = torch.clip(truth, min=0, max=70)
    mask = torch.logical_not(mask.bool())
    radial_size = pred.size(3)
    azimuthal_size = pred[mask].size(0) // radial_size
    sector_weights = get_sector_weights(radial_size, azimuthal_size)
    wmae = torch.sum(torch.abs(pred[mask] - truth[mask]) * sector_weights).item()
    return wmae


def evaluate_wrmse(pred: torch.Tensor, truth: torch.Tensor, mask: torch.Tensor) -> float:
    assert pred.size() == truth.size()
    pred, truth = pred.cpu(), truth.cpu()
    pred = torch.clip(pred, min=0, max=70)
    truth = torch.clip(truth, min=0, max=70)
    mask = torch.logical_not(mask.bool())
    radial_size = pred.size(3)
    azimuthal_size = pred[mask].size(0) // radial_size
    sector_weights = get_sector_weights(radial_size, azimuthal_size)
    wrmse = torch.sqrt(torch.sum(torch.square(pred[mask] - truth[mask]) * sector_weights)).item()
    return wrmse


def evaluate_wmbe(pred: torch.Tensor, truth: torch.Tensor, mask: torch.Tensor) -> float:
    assert pred.size() == truth.size()
    pred, truth = pred.cpu(), truth.cpu()
    pred = torch.clip(pred, min=0, max=70)
    truth = torch.clip(truth, min=0, max=70)
    mask = torch.logical_not(mask.bool())
    radial_size = pred.size(3)
    azimuthal_size = pred[mask].size(0) // radial_size
    sector_weights = get_sector_weights(radial_size, azimuthal_size)
    wmbe = torch.sum((pred[mask] - truth[mask]) * sector_weights).item()
    return wmbe


def evaluate_wmae_multi_thresholds(pred: torch.Tensor, truth: torch.Tensor, mask: torch.Tensor, 
                                   thresholds: list = [0, 10, 20, 30, 40]) -> Tuple[list, list]:
    assert pred.size() == truth.size()
    pred, truth = pred.cpu(), truth.cpu()
    pred = torch.clip(pred, min=0, max=70)
    truth = torch.clip(truth, min=0, max=70)
    mask = torch.logical_not(mask.bool())
    radial_size = pred.size(3)
    azimuthal_size = pred[mask].size(0) // radial_size
    sector_weights = get_sector_weights(radial_size, azimuthal_size)
    wmaes = []
    for i, threshold in enumerate(thresholds):
        if i < len(thresholds) - 1:
            loc = torch.logical_and(truth[mask] >= threshold, truth[mask] < thresholds[i + 1])
        else:
            loc = truth[mask] >= threshold
        if torch.sum(loc) == 0:
            wmae = 0
        else:
            pred_flatten, truth_flatten = pred[mask][loc], truth[mask][loc]
            sector_weights_thr = sector_weights[loc] / torch.sum(sector_weights[loc])
            wmae = torch.sum(torch.abs(pred_flatten - truth_flatten) * sector_weights_thr)
            wmae = torch.nan_to_num(wmae).item()
        wmaes.append(wmae)
    return thresholds, wmaes


def evaluate_wrmse_multi_thresholds(pred: torch.Tensor, truth: torch.Tensor, mask: torch.Tensor, 
                                   thresholds: list = [0, 10, 20, 30, 40]) -> Tuple[list, list]:
    assert pred.size() == truth.size()
    pred, truth = pred.cpu(), truth.cpu()
    pred = torch.clip(pred, min=0, max=70)
    truth = torch.clip(truth, min=0, max=70)
    mask = torch.logical_not(mask.bool())
    radial_size = pred.size(3)
    azimuthal_size = pred[mask].size(0) // radial_size
    sector_weights = get_sector_weights(radial_size, azimuthal_size)
    wrmses = []
    for i, threshold in enumerate(thresholds):
        if i < len(thresholds) - 1:
            loc = torch.logical_and(truth[mask] >= threshold, truth[mask] < thresholds[i + 1])
        else:
            loc = truth[mask] >= threshold
        if torch.sum(loc) == 0:
            wrmse = 0
        else:
            pred_flatten, truth_flatten = pred[mask][loc], truth[mask][loc]
            sector_weights_thr = sector_weights[loc] / torch.sum(sector_weights[loc])
            wrmse = torch.sqrt(torch.sum(torch.square(pred_flatten - truth_flatten) * sector_weights_thr))
            wrmse = torch.nan_to_num(wrmse).item()
        wrmses.append(wrmse)
    return thresholds, wrmses


def evaluate_wmbe_multi_thresholds(pred: torch.Tensor, truth: torch.Tensor, mask: torch.Tensor,
                                  thresholds: list = [0, 10, 20, 30, 40]) -> Tuple[list, list]:
    assert pred.size() == truth.size()
    pred, truth = pred.cpu(), truth.cpu()
    pred = torch.clip(pred, min=0, max=70)
    truth = torch.clip(truth, min=0, max=70)
    mask = torch.logical_not(mask.bool())
    radial_size = pred.size(3)
    azimuthal_size = pred[mask].size(0) // radial_size
    sector_weights = get_sector_weights(radial_size, azimuthal_size)
    wmbes = []
    for i, threshold in enumerate(thresholds):
        if i < len(thresholds) - 1:
            loc = torch.logical_and(truth[mask] >= threshold, truth[mask] < thresholds[i + 1])
        else:
            loc = truth[mask] >= threshold
        if torch.sum(loc) == 0:
            wmbe = 0
        else:
            pred_flatten, truth_flatten = pred[mask][loc], truth[mask][loc]
            sector_weights_thr = sector_weights[loc] / torch.sum(sector_weights[loc])
            wmbe = torch.sum((pred_flatten - truth_flatten) * sector_weights_thr)
            wmbe = torch.nan_to_num(wmbe).item()
        wmbes.append(wmbe)
    return thresholds, wmbes
