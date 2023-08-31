from typing import Tuple
import torch
import torch.nn.functional as F


def evaluate_mae(pred: torch.Tensor, truth: torch.Tensor, mask: torch.Tensor) -> float:
    assert pred.size() == truth.size()
    pred, truth = pred.cpu(), truth.cpu()
    pred = torch.clip(pred, min=0, max=70)
    truth = torch.clip(truth, min=0, max=70)
    mask = torch.logical_not(mask.bool())
    mae = F.l1_loss(pred[mask], truth[mask]).item()
    return mae


def evaluate_rmse(pred: torch.Tensor, truth: torch.Tensor, mask: torch.Tensor) -> float:
    assert pred.size() == truth.size()
    pred, truth = pred.cpu(), truth.cpu()
    pred = torch.clip(pred, min=0, max=70)
    truth = torch.clip(truth, min=0, max=70)
    mask = torch.logical_not(mask.bool())
    rmse = torch.sqrt(F.mse_loss(pred[mask], truth[mask])).item()
    return rmse


def evaluate_mbe(pred: torch.Tensor, truth: torch.Tensor, mask: torch.Tensor) -> float:
    assert pred.size() == truth.size()
    pred, truth = pred.cpu(), truth.cpu()
    pred = torch.clip(pred, min=0, max=70)
    truth = torch.clip(truth, min=0, max=70)
    mask = torch.logical_not(mask.bool())
    mbe = torch.mean(pred[mask] - truth[mask]).item()
    return mbe


def evaluate_mae_multi_thresholds(pred: torch.Tensor, truth: torch.Tensor, mask: torch.Tensor, 
                                  thresholds: list = [0, 10, 20, 30, 40]) -> Tuple[list, list]:
    assert pred.size() == truth.size()
    pred, truth = pred.cpu(), truth.cpu()
    pred = torch.clip(pred, min=0, max=70)
    truth = torch.clip(truth, min=0, max=70)
    mask = torch.logical_not(mask.bool())
    maes = []
    for i, threshold in enumerate(thresholds):
        if i < len(thresholds) - 1:
            loc = torch.logical_and(truth[mask] >= threshold, truth[mask] < thresholds[i + 1])
        else:
            loc = truth[mask] >= threshold
        pred_flatten, truth_flatten = pred[mask][loc], truth[mask][loc]
        if len(pred_flatten) == 0:
            mae = 0
        else:
            mae = F.l1_loss(pred_flatten, truth_flatten)
            mae = torch.nan_to_num(mae).item()
        maes.append(mae)
    return thresholds, maes


def evaluate_rmse_multi_thresholds(pred: torch.Tensor, truth: torch.Tensor, mask: torch.Tensor, 
                                   thresholds: list = [0, 10, 20, 30, 40]) -> Tuple[list, list]:
    assert pred.size() == truth.size()
    pred, truth = pred.cpu(), truth.cpu()
    pred = torch.clip(pred, min=0, max=70)
    truth = torch.clip(truth, min=0, max=70)
    mask = torch.logical_not(mask.bool())
    rmses = []
    for i, threshold in enumerate(thresholds):
        if i < len(thresholds) - 1:
            loc = torch.logical_and(truth[mask] >= threshold, truth[mask] < thresholds[i + 1])
        else:
            loc = truth[mask] >= threshold
        pred_flatten, truth_flatten = pred[mask][loc], truth[mask][loc]
        if len(pred_flatten) == 0:
            rmse = 0
        else:
            rmse = torch.sqrt(F.mse_loss(pred_flatten, truth_flatten))
            rmse = torch.nan_to_num(rmse).item()
        rmses.append(rmse)
    return thresholds, rmses


def evaluate_mbe_multi_thresholds(pred: torch.Tensor, truth: torch.Tensor, mask: torch.Tensor,
                                  thresholds: list = [0, 10, 20, 30, 40]) -> Tuple[list, list]:
    assert pred.size() == truth.size()
    pred, truth = pred.cpu(), truth.cpu()
    pred = torch.clip(pred, min=0, max=70)
    truth = torch.clip(truth, min=0, max=70)
    mask = torch.logical_not(mask.bool())
    mbes = []
    for i, threshold in enumerate(thresholds):
        if i < len(thresholds) - 1:
            loc = torch.logical_and(truth[mask] >= threshold, truth[mask] < thresholds[i + 1])
        else:
            loc = truth[mask] >= threshold
        pred_flatten, truth_flatten = pred[mask][loc], truth[mask][loc]
        if len(pred_flatten) == 0:
            mbe = 0
        else:
            mbe = torch.mean(pred_flatten - truth_flatten)
            mbe = torch.nan_to_num(mbe).item()
        mbes.append(mbe)
    return thresholds, mbes
