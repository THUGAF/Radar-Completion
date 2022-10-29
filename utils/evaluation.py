import numpy as np
import torch
import torch.nn.functional as F


def evaluate_mae(pred: torch.Tensor, truth: torch.Tensor, mask: torch.Tensor):
    assert pred.size() == truth.size()
    pred, truth = pred.cpu(), truth.cpu()
    mask = torch.logical_not(mask.bool())
    mae = F.l1_loss(pred[mask], truth[mask]).item()
    return mae


def evaluate_rmse(pred: torch.Tensor, truth: torch.Tensor, mask: torch.Tensor):
    assert pred.size() == truth.size()
    pred, truth = pred.cpu(), truth.cpu()
    mask = torch.logical_not(mask.bool())
    rmse = torch.sqrt(F.mse_loss(pred[mask], truth[mask])).item()
    return rmse


def evaluate_cossim(pred: torch.Tensor, truth: torch.Tensor, mask: torch.Tensor):
    assert pred.size() == truth.size()
    pred, truth = pred.cpu(), truth.cpu()
    mask = torch.logical_not(mask.bool())
    cossim = torch.mean(torch.cosine_similarity(
            pred[mask].view(pred.size(0), -1), 
            truth[mask].view(truth.size(0), -1))).item()
    return cossim
