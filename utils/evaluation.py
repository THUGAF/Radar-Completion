import numpy as np
import torch
import torch.nn.functional as F
import utils.ssim as ssim
import utils.scaler as scaler


def evaluate_mae(pred, truth):
    assert pred.size() == truth.size()
    pred, truth = pred.cpu(), truth.cpu()
    mae = F.l1_loss(pred, truth).item()
    return mae


def evaluate_rmse(pred, truth): 
    assert pred.size() == truth.size()
    pred, truth = pred.cpu(), truth.cpu()
    rmse = torch.sqrt(F.mse_loss(pred, truth)).item()
    return rmse


def evaluate_cossim(pred, truth):
    assert pred.size() == truth.size()
    pred, truth = pred.cpu(), truth.cpu()
    cossim = torch.mean(torch.cosine_similarity(
            pred.view(pred.size(0), -1), truth.view(truth.size(0), -1))).item()
    return cossim


def evaluate_ssim(pred, truth):
    assert pred.size() == truth.size()
    pred, truth = pred.cpu(), truth.cpu()
    pred, truth = scaler.convert_to_gray(pred), scaler.convert_to_gray(truth)
    pred, truth = pred.float(), truth.float()
    ssim_ = ssim.ssim(pred, truth).item()
    return ssim_


def evaluate_psnr(pred, truth):
    assert pred.size() == truth.size()
    pred, truth = pred.cpu(), truth.cpu()
    pred, truth = scaler.convert_to_gray(pred), scaler.convert_to_gray(truth)
    pred, truth = pred.float(), truth.float()
    mse = [F.mse_loss(pred[b], truth[b]) for b in range(pred.size(0))]
    psnr = np.mean(10 * np.log10(np.max(pred.numpy(), axis=(1, 2, 3)) ** 2 / mse))
    return psnr
