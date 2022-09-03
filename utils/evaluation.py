import numpy as np
import torch
import torch.nn.functional as F
import utils.ssim as ssim
import utils.scaler as scaler


def _count(pred, truth, threshold):
    assert pred.size() == truth.size()
    pred, truth = pred.cpu(), truth.cpu()
    seq_len = pred.size(0)
    
    hits = []
    misses = []
    false_alarms = []
    correct_rejections = []

    for s in range(seq_len):
        stat = 2 * (truth[s] > threshold).int() + (pred[s] > threshold).int()
        hit = torch.sum(stat == 3).item()
        miss = torch.sum(stat == 2).item()
        false_alarm = torch.sum(stat == 1).item()
        correct_rejection = torch.sum(stat == 0).item()
        hits.append(hit)
        misses.append(miss)
        false_alarms.append(false_alarm)
        correct_rejections.append(correct_rejection)

    return np.array(hits), np.array(misses), np.array(false_alarms), np.array(correct_rejections)


def evaluate_forecast(pred, truth, threshold, eps=1e-4):
    r"""To calculate POD, FAR, CSI and HSS for the prediction at each time step.
    
    Args:
        pred (torch.Tensor): The prediction sequence in tensor form with 5D shape `(S, B, C, H, W)`.
        truth (torch.Tensor): The ground truth sequence in tensor form with 5D shape `(S, B, C, H, W)`.
        threshold (float, optional): The threshold of POD, FAR, CSI and HSS. Range: (0, 1).
    
    Return:
        numpy.ndarray: POD at each time step.
        numpy.ndarray: FAR at each time step.
        numpy.ndarray: CSI at each time step.
        numpy.ndarray: HSS at each time step.
    """

    h, m, f, c = _count(pred, truth, threshold)
    pod = h / (h + m + eps)
    far = f / (h + f + eps)
    csi = h / (h + m + f + eps)
    hss = 2 * (h * c - m * f) / ((h + m) * (m + c) + (h + f) * (f + c) + eps)
    
    return pod, far, csi, hss


def evaluate_ssd(tensor):
    r"""To calculate SSD for the prediction at each time step.

    Args:
        tensor (torch.Tensor): Tensor with 5D shape `(S, B, C, H, W)`.
    Return:
        numpy.ndarray: SSD at each time step.
    """

    tensor = tensor.cpu()
    tensor = scaler.convert_to_gray(tensor)
    seq_len, batch_size = tensor.size(0), tensor.size(1)
    
    ssd_list = []
    for s in range(seq_len):
        left_pad = F.pad(tensor[s], (1, 0, 0, 0))
        right_pad = F.pad(tensor[s], (0, 1, 0, 0))
        up_pad = F.pad(tensor[s], (0, 0, 1, 0))
        bottom_pad = F.pad(tensor[s], (0, 0, 0, 1))

        diff_h = left_pad - right_pad
        diff_v = up_pad - bottom_pad
        # smd = torch.sum(torch.abs(diff_h[:, :, 1:-1])) + torch.sum(torch.abs(diff_v[:, :, 1:-1]))
        ssd = torch.sum(torch.pow(diff_h[:, :, 1:-1], 2)) + torch.sum(torch.pow(diff_v[:, :, 1:-1], 2))
        ssd = ssd / batch_size
        ssd_list.append(ssd)
    
    return np.array(ssd_list)


def evaluate_ssdr(pred, truth):
    ssd_pred, ssd_truth = evaluate_ssd(pred), evaluate_ssd(truth)
    ssdr = ssd_pred / ssd_truth
    return ssdr


def evaluate_cc(pred, truth):
    assert pred.size() == truth.size()
    pred, truth = pred.cpu(), truth.cpu()
    seq_len = pred.size(0)

    cc_list = []
    for s in range(seq_len):
        cc = torch.corrcoef(torch.stack([pred[s].flatten(), truth[s].flatten()]))[0, 1]
        cc_list.append(cc)

    return np.array(cc_list)
    

def evaluate_me(pred, truth):
    assert pred.size() == truth.size()
    pred, truth = pred.cpu(), truth.cpu()
    seq_len = pred.size(0)

    me_list = []
    for s in range(seq_len):
        me = torch.mean(pred[s] - truth[s])
        me_list.append(me)

    return np.array(me_list)


def evaluate_mae(pred, truth):
    assert pred.size() == truth.size()
    pred, truth = pred.cpu(), truth.cpu()
    seq_len = pred.size(0)

    mae_list = []
    for s in range(seq_len):
        mae = F.l1_loss(pred[s], truth[s])
        mae_list.append(mae)

    return np.array(mae_list)


def evaluate_rmse(pred, truth): 
    assert pred.size() == truth.size()
    pred, truth = pred.cpu(), truth.cpu()
    seq_len = pred.size(0)

    rmse_list = []
    for s in range(seq_len):
        rmse = torch.sqrt(F.mse_loss(pred[s], truth[s]))
        rmse_list.append(rmse)

    return np.array(rmse_list)


def evaluate_ssim(pred, truth):
    assert pred.size() == truth.size()
    pred, truth = pred.cpu(), truth.cpu()
    pred, truth = scaler.convert_to_gray(pred), scaler.convert_to_gray(truth)
    pred, truth = pred.float(), truth.float()
    seq_len = pred.size(0)

    ssim_list = []
    for s in range(seq_len):
        ssim_ = ssim.ssim(pred[s], truth[s])
        ssim_list.append(ssim_)

    return np.array(ssim_list)


def evaluate_psnr(pred, truth):
    assert pred.size() == truth.size()
    pred, truth = pred.cpu(), truth.cpu()
    pred, truth = scaler.convert_to_gray(pred), scaler.convert_to_gray(truth)
    pred, truth = pred.float(), truth.float()
    seq_len = pred.size(0)

    psnr_list = []
    for s in range(seq_len):
        mse = F.mse_loss(pred[s], truth[s])
        psnr = 10 * torch.log10(torch.max(pred) ** 2 / mse)
        psnr_list.append(psnr)

    return np.array(psnr_list)


def evaluate_cvr(pred, truth):
    assert pred.size() == truth.size()
    pred, truth = pred.cpu(), truth.cpu()
    seq_len = pred.size(0)

    cvr_list = []
    for s in range(seq_len):
        pred_cv = torch.std(pred[s]) / torch.mean(pred[s])
        truth_cv = torch.std(truth[s]) / torch.mean(truth[s])
        cvr_list.append(pred_cv / truth_cv)

    return np.array(cvr_list)
