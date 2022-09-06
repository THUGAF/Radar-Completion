import torch
import torch.nn as nn
import torch.nn.functional as F


def completion_network_loss(input, output, mask):
    return F.mse_loss(output * mask, input * mask)


def biased_mse_loss(pred, truth, vmax):
    weight = (truth < 10 / vmax) * 1 \
        + (torch.logical_and(truth >= 10 / vmax, truth < 20 / vmax)) * 2 \
        + (torch.logical_and(truth >= 20 / vmax, truth < 30 / vmax)) * 5 \
        + (torch.logical_and(truth >= 30 / vmax, truth < 40 / vmax)) * 10 \
        + (truth >= 40 / vmax) * 30
    return torch.mean(weight * (pred - truth) ** 2)


def biased_mae_loss(pred, truth, vmax):
    weight = (truth < 10 / vmax) * 1 \
        + (torch.logical_and(truth >= 10 / vmax, truth < 20 / vmax)) * 2 \
        + (torch.logical_and(truth >= 20 / vmax, truth < 30 / vmax)) * 5 \
        + (torch.logical_and(truth >= 30 / vmax, truth < 40 / vmax)) * 10 \
        + (truth >= 40 / vmax) * 30
    return torch.mean(weight * torch.abs(pred - truth))


def cal_d_loss(fake_score, real_score, loss_func=nn.BCELoss()):
    """Calculate loss function of the discriminators.

    Args:
        fake_score: Score of fake.
        real_score: Score of real.

    Returns:
        torch.Tensor: Loss of discriminator.
    """
    label = torch.ones_like(fake_score).type_as(fake_score)
    loss_pred = loss_func(fake_score, label * 0.0)
    loss_truth = loss_func(real_score, label * 1.0)
    d_loss = (loss_pred + loss_truth) / 2
    return d_loss


def cal_g_loss(fake_score, loss_func=nn.BCELoss()):
    """Calculate loss function of the generator.

    Args:
        fake_score: Score of fake.

    Returns:
        torch.Tensor: Loss of the generator.
    """

    label = torch.ones_like(fake_score).type_as(fake_score)
    g_loss = loss_func(fake_score, label * 1.0)
    return g_loss
