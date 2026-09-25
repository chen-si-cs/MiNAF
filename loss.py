import torch
import torch.nn as nn


def compute_loss(pred_spec, gt_spec):
    l1_loss_magnitude = torch.nn.functional.l1_loss(pred_spec, gt_spec)

    return l1_loss_magnitude
