# -*- coding: utf-8 -*-
"""Descriptor-guided detection: the architectural part of Section II-C.

The supervisor asked for a real structural contribution rather than "a detector
plus a post-processing step".  Three small modules turn the explicit descriptor
from a mere extra input channel into a guide inside the network:

1. DescriptorGate        F' = F * (1 + gamma * P_mv)
   a per-pixel multiplicative gate driven by the analytic descriptor.  gamma is
   a single learnable scalar initialised at 0, so training starts exactly from
   the baseline detector and the gate only opens if it pays off.

2. MembershipHead        1x1 conv -> sigmoid, attached to the neck
   predicts a pixel-level ship membership map and is supervised by the
   descriptor itself (the teacher soft label / P_mv), so the detector carries
   an interpretable mask branch instead of a pure black box.

3. descriptor_consistency_loss
   lambda * BCE(mask_head, descriptor_map) - keeps the learned mask faithful to
   the explicit rule; because the target is analytic, this term introduces no
   annotation of its own.

At inference the mask branch is used twice:
  * rescoring - the box score is multiplied by the mean membership inside it,
    which suppresses boxes that cover mostly water (clutter false alarms);
  * tightening - the mask inside the box gives B' = bbox(mask), the explicit
    refinement of src/assist.py.

Hook-up in the existing training script (05_detector.py), after the neck:
    gate = DescriptorGate(init=0.0)                 # one scalar
    head = MembershipHead(in_ch=neck_ch)
    neck = gate(neck, descriptor_map)               # same shape
    mask_logits = head(neck)
    loss = det_loss + args.lambda_pix * descriptor_consistency_loss(
        mask_logits, descriptor_map)
The descriptor map of a batch is P_mv = exp(-c / (P_m + P_v)) from the Yamaguchi
components (see scripts/06_assist.py:descriptor_map).
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class DescriptorGate(nn.Module):
    """F' = F * (1 + gamma * D), with D the descriptor map in [0, 1]."""

    def __init__(self, init: float = 0.0):
        super().__init__()
        self.gamma = nn.Parameter(torch.tensor(float(init)))

    def forward(self, features: torch.Tensor, descriptor: torch.Tensor) -> torch.Tensor:
        if descriptor.dim() == features.dim() - 1:      # (B, H, W) -> (B, 1, H, W)
            descriptor = descriptor.unsqueeze(1)
        if descriptor.shape[-2:] != features.shape[-2:]:
            descriptor = F.interpolate(descriptor, size=features.shape[-2:],
                                       mode='bilinear', align_corners=False)
        return features * (1.0 + self.gamma * descriptor)


class MembershipHead(nn.Module):
    """Pixel-level ship membership predicted by the detector."""

    def __init__(self, in_ch: int, hidden: Optional[int] = None):
        super().__init__()
        hidden = hidden or max(16, in_ch // 4)
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, hidden, 3, padding=1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, 1, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features)                        # logits, (B, 1, H, W)


def descriptor_consistency_loss(mask_logits: torch.Tensor,
                                descriptor: torch.Tensor,
                                weight: Optional[torch.Tensor] = None) -> torch.Tensor:
    """BCE between the learned membership and the analytic descriptor."""
    if descriptor.dim() == mask_logits.dim() - 1:
        descriptor = descriptor.unsqueeze(1)
    if descriptor.shape[-2:] != mask_logits.shape[-2:]:
        descriptor = F.interpolate(descriptor, size=mask_logits.shape[-2:],
                                   mode='bilinear', align_corners=False)
    return F.binary_cross_entropy_with_logits(mask_logits, descriptor.clamp(1e-6, 1 - 1e-6),
                                              weight=weight)


@torch.no_grad()
def mask_score(mask_logits: torch.Tensor, boxes: torch.Tensor) -> torch.Tensor:
    """Mean membership inside each box (B, 4) -> (B,) - the rescoring factor."""
    prob = torch.sigmoid(mask_logits)
    if prob.dim() == 3:
        prob = prob.unsqueeze(1)
    n, _, h, w = prob.shape
    out = []
    for i in range(boxes.shape[0]):
        x1, y1, x2, y2 = boxes[i].tolist()
        x1, x2 = int(max(0, min(x1, w - 1))), int(max(1, min(x2, w)))
        y1, y2 = int(max(0, min(y1, h - 1))), int(max(1, min(y2, h)))
        patch = prob[i, 0, y1:y2, x1:x2]
        out.append(patch.mean() if patch.numel() else torch.tensor(0.0, device=prob.device))
    return torch.stack(out)


@torch.no_grad()
def tighten_boxes(mask_logits: torch.Tensor, boxes: torch.Tensor, threshold: float,
                  min_area: int) -> tuple:
    """bbox of the mask inside each box; boxes whose mask is too small are dropped."""
    prob = (torch.sigmoid(mask_logits) >= threshold)
    if prob.dim() == 3:
        prob = prob.unsqueeze(1)
    keep, new_boxes = [], []
    for i in range(boxes.shape[0]):
        x1, y1, x2, y2 = (int(v) for v in boxes[i].tolist())
        patch = prob[i, 0, max(0, y1):y2, max(0, x1):x2]
        if patch.numel() == 0:
            continue
        ys, xs = torch.nonzero(patch, as_tuple=True)
        if ys.numel() < min_area:
            continue
        keep.append(i)
        new_boxes.append([x1 + xs.min().item(), y1 + ys.min().item(),
                          x1 + xs.max().item() + 1, y1 + ys.max().item() + 1])
    idx = torch.as_tensor(keep, dtype=torch.long, device=boxes.device)
    return idx, torch.as_tensor(new_boxes, dtype=boxes.dtype, device=boxes.device)
