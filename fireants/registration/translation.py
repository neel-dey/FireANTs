# Copyright (c) 2026 Rohit Jena. All rights reserved.
#
# This file is part of FireANTs, distributed under the terms of
# the FireANTs License version 1.0. A copy of the license can be found
# in the LICENSE file at the root of this repository.
#
# IMPORTANT: This code is part of FireANTs and its use, reproduction, or
# distribution must comply with the full license terms, including:
# - Maintaining all copyright notices and bibliography references
# - Using only approved (re)-distribution channels
# - Proper attribution in derivative works
#
# For full license details, see: https://github.com/rohitrango/FireANTs/blob/main/LICENSE


import math

import torch


def translation_parameters(fixed_images, normalize_translation, translation_lr, optimizer_lr):
    """Return a physical radius and dimensionless translation learning rate.

    The radius is the RMS half-extent of the fixed physical FOV. It is invariant
    to image origin, orientation and resampling at the same physical extent.
    Legacy mode retains translations in physical units and the original LR.
    """
    matrix = fixed_images.get_torch2phy()[:, :-1, :-1].detach()
    if not normalize_translation:
        if translation_lr is not None:
            raise ValueError("translation_lr requires normalize_translation=True")
        return torch.ones((matrix.shape[0], 1), device=matrix.device, dtype=matrix.dtype), optimizer_lr
    rate = optimizer_lr if translation_lr is None else translation_lr
    if not math.isfinite(rate) or rate <= 0:
        raise ValueError("translation_lr must be finite and positive")
    radius = matrix.square().sum(dim=(-2, -1)).div(matrix.shape[-1]).sqrt()[:, None]
    if not torch.isfinite(radius).all() or (radius <= 0).any():
        raise ValueError("Fixed images must have a finite, positive physical FOV")
    return radius, rate
