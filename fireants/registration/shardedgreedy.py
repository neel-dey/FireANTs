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

"""Greedy deformable registration split over several GPUs from one process.

The fixed grid is cut into slabs along one axis, one slab per device. Every
quantity that couples neighbouring slabs (the local loss windows, the gradient
and warp smoothing, the compositional update) reads a halo copied from the
neighbouring device, so the optimization follows `GreedyRegistration` up to
floating-point summation order. Halos of the moved image are copied inside
autograd, which carries the loss gradient across the seams.

Optionally the loss is evaluated a few feature channels at a time
(`channel_chunk`), which bounds the loss memory by the chunk instead of the
full channel count. This also helps on a single device.
"""
import copy
import math
from typing import List, Optional, Sequence, Union

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from tqdm import tqdm

from fireants.interpolator import fireants_interpolator
from fireants.io.image import BatchedImages, FakeBatchedImages
from fireants.losses.cc import gaussian_1d, separable_filtering
from fireants.registration.abstract import AbstractRegistration
from fireants.registration.deformablemixin import DeformableMixin
from fireants.registration.optimizers.adam import adam_update_fused
from fireants.utils.globals import MIN_IMG_SIZE
from fireants.utils.imageutils import downsample

import logging
logger = logging.getLogger(__name__)

_MASK_EPS = 1e-8  # as in fireants.losses.maskedutils.mask_loss_function


def _split(size: int, parts: int) -> List[tuple]:
    """[lo, hi) index ranges of `parts` near-equal chunks of `size`."""
    base, rem = divmod(size, parts)
    ranges, lo = [], 0
    for i in range(parts):
        hi = lo + base + (1 if i < rem else 0)
        ranges.append((lo, hi))
        lo = hi
    return ranges


def _box_matrix(los, his, sizes) -> torch.Tensor:
    """4x4 map from a voxel box's own normalized coordinates to those of the full grid.

    `los`, `his`, `sizes` are per spatial axis (z, y, x); the matrix is in the
    (x, y, z) order of sampling grids, with align_corners=True.
    """
    mat = torch.eye(4, dtype=torch.float64)
    for axis, (lo, hi, n) in enumerate(zip(los, his, sizes)):
        g_lo = 2.0 * lo / (n - 1) - 1.0
        g_hi = 2.0 * (hi - 1) / (n - 1) - 1.0
        k = 2 - axis
        mat[k, k] = (g_hi - g_lo) / 2.0
        mat[k, 3] = (g_hi + g_lo) / 2.0
    return mat


def _staged_copy(tensor, device):
    host = torch.empty(tensor.shape, dtype=tensor.dtype, pin_memory=True)
    host.copy_(tensor)
    return host.to(device, non_blocking=True)


class _StagedCopy(torch.autograd.Function):
    """Differentiable device-to-device copy through pinned host memory."""

    @staticmethod
    def forward(ctx, tensor, device):
        ctx.source = tensor.device
        return _staged_copy(tensor, device)

    @staticmethod
    def backward(ctx, grad):
        return _staged_copy(grad, ctx.source), None


def _peer_copy_fails(src, dst) -> bool:
    """Whether a direct copy from `src` to `dst` is seen to corrupt data.

    Some hosts report peer access between two GPUs and still corrupt the copies (IOMMU / PCIe ACS),
    sometimes only intermittently, so passing this check does not prove the link safe.
    """
    for numel in (1 << 10, 1 << 20, 1 << 24):
        probe = torch.randn(numel, device=src)
        copied = probe.to(dst)
        torch.cuda.synchronize(src)
        torch.cuda.synchronize(dst)
        if not torch.equal(probe.cpu(), copied.cpu()):
            return True
    return False


class _Shard:
    """State of one slab: its device, index range on the sharded axis, and tensors."""

    def __init__(self, device, lo, hi):
        self.device = device
        self.lo, self.hi = lo, hi
        self.warp = None
        self.exp_avg = None
        self.exp_avg_sq = None


class ShardedGreedyRegistration(AbstractRegistration, DeformableMixin):
    """Compositive greedy registration over several CUDA devices in one process.

    Takes the arguments of `GreedyRegistration`, plus:

    Args:
        devices: CUDA devices to split the fixed grid over. One device is allowed.
        dim_to_shard: spatial axis (0, 1, 2) to cut; the longest fixed axis by default.
        channel_chunk: number of image channels per loss evaluation. None evaluates
            all channels at once. Smaller values need less memory and more time.
        moving_margin: slack, in voxels, around the part of the moving image each
            slab keeps on its device. The part is re-cut when the warp outgrows it.
        peer_copies: copy halos directly between GPUs instead of through pinned host
            memory. Faster over NVLink, but some PCIe hosts corrupt such copies, at
            times intermittently; enable it only on hardware known to be sound.

    The images may live on the CPU: only slabs are moved to the devices. The
    loss must be local (`cc`, `fusedcc`, `mse` and their masked variants), the
    deformation compositive and the optimizer Adam.
    """

    def __init__(self, scales: List[float], iterations: List[int],
                 fixed_images: BatchedImages, moving_images: BatchedImages,
                 devices: Sequence[Union[str, torch.device]],
                 loss_type: str = "cc",
                 deformation_type: str = "compositive",
                 optimizer: str = "Adam", optimizer_params: dict = {},
                 optimizer_lr: float = 0.5,
                 mi_kernel_type: str = "gaussian", cc_kernel_type: str = "rectangular",
                 cc_kernel_size: int = 3,
                 smooth_warp_sigma: float = 0.5,
                 smooth_grad_sigma: float = 1.0,
                 loss_params: dict = {},
                 reduction: str = "mean",
                 tolerance: float = 1e-6, max_tolerance_iters: int = 10,
                 init_affine: Optional[torch.Tensor] = None,
                 warp_reg=None, displacement_reg=None,
                 blur: bool = True,
                 freeform: bool = False,
                 dim_to_shard: Optional[int] = None,
                 channel_chunk: Optional[int] = None,
                 moving_margin: int = 16,
                 peer_copies: bool = False,
                 custom_loss: nn.Module = None, **kwargs) -> None:
        super().__init__(scales=scales, iterations=iterations, fixed_images=fixed_images, moving_images=moving_images,
                         loss_type=loss_type, mi_kernel_type=mi_kernel_type, cc_kernel_type=cc_kernel_type,
                         custom_loss=custom_loss, loss_params=loss_params,
                         cc_kernel_size=cc_kernel_size, reduction=reduction,
                         tolerance=tolerance, max_tolerance_iters=max_tolerance_iters, **kwargs)
        self.devices = [torch.device(d) for d in devices]
        if not self.devices or any(d.type != "cuda" for d in self.devices):
            raise ValueError("devices must be a non-empty list of CUDA devices")
        if len(set(self.devices)) != len(self.devices):
            raise ValueError("devices must be distinct")
        if self.dims != 3:
            raise NotImplementedError("sharded registration supports 3D images only")
        if deformation_type != "compositive":
            raise NotImplementedError("sharded registration supports the compositive deformation only")
        if optimizer.lower() != "adam":
            raise NotImplementedError("sharded registration supports the Adam optimizer only")
        if reduction not in ("mean", "sum"):
            raise ValueError(f"unsupported reduction: {reduction}")
        if not (hasattr(self.loss_fn, "forward_util") and hasattr(self.loss_fn, "get_image_padding")) \
                or "MutualInformation" in type(self.loss_fn).__name__:
            raise NotImplementedError(
                f"{type(self.loss_fn).__name__} is not a local loss; sharded registration needs cc, fusedcc or mse")
        if fixed_images().shape[0] != moving_images().shape[0]:
            raise NotImplementedError("sharded registration needs equal fixed and moving batch sizes")

        oparams = dict(optimizer_params)
        self.beta1 = oparams.pop("beta1", 0.9)
        self.beta2 = oparams.pop("beta2", 0.99)
        self.eps = oparams.pop("eps", 1e-8)
        self.weight_decay = oparams.pop("weight_decay", 0)
        self.scaledown = oparams.pop("scaledown", False)
        self.reset_step = oparams.pop("reset_step", True)
        if warp_reg is not None or displacement_reg is not None:
            raise NotImplementedError("sharded registration does not support warp regularizers")
        if oparams.pop("freeform", False) or freeform:
            raise NotImplementedError("sharded registration does not support freeform updates")
        if oparams:
            raise NotImplementedError(f"unsupported optimizer_params for sharded registration: {sorted(oparams)}")

        self.optimizer_lr = optimizer_lr
        self.reduction = reduction
        self.blur = blur
        self.smooth_grad_sigma = smooth_grad_sigma
        self.warp_sigma = smooth_warp_sigma
        self.smooth_warp_sigma = 0  # the step smooths the warp, as CompositiveWarp does
        self.channel_chunk = channel_chunk
        self.moving_margin = int(moving_margin)
        self.output_device = self.devices[0]
        self.step_t = 0

        fixed_size = list(fixed_images.shape[2:])
        if dim_to_shard is None:
            dim_to_shard = int(np.argmax(fixed_size))
        self.dim_to_shard = dim_to_shard

        if init_affine is None:
            init_affine = torch.eye(self.dims + 1)[None].repeat(self.opt_size, 1, 1)
        init_affine = init_affine.detach().to(self.dtype).cpu()
        if tuple(init_affine.shape[1:]) == (self.dims, self.dims + 1):
            row = torch.zeros(self.opt_size, 1, self.dims + 1, dtype=self.dtype)
            row[:, 0, -1] = 1.0
            init_affine = torch.cat([init_affine, row], dim=1)
        elif tuple(init_affine.shape[1:]) != (self.dims + 1, self.dims + 1):
            raise ValueError(f"Invalid initial affine shape: {init_affine.shape}")
        self.affine = init_affine.contiguous()

        self.peer_copies = peer_copies
        if peer_copies and any(_peer_copy_fails(a, b) for a in self.devices for b in self.devices if a != b):
            logger.warning("Direct GPU-to-GPU copies corrupt data on this host; copying through host memory.")
            self.peer_copies = False

        # one loss module per device, returning the per-voxel loss
        self._losses = {}
        for device in self.devices:
            loss = copy.deepcopy(self.loss_fn).to(device)
            loss.reduction = "none"
            self._losses[device] = loss

        first = [max(int(s / scales[0]), MIN_IMG_SIZE) for s in fixed_size] if scales[0] > 1 else fixed_size
        self._level_size = None
        self._shards: List[_Shard] = []
        self._build_shards(first, halo=1)
        for shard in self._shards:
            shape = [self.opt_size, *self._shard_size(shard, first), self.dims]
            shard.warp = nn.Parameter(torch.zeros(shape, dtype=self.dtype, device=shard.device))
            shard.exp_avg = torch.zeros_like(shard.warp)
            shard.exp_avg_sq = torch.zeros_like(shard.warp)

    # ------------------------------------------------------------------ layout

    @property
    def _vdim(self):
        """Position of the sharded axis in a [N, Z, Y, X, 3] field."""
        return 1 + self.dim_to_shard

    @property
    def _idim(self):
        """Position of the sharded axis in a [N, C, Z, Y, X] image."""
        return 2 + self.dim_to_shard

    def _shard_size(self, shard, level_size):
        size = list(level_size)
        size[self.dim_to_shard] = shard.hi - shard.lo
        return size

    def _build_shards(self, level_size, halo):
        """Cut the level grid into as many slabs as devices, each at least `halo` thick."""
        length = level_size[self.dim_to_shard]
        parts = max(1, min(len(self.devices), length // max(halo, 2)))
        self._level_size = list(level_size)
        self._shards = [_Shard(self.devices[i], lo, hi) for i, (lo, hi) in enumerate(_split(length, parts))]

    def _move(self, tensor, device):
        """`tensor.to(device)`, inside autograd; between GPUs through the host unless `peer_copies`."""
        device = torch.device(device)
        if tensor.device == device or "cpu" in (tensor.device.type, device.type) or self.peer_copies:
            return tensor.to(device)
        return _StagedCopy.apply(tensor, device)

    def _gather(self, name, device) -> torch.Tensor:
        return torch.cat([self._move(getattr(s, name).detach(), device) for s in self._shards], dim=self._vdim)

    def _resize_state(self, level_size, halo):
        """Resample the warp and the Adam moments to a new level and re-cut them."""
        old_size = self._level_size
        device = self.devices[0]
        full = {}
        for name in ("warp", "exp_avg", "exp_avg_sq"):
            field = self._gather(name, device)
            for shard in self._shards:
                setattr(shard, name, None)
            if list(old_size) != list(level_size):
                field = F.interpolate(field.permute(0, 4, 1, 2, 3), size=list(level_size), mode="trilinear",
                                      align_corners=True).permute(0, 2, 3, 4, 1)
            full[name] = field
        self._build_shards(level_size, halo)
        for shard in self._shards:
            for name, field in full.items():
                part = self._move(field.narrow(self._vdim, shard.lo, shard.hi - shard.lo), shard.device).contiguous()
                setattr(shard, name, nn.Parameter(part) if name == "warp" else part)
        if self.reset_step:
            self.step_t = 0

    def _pad(self, tensors, dim, width):
        """Append to every slab the `width` bordering slices of its neighbours."""
        if width <= 0 or len(tensors) == 1:
            return list(tensors)
        out = []
        for i, tensor in enumerate(tensors):
            parts = [tensor]
            if i > 0:
                prev = tensors[i - 1]
                parts.insert(0, self._move(prev.narrow(dim, prev.shape[dim] - width, width), tensor.device))
            if i < len(tensors) - 1:
                parts.append(self._move(tensors[i + 1].narrow(dim, 0, width), tensor.device))
            out.append(torch.cat(parts, dim=dim))
        return out

    def _crop(self, tensor, index, dim, width):
        """Undo `_pad` for the slab at `index`."""
        if width <= 0 or len(self._shards) == 1:
            return tensor
        lo = width if index > 0 else 0
        hi = tensor.shape[dim] - (width if index < len(self._shards) - 1 else 0)
        return tensor.narrow(dim, lo, hi - lo)

    def _halo_range(self, shard, width):
        lo = max(shard.lo - width, 0) if len(self._shards) > 1 else shard.lo
        hi = min(shard.hi + width, self._level_size[self.dim_to_shard]) if len(self._shards) > 1 else shard.hi
        return lo, hi

    def _box(self, lo, hi, size):
        """Box matrix of the [lo, hi) range of the sharded axis on a grid of `size`."""
        los, his = [0] * self.dims, list(size)
        los[self.dim_to_shard], his[self.dim_to_shard] = lo, hi
        return _box_matrix(los, his, size)

    def _smooth(self, fields, gaussians_per_device, radius):
        """Gaussian-filter a sharded [N, Z, Y, X, 3] field as one volume."""
        padded = self._pad(fields, self._vdim, radius)
        out = []
        for i, (shard, field) in enumerate(zip(self._shards, padded)):
            with torch.cuda.device(shard.device):
                field = separable_filtering(field.permute(0, 4, 1, 2, 3).contiguous(),
                                            gaussians_per_device[shard.device]).permute(0, 2, 3, 4, 1)
                out.append(self._crop(field, i, self._vdim, radius).contiguous())
        return out

    # ------------------------------------------------------------------ images

    def _level_image(self, arrays, size, mode, scale):
        """The image at one pyramid level, as `GreedyRegistration` computes it, a few channels at a time."""
        if scale == 1:
            return arrays
        device = self.devices[0]
        img, mask = self._split_image_and_mask_last_channel(arrays)
        out = torch.empty([*arrays.shape[:2], *size], dtype=arrays.dtype, device=arrays.device)
        clamp_range = (img.min().item(), img.max().item()) if self.blur else None
        # the FFT downsampler holds several complex copies of its input: keep the input near 128 MiB
        chunk = max(1, int(2 ** 25 // max(int(np.prod(arrays.shape[2:])) * arrays.shape[0], 1)))
        with torch.cuda.device(device):
            for c0 in range(0, img.shape[1], chunk):
                part = self._move(img[:, c0:c0 + chunk], device)
                if self.blur:
                    part = downsample(part, size=size, mode=mode, clamp_range=clamp_range)
                else:
                    part = F.interpolate(part, size=size, mode=mode, align_corners=True)
                out[:, c0:c0 + part.shape[1]] = self._move(part, out.device)
            if mask is not None:
                out[:, -1:] = self._move(
                    F.interpolate(self._move(mask, device), size=size, mode=mode, align_corners=True), out.device)
        return out

    def _channel_chunks(self, channels):
        step = self.channel_chunk or channels
        return [(c0, min(c0 + step, channels)) for c0 in range(0, channels, step)]

    def _cut_fixed(self, fixed_level, halo):
        image_channels = fixed_level.shape[1] - (1 if self.masked else 0)
        for shard in self._shards:
            lo, hi = self._halo_range(shard, halo)
            slab = fixed_level.narrow(self._idim, lo, hi - lo)
            shard.fixed = [self._move(slab[:, c0:c1], shard.device).contiguous()
                           for c0, c1 in self._channel_chunks(image_channels)]
            shard.fixed_mask = None
            if self.masked:
                shard.fixed_mask = self._move(fixed_level[:, -1:].narrow(
                    self._idim, shard.lo, shard.hi - shard.lo), shard.device).contiguous()

    def _cut_moving(self, shard, moving_level, affine, margin):
        """Keep on the shard's device the box of the moving image its slab can sample."""
        msize = list(moving_level.shape[2:])
        out_box = self._box(shard.lo, shard.hi, self._level_size)
        corners = torch.tensor([[x, y, z, 1.0] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)],
                               dtype=torch.float64)
        mapped = torch.einsum("bij,kj->bki", affine.double() @ out_box, corners)[..., :3]  # [N, 8, xyz]
        g_min, g_max = mapped.amin(dim=(0, 1)), mapped.amax(dim=(0, 1))
        los, his = [], []
        for axis, n in enumerate(msize):
            k = 2 - axis
            lo = math.floor((g_min[k].item() + 1) / 2 * (n - 1)) - margin
            hi = math.ceil((g_max[k].item() + 1) / 2 * (n - 1)) + margin + 1
            lo, hi = max(lo, 0), min(hi, n)
            if hi - lo < 2:  # the slab falls outside the moving image
                lo = min(max(lo, 0), n - 2)
                hi = lo + 2
            los.append(lo)
            his.append(hi)
        in_box = _box_matrix(los, his, msize)
        crop = moving_level[:, :, los[0]:his[0], los[1]:his[1], los[2]:his[2]]
        image_channels = crop.shape[1] - (1 if self.masked else 0)
        shard.moving = [self._move(crop[:, c0:c1], shard.device).contiguous()
                        for c0, c1 in self._channel_chunks(image_channels)]
        shard.moving_mask = self._move(crop[:, -1:], shard.device).contiguous() if self.masked else None
        shard.moving_margin = margin
        shard.moving_is_full = all(lo == 0 and hi == n for lo, hi, n in zip(los, his, msize))
        shard.sample_affine = (torch.linalg.inv(in_box) @ affine.double() @ out_box)[:, :3].to(
            device=shard.device, dtype=self.dtype).contiguous()
        scale = torch.stack([in_box[k, k] for k in range(3)])
        shard.sample_scale = None if torch.all(scale == 1) else (1.0 / scale).to(device=shard.device, dtype=self.dtype)
        # a displacement component in voxels of the moving level grid
        shard.disp_to_voxels = torch.tensor([(msize[2 - k] - 1) / 2.0 for k in range(3)],
                                            device=shard.device, dtype=self.dtype)

    def _refresh_moving(self, moving_level, affine):
        """Re-cut the moving boxes that the current warp no longer fits in."""
        shards = [s for s in self._shards if not s.moving_is_full]
        if not shards:
            return
        device = self.devices[0]
        need = torch.stack([self._move((s.warp.detach().abs().amax(dim=(0, 1, 2, 3)) * s.disp_to_voxels).max(), device)
                            for s in shards]).tolist()
        for shard, voxels in zip(shards, need):
            required = math.ceil(voxels) + 1
            if required > shard.moving_margin:
                self._cut_moving(shard, moving_level, affine, required + self.moving_margin)

    def _sample(self, shard, image):
        """Moving `image` (a box on the shard's device) resampled onto the shard's slab."""
        warp = shard.warp if shard.sample_scale is None else shard.warp * shard.sample_scale
        return fireants_interpolator(image, affine=shard.sample_affine, grid=warp, mode="bilinear",
                                     align_corners=True, is_displacement=True)

    # ------------------------------------------------------------------ optimization

    def _loss_and_gradients(self, cc_halo):
        """Accumulate d(loss)/d(warp) in every slab's `warp.grad` and return the loss."""
        device = self.devices[0]
        batch = self.opt_size
        ratio = self.masked and self.reduction == "mean"
        space = tuple(range(1, self.dims + 2))

        denominator = None
        if ratio:
            # loss = mean_b N_b / (D_b + eps): take dD/dwarp here, dN/dwarp below, and combine
            parts = []
            for shard in self._shards:
                with torch.cuda.device(shard.device):
                    parts.append((shard.fixed_mask * self._sample(shard, shard.moving_mask)).sum(dim=space))
            denominator = sum(self._move(p, device) for p in parts)
            denominator.sum().backward()
            for shard in self._shards:
                shard.grad_denominator, shard.warp.grad = shard.warp.grad, None
            denominator = denominator.detach()

        weight = 1.0
        if self.reduction == "mean" and not self.masked:
            channels = sum(c.shape[1] for c in self._shards[0].fixed)
            weight = 1.0 / (batch * channels * float(np.prod(self._level_size)))

        numerator = torch.zeros(batch, dtype=self.dtype, device=device)
        for k in range(len(self._shards[0].fixed)):
            moved = []
            for shard in self._shards:
                with torch.cuda.device(shard.device):
                    moved.append(self._sample(shard, shard.moving[k]))
            padded = self._pad(moved, self._idim, cc_halo)
            del moved
            parts = []
            for i, shard in enumerate(self._shards):
                with torch.cuda.device(shard.device):
                    values = self._losses[shard.device].forward_util(padded[i], shard.fixed[k])
                    values = self._crop(values, i, self._idim, cc_halo)
                    if self.masked:
                        values = values * (shard.fixed_mask * self._sample(shard, shard.moving_mask))
                    parts.append(values.sum(dim=space))
            del padded, values
            chunk_sum = sum(self._move(p, device) for p in parts)
            (chunk_sum.sum() * weight).backward()
            numerator += chunk_sum.detach()
            del parts, chunk_sum

        if not ratio:
            return (numerator.sum() * weight).item()
        scale = 1.0 / (denominator + _MASK_EPS)
        for shard in self._shards:
            with torch.cuda.device(shard.device):
                shape = [batch] + [1] * (self.dims + 1)
                a = self._move(scale / batch, shard.device).view(shape)
                b = self._move(numerator * scale * scale / batch, shard.device).view(shape)
                shard.warp.grad.mul_(a).sub_(shard.grad_denominator * b)
                shard.grad_denominator = None
        return (numerator * scale).mean().item()

    def _step(self, halos):
        """The WarpAdam diffeomorphic update, with its global quantities taken over all slabs."""
        grad_halo, warp_halo, compose_halo = halos
        grads = [s.warp.grad for s in self._shards]
        if self.smooth_grad_sigma > 0:
            grads = self._smooth(grads, self._grad_gaussians, grad_halo)

        self.step_t += 1
        bias1 = 1 - self.beta1 ** self.step_t
        bias2 = 1 - self.beta2 ** self.step_t
        norms = []
        for shard, grad in zip(self._shards, grads):
            with torch.cuda.device(shard.device):
                if self.weight_decay > 0:
                    grad.add_(shard.warp.data, alpha=self.weight_decay)
                shard.exp_avg.mul_(self.beta1).add_(grad, alpha=1 - self.beta1)
                shard.exp_avg_sq.mul_(self.beta2).addcmul_(grad, grad.conj(), value=1 - self.beta2)
                adam_update_fused(grad, shard.exp_avg, shard.exp_avg_sq, bias1, bias2, self.eps)
                norms.append(grad.norm(p=2, dim=-1).flatten(1).max(1).values)
        device = self.devices[0]
        gradmax = self.eps + torch.stack([self._move(n, device) for n in norms]).max(0).values
        if not self.scaledown:
            gradmax = torch.clamp(gradmax, min=1)
        gradmax = gradmax.reshape(-1, *([1] * (self.dims + 1)))
        half_resolution = 1.0 / (max(self._level_size) - 1)

        old = self._pad([s.warp.data for s in self._shards], self._vdim, compose_halo)
        updated = []
        for shard, grad, padded in zip(self._shards, grads, old):
            with torch.cuda.device(shard.device):
                grad.div_(self._move(gradmax, shard.device)).mul_(half_resolution).mul_(-self.optimizer_lr)
                v = grad if shard.compose_scale is None else grad * shard.compose_scale
                grad.add_(fireants_interpolator.warp_composer(
                    padded.contiguous(), affine=shard.compose_affine, v=v.contiguous(), align_corners=True))
                updated.append(grad)
        del old
        if self.warp_sigma > 0:
            updated = self._smooth(updated, self._warp_gaussians, warp_halo)
        for shard, field in zip(self._shards, updated):
            shard.warp.data.copy_(field)

    def _prepare_compose(self, compose_halo):
        for shard in self._shards:
            lo, hi = self._halo_range(shard, compose_halo)
            in_box = self._box(lo, hi, self._level_size)
            out_box = self._box(shard.lo, shard.hi, self._level_size)
            affine = (torch.linalg.inv(in_box) @ out_box)[:3]
            shard.compose_affine = affine[None].repeat(self.opt_size, 1, 1).to(
                device=shard.device, dtype=self.dtype).contiguous()
            scale = torch.stack([in_box[k, k] for k in range(3)])
            shard.compose_scale = None if torch.all(scale == 1) else (1.0 / scale).to(
                device=shard.device, dtype=self.dtype)

    def _gaussians(self, sigma):
        return {d: [gaussian_1d(s, truncated=2) for s in (torch.zeros(self.dims, device=d, dtype=self.dtype) + sigma)]
                for d in self.devices}

    def _begin_level(self, scale, iters):
        """Resample the state to a pyramid level and put the level's image slabs on the devices."""
        fixed_arrays, moving_arrays = self.fixed_images(), self.moving_images()
        for loss in [self.loss_fn, *self._losses.values()]:
            if hasattr(loss, "set_current_scale_and_iterations"):
                loss.set_current_scale_and_iterations(scale, iters)
        sizes = []
        for arrays in (fixed_arrays, moving_arrays):
            size = list(arrays.shape[2:])
            sizes.append([max(int(s / scale), MIN_IMG_SIZE) for s in size] if scale > 1 else size)

        self._cc_halo = self._losses[self.devices[0]].get_image_padding()
        # a step moves a point by at most lr/2 voxels along the longest axis
        self._compose_halo = math.ceil(self.optimizer_lr / 2) + 1
        self._resize_state(sizes[0], halo=max(self._cc_halo, self._grad_halo, self._warp_halo, self._compose_halo))
        self._prepare_compose(self._compose_halo)

        fixed_level = self._level_image(fixed_arrays, sizes[0], self.fixed_images.interpolate_mode, scale)
        self._cut_fixed(fixed_level, self._cc_halo)
        del fixed_level
        self._moving_level = self._level_image(moving_arrays, sizes[1], self.moving_images.interpolate_mode, scale)
        for shard in self._shards:
            self._cut_moving(shard, self._moving_level, self._affine, self.moving_margin)

    def _end_level(self):
        self._moving_level = None
        for shard in self._shards:
            shard.warp.grad = None
            shard.fixed = shard.fixed_mask = shard.moving = shard.moving_mask = None
        for device in self.devices:
            with torch.cuda.device(device):
                torch.cuda.empty_cache()

    def _prepare(self):
        fixed_t2p = self.fixed_images.get_torch2phy().to(self.dtype).cpu()
        moving_p2t = self.moving_images.get_phy2torch().to(self.dtype).cpu()
        self._affine = torch.matmul(moving_p2t, torch.matmul(self.affine, fixed_t2p)).to(self.dtype)  # [N, 4, 4]
        self._grad_gaussians = self._gaussians(self.smooth_grad_sigma) if self.smooth_grad_sigma > 0 else None
        self._warp_gaussians = self._gaussians(self.warp_sigma) if self.warp_sigma > 0 else None
        radius = lambda g: (len(g[self.devices[0]][0]) - 1) // 2 if g else 0
        self._grad_halo, self._warp_halo = radius(self._grad_gaussians), radius(self._warp_gaussians)

    def optimize(self):
        """Optimize the warp over the pyramid, following `GreedyRegistration.optimize`."""
        self._prepare()
        for scale, iters in zip(self.scales, self.iterations):
            self.convergence_monitor.reset()
            self._begin_level(scale, iters)
            best = self.best_iterate([s.warp for s in self._shards])
            scale_factor = 1 if self.reduction == "mean" else np.prod(
                [self.opt_size, self.fixed_images().shape[1], *self._level_size])
            pbar = tqdm(range(iters)) if self.progress_bar else range(iters)
            for i in pbar:
                for shard in self._shards:
                    shard.warp.grad = None
                self._refresh_moving(self._moving_level, self._affine)
                loss = self._loss_and_gradients(self._cc_halo)
                if self.progress_bar:
                    pbar.set_description("scale: {}, iter: {}/{}, loss: {:4f}".format(scale, i, iters, loss / scale_factor))
                if best is not None:
                    best.measured(loss)
                self._step((self._grad_halo, self._warp_halo, self._compose_halo))
                if self.convergence_monitor.converged(loss):
                    break
            if best is not None and best.restore():
                for shard in self._shards:
                    shard.exp_avg.zero_()
                    shard.exp_avg_sq.zero_()
                self.step_t = 0
            self._end_level()

    # ------------------------------------------------------------------ results

    def get_warp(self, device=None) -> torch.Tensor:
        """The full displacement field [N, Z, Y, X, 3] on `device`."""
        return self._gather("warp", self.output_device if device is None else device)

    def get_warp_parameters(self, fixed_images: Union[BatchedImages, FakeBatchedImages],
                            moving_images: Union[BatchedImages, FakeBatchedImages],
                            shape=None, displacement=False):
        """Affine and displacement field for `fireants_interpolator`, on `output_device`."""
        device = self.output_device
        shape = list(fixed_images.shape[2:]) if shape is None else list(shape)
        fixed_t2p = fixed_images.get_torch2phy().to(device=device, dtype=self.dtype)
        moving_p2t = moving_images.get_phy2torch().to(device=device, dtype=self.dtype)
        affine = torch.matmul(moving_p2t, torch.matmul(self.affine.to(device), fixed_t2p))[:, :-1].contiguous()
        warp_field = self.get_warp(device)
        if list(warp_field.shape[1:-1]) != shape:
            warp_field = F.interpolate(warp_field.permute(0, 4, 1, 2, 3), size=shape, mode="trilinear",
                                       align_corners=True).permute(0, 2, 3, 4, 1)
        return {"affine": affine, "grid": warp_field.contiguous()}

    def get_inverse_warp_parameters(self, fixed_images, moving_images, shape=None, **kwargs):
        raise NotImplementedError("invert the gathered field with fireants.utils.warputils.compositive_warp_inverse")
