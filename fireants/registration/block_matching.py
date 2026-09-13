"""Affine registration from local NCC correspondences and least trimmed squares."""

import itertools
import math

import torch
from torch.nn import functional as F
from tqdm import tqdm

from fireants.losses.cc import gaussian_1d
from fireants.registration.affine import AffineRegistration
from fireants.utils.globals import MIN_IMG_SIZE
from fireants.utils.imageutils import downsample


def _extension():
    try:
        import fireants_fused_ops as ops
        if not hasattr(ops, "block_matching_ncc"):
            raise ImportError("Block-matching kernels are missing")
    except ImportError as error:
        raise RuntimeError("Build fused_ops to use backend='cuda' for block matching") from error
    return ops


def _check_image(image, mask, backend):
    if image.ndim not in (2, 3) or image.dtype not in (torch.float32, torch.float64):
        raise ValueError("Expected a scalar 2D/3D float32 or float64 image")
    if mask.shape != image.shape or mask.device != image.device or mask.dtype != torch.bool:
        raise ValueError("The mask must be boolean with the image's shape and device")
    if backend not in ("torch", "cuda"):
        raise ValueError("backend must be 'torch' or 'cuda'")
    if backend == "cuda" and not image.is_cuda:
        raise ValueError("backend='cuda' requires CUDA tensors")


def _offsets(width, dims, device):
    return torch.tensor(list(itertools.product(range(width), repeat=dims)),
                        dtype=torch.long, device=device).flip(-1)


def _gather(image, points):
    """Gather clipped integer xyz coordinates and return an in-bounds mask."""
    sizes = torch.tensor(image.shape[::-1], device=image.device)
    valid = ((points >= 0) & (points < sizes)).all(-1)
    clipped = points.clamp_min(0).minimum(sizes - 1)
    index = clipped[..., -1]
    for axis in range(image.ndim - 2, -1, -1):
        index = index * sizes[axis] + clipped[..., axis]
    return image.reshape(-1)[index], valid


@torch.no_grad()
def select_blocks(image, mask, fraction=.5, backend="torch"):
    """Select up to fraction of all grid blocks, ordered by valid-sample variance."""
    _check_image(image, mask, backend)
    if not 0 < fraction <= 1:
        raise ValueError("fraction must be in (0, 1]")
    dims = image.ndim
    axes = [torch.arange(0, n, 4, device=image.device) for n in image.shape]
    origins = torch.stack(torch.meshgrid(*axes, indexing="ij"), -1).reshape(-1, dims).flip(-1).contiguous()
    if backend == "cuda":
        variance = _extension().block_matching_variance(image.contiguous(), mask.contiguous(), origins)
    else:
        values, inside = _gather(image, origins[:, None] + _offsets(4, dims, image.device))
        mask_values, _ = _gather(mask, origins[:, None] + _offsets(4, dims, image.device))
        valid = inside & mask_values & torch.isfinite(values)
        count = valid.sum(-1)
        values = torch.where(valid, values.double(), 0.)
        mean = values.sum(-1) / count.clamp_min(1)
        variance = ((values - mean[:, None]).square() * valid).sum(-1) / count.clamp_min(1)
        variance = torch.where((count > 4**dims / 2) & (variance > 1e-12), variance, -1.)
    eligible = torch.nonzero(variance >= 0).flatten()
    count = min(len(eligible), math.ceil(len(origins) * fraction))
    order = torch.argsort(variance[eligible], descending=True, stable=True)[:count]
    return origins[eligible[order]].contiguous()


@torch.no_grad()
def match_blocks(reference, warped, mask, origins, backend="torch"):
    """Find absolute NCC maxima over integer offsets in [-3, 3] for each block.

    Return xyz displacements and scores. Unmatched blocks have score -1.
    Scores are rounded to 1e-10; ties choose the first offset in z, y, x order.
    """
    _check_image(reference, mask, backend)
    if warped.shape != reference.shape or warped.dtype != reference.dtype or warped.device != reference.device:
        raise ValueError("Reference and warped images must have matching shape, dtype, and device")
    if origins.ndim != 2 or origins.shape[1] != reference.ndim or origins.dtype != torch.long or origins.device != reference.device:
        raise ValueError("origins must contain integer xyz coordinates on the image device")
    if backend == "cuda":
        return _extension().block_matching_ncc(reference.contiguous(), warped.contiguous(),
                                              mask.contiguous(), origins.contiguous())
    dims = reference.ndim
    offsets = _offsets(4, dims, reference.device)
    shifts = _offsets(7, dims, reference.device) - 3
    displacement = torch.zeros_like(origins)
    scores = reference.new_full((len(origins),), -1., dtype=torch.float64)
    # Limit intermediate storage independently of image size.
    for start in range(0, len(origins), 512):
        points = origins[start:start + 512, None] + offsets
        fixed, inside = _gather(reference, points)
        mask_values, _ = _gather(mask, points)
        fixed_valid = inside & mask_values & torch.isfinite(fixed)
        best = scores[start:start + len(points)]
        best_shift = displacement[start:start + len(points)]
        for offset_start in range(0, len(shifts), 16):
            candidates = shifts[offset_start:offset_start + 16]
            moving, moving_inside = _gather(warped, points[:, None] + candidates[None, :, None])
            valid = fixed_valid[:, None] & moving_inside & torch.isfinite(moving)
            count = valid.sum(-1)
            x = torch.where(valid, fixed[:, None].double(), 0.)
            y = torch.where(valid, moving.double(), 0.)
            mean_x = x.sum(-1) / count.clamp_min(1)
            mean_y = y.sum(-1) / count.clamp_min(1)
            x = torch.where(valid, x - mean_x[..., None], 0.)
            y = torch.where(valid, y - mean_y[..., None], 0.)
            vx, vy = x.square().sum(-1), y.square().sum(-1)
            correlation = (x * y).sum(-1).abs() / (vx * vy).sqrt().clamp_min(1e-30)
            correlation = torch.where((count > 4**dims / 2) & (vx > 1e-12) & (vy > 1e-12), correlation, -1.)
            correlation = torch.round(correlation * 1e10) / 1e10
            value, index = correlation.max(-1)
            improved = value > best
            best_shift.copy_(torch.where(improved[:, None], candidates[index], best_shift))
            best.copy_(torch.where(improved, value, best))
    return displacement, scores


@torch.no_grad()
def fit_affine(source, target, backend="torch"):
    """Fit a physical affine with centered, normalized coordinates in float64.

    Both implementations solve the same small normal system on the input
    device. Rank-deficient correspondences raise ValueError.
    """
    if source.ndim != 2 or source.shape[1] not in (2, 3) or source.shape != target.shape:
        raise ValueError("Expected corresponding [N, 2] or [N, 3] point arrays")
    if source.device != target.device or source.dtype not in (torch.float32, torch.float64) or target.dtype != source.dtype:
        raise ValueError("Point arrays must share a float32/float64 dtype and device")
    if backend not in ("torch", "cuda") or backend == "cuda" and not source.is_cuda:
        raise ValueError("Use backend='torch', or backend='cuda' with CUDA points")
    dims = source.shape[1]
    if len(source) < dims + 1 or not (torch.isfinite(source).all() & torch.isfinite(target).all()):
        raise ValueError("Insufficient or nonfinite affine correspondences")
    x, y = source.double(), target.double()
    center_x, center_y = x.mean(0), y.mean(0)
    scale = (x - center_x).square().mean().sqrt().clamp_min(1e-12)
    x = ((x - center_x) / scale).contiguous()
    y = (y - center_y).contiguous()
    if backend == "cuda":
        coefficients = _extension().block_matching_lsq(x, y)
    else:
        design = torch.cat([x, torch.ones_like(x[:, :1])], 1)
        gram = design.T @ design
        eigenvalues = torch.linalg.eigvalsh(gram)
        if eigenvalues[0] <= 1e-10 * eigenvalues[-1]:
            raise ValueError("Rank-deficient affine correspondences")
        coefficients = torch.linalg.solve(gram, design.T @ y).T
    if not torch.isfinite(coefficients).all():
        raise ValueError("Rank-deficient affine correspondences")
    matrix = torch.eye(dims + 1, dtype=torch.float64, device=source.device)
    matrix[:dims, :dims] = coefficients[:, :dims] / scale
    matrix[:dims, -1] = center_y + coefficients[:, -1] - matrix[:dims, :dims] @ center_x
    return matrix.to(source.dtype)


@torch.no_grad()
def fit_affine_lts(source, target, fraction=.5, iterations=20, backend="torch"):
    """Refit the closest correspondence fraction until the trimmed error stops decreasing."""
    if not 0 < fraction <= 1 or iterations < 1:
        raise ValueError("Use fraction in (0, 1] and a positive iteration count")
    matrix = fit_affine(source, target, backend)
    count = max(source.shape[1] + 1, math.ceil(len(source) * fraction))
    best_error = float("inf")
    for _ in range(iterations):
        residual = (source @ matrix[:-1, :-1].T + matrix[:-1, -1] - target).square().sum(-1)
        indices = torch.argsort(residual, stable=True)[:count]
        error = residual[indices].mean()
        if error >= best_error:
            break
        best_error = error
        try:
            updated = fit_affine(source[indices], target[indices], backend)
        except ValueError:
            break
        if torch.allclose(updated, matrix, atol=1e-6, rtol=0):
            matrix = updated
            break
        matrix = updated
    return matrix


class BlockMatchingRegistration(AffineRegistration):
    """Refine an affine using scalar block NCC and least trimmed squares.

    Images contain one scalar channel, or scalar plus mask when masked=True.
    backend='torch' needs no extension and works on CPU or CUDA. backend='cuda'
    uses fused_ops kernels for block variance, matching, and the 2D/3D affine
    solve. Resampling, pyramids, and inlier selection use Torch on that device.
    """

    def __init__(self, scales, iterations, fixed_images, moving_images, init_affine=None,
                 backend="torch", masked=False, block_fraction=.5, inlier_fraction=.5,
                 symmetric=True, dtype=torch.float32, progress_bar=True,
                 allow_repeated_scales=False):
        super().__init__(scales, iterations, fixed_images, moving_images, init_rigid=init_affine,
                         loss_type="noop", around_center=False, dtype=dtype,
                         progress_bar=progress_bar, allow_repeated_scales=allow_repeated_scales)
        self.masked = masked
        self.backend = backend
        self.block_fraction = block_fraction
        self.inlier_fraction = inlier_fraction
        self.symmetric = symmetric
        for image in (fixed_images(), moving_images()):
            if image.shape[1] != (2 if masked else 1):
                raise ValueError("Block matching requires one scalar channel and an optional last-channel mask")
            _check_image(image[0, 0], torch.ones_like(image[0, 0], dtype=torch.bool), backend)
        if not 0 < block_fraction <= 1 or not 0 < inlier_fraction <= 1:
            raise ValueError("Block and inlier fractions must be in (0, 1]")
        self.affine.requires_grad_(False)
        self.history = []

    def _level(self, arrays, scale):
        arrays = arrays.to(self.dtype)
        if scale == 1:
            return arrays
        size = [max(MIN_IMG_SIZE, int(n / scale)) for n in arrays.shape[2:]]
        sigma = [.5 * n / s for n, s in zip(arrays.shape[2:], size)]
        kernels = [gaussian_1d(torch.tensor(s, device=arrays.device), truncated=2) for s in sigma]
        image, mask = self._split_image_and_mask_last_channel(arrays)
        mode = self.fixed_images.interpolate_mode
        image = downsample(image, size, mode, gaussians=kernels, use_fft=False)
        if mask is not None:
            mask = F.interpolate(mask, size=size, mode=mode, align_corners=True)
        return self._concat_image_and_mask_last_channel(image, mask)

    def _correspondences(self, reference, moving, reference_t2p, moving_p2t, matrix, origins):
        dims = self.dims
        theta = (moving_p2t @ matrix @ reference_t2p)[:dims]
        shape = (1, 1, *reference.shape[1:])
        grid = F.affine_grid(theta[None], shape, align_corners=True)
        warped = F.grid_sample(moving[None, :1], grid, align_corners=True)[0, 0]
        warped = warped.masked_fill((grid[0].abs() > 1).any(-1), float("nan"))
        mask = reference[-1] > .5 if self.masked else torch.ones_like(reference[0], dtype=torch.bool)
        shifts, scores = match_blocks(reference[0], warped, mask, origins, self.backend)
        valid = scores > 0
        source = origins[valid].to(matrix.dtype)
        target = source + shifts[valid]
        size = torch.tensor(reference.shape[1:][::-1], device=source.device)
        source = 2 * source / (size - 1) - 1
        target = 2 * target / (size - 1) - 1
        source = source @ reference_t2p[:dims, :dims].T + reference_t2p[:dims, -1]
        target = target @ reference_t2p[:dims, :dims].T + reference_t2p[:dims, -1]
        target = target @ matrix[:dims, :dims].T + matrix[:dims, -1]
        return source, target

    @torch.no_grad()
    def optimize(self):
        fixed_t2p = self.fixed_images.get_torch2phy().to(self.dtype)
        moving_t2p = self.moving_images.get_torch2phy().to(self.dtype)
        fixed_p2t, moving_p2t = torch.linalg.inv(fixed_t2p), torch.linalg.inv(moving_t2p)
        self.history = []
        for scale, iterations in zip(self.scales, self.iterations):
            fixed = self._level(self.fixed_images(), scale)
            moving = self._level(self.moving_images(), scale)
            for batch in range(self.opt_size):
                fi, mi = min(batch, len(fixed) - 1), min(batch, len(moving) - 1)
                ref, mov = fixed[fi], moving[mi]
                mask = ref[-1] > .5 if self.masked else torch.ones_like(ref[0], dtype=torch.bool)
                origins = select_blocks(ref[0], mask, self.block_fraction, self.backend)
                if self.symmetric:
                    mask = mov[-1] > .5 if self.masked else torch.ones_like(mov[0], dtype=torch.bool)
                    reverse_origins = select_blocks(mov[0], mask, self.block_fraction, self.backend)
                matrix = self.get_affine_matrix()[batch].clone()
                reverse = torch.linalg.inv(matrix)
                completed = 0
                for _ in tqdm(range(iterations), disable=not self.progress_bar,
                              desc=f"Block matching, scale {scale}, pair {batch}"):
                    try:
                        x, y = self._correspondences(ref, mov, fixed_t2p[fi], moving_p2t[mi], matrix, origins)
                        updated = fit_affine_lts(x, y, self.inlier_fraction, backend=self.backend)
                        if self.symmetric:
                            bx, by = self._correspondences(mov, ref, moving_t2p[mi], fixed_p2t[fi], reverse, reverse_origins)
                            updated_reverse = fit_affine_lts(bx, by, self.inlier_fraction, backend=self.backend)
                            forward = .5 * (updated + torch.linalg.inv(updated_reverse))
                            backward = .5 * (updated_reverse + torch.linalg.inv(updated))
                        else:
                            forward, backward = updated, torch.linalg.inv(updated)
                        if not torch.isfinite(forward).all() or torch.linalg.det(forward[:-1, :-1]) * torch.linalg.det(matrix[:-1, :-1]) <= 0:
                            raise ValueError("Invalid affine update")
                    except (ValueError, torch.linalg.LinAlgError) as error:
                        self.history.append(dict(scale=scale, batch=batch, iterations=completed, reason=str(error)))
                        break
                    matrix, reverse = forward, backward
                    completed += 1
                else:
                    self.history.append(dict(scale=scale, batch=batch, iterations=completed, reason="completed"))
                self.affine[batch].copy_(matrix[:-1])
