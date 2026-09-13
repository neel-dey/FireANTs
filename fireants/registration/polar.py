# Distributed under the FireANTs License version 1.0; see LICENSE.

"""Rotation and symmetric log-stretch parameters for an affine linear part."""

import math

import torch
from torch import nn


class PolarLinear(nn.Module):
    """Represent an invertible matrix as R Q exp(S), with S symmetric.

    Q is the initial orthogonal polar factor, including any reflection. R is
    an optimized rotation initialized to identity. The determinant sign is fixed.
    """

    def __init__(self, linear, scale_bounds=None):
        super().__init__()
        batch, dims, _ = linear.shape
        if dims not in (2, 3):
            raise ValueError("Polar affine supports only 2D and 3D")
        self.dims = dims
        # Matrix decompositions and exponentials require at least float32 here.
        work_dtype = torch.float64 if linear.dtype == torch.float64 else torch.float32
        with torch.no_grad():
            initial = linear.detach().to(work_dtype)
            if not torch.isfinite(initial).all():
                raise ValueError("Polar affine initialization must be finite")
            u, scales, vh = torch.linalg.svd(initial)
            if (scales <= torch.finfo(work_dtype).eps * scales[:, :1]).any():
                raise ValueError("Polar affine initialization must be nonsingular")
            if scale_bounds is not None:
                scales = scales.clamp(*scale_bounds)
            self.register_buffer("orthogonal", u @ vh)
            basis = torch.zeros(dims * (dims + 1) // 2, dims, dims,
                                device=linear.device, dtype=work_dtype)
            for i in range(dims):
                basis[i, i, i] = 1
            k = dims
            for i in range(dims):
                for j in range(i + 1, dims):
                    basis[k, i, j] = basis[k, j, i] = 1 / math.sqrt(2)
                    k += 1
            self.register_buffer("basis", basis)
            symmetric = vh.mT @ torch.diag_embed(scales.log()) @ vh
            stretch = torch.einsum("bij,kij->bk", symmetric, basis)
        self.log_stretch = nn.Parameter(stretch.to(linear.dtype))
        rotation = linear.new_zeros(batch, 1 if dims == 2 else 4)
        if dims == 3:
            rotation[:, 0] = 1
        self.rotation = nn.Parameter(rotation)

    def symmetric(self):
        """Return S in the precision used for the matrix exponential."""
        return torch.einsum("bk,kij->bij", self.log_stretch.to(self.basis.dtype), self.basis)

    def forward(self):
        if self.dims == 2:
            angle = self.rotation[:, 0].to(self.basis.dtype)
            c, s = angle.cos(), angle.sin()
            rotation = torch.stack((c, -s, s, c), dim=-1).reshape(-1, 2, 2)
        else:
            q = self.rotation.to(self.basis.dtype)
            norm = torch.linalg.vector_norm(q, dim=-1, keepdim=True)
            identity = torch.zeros_like(q)
            identity[:, 0] = 1
            tiny = torch.finfo(q.dtype).tiny
            q = torch.where(norm > tiny, q / norm.clamp_min(tiny), identity)
            w, x, y, z = q.unbind(-1)
            rotation = torch.stack((
                1 - 2 * (y*y + z*z), 2 * (x*y - w*z), 2 * (x*z + w*y),
                2 * (x*y + w*z), 1 - 2 * (x*x + z*z), 2 * (y*z - w*x),
                2 * (x*z - w*y), 2 * (y*z + w*x), 1 - 2 * (x*x + y*y),
            ), dim=-1).reshape(-1, 3, 3)
        linear = rotation @ self.orthogonal @ torch.linalg.matrix_exp(self.symmetric())
        return linear.to(self.log_stretch.dtype)

    @torch.no_grad()
    def project_scale(self, bounds):
        """Bound principal stretches by clipping the eigenvalues of S."""
        low, high = (math.log(value) for value in bounds)
        values, vectors = torch.linalg.eigh(self.symmetric())
        clamped = values.clamp(low, high)
        changed = (values != clamped).any(dim=-1)
        if not changed.any():
            return
        symmetric = vectors @ torch.diag_embed(clamped) @ vectors.mT
        parameters = torch.einsum("bij,kij->bk", symmetric, self.basis)
        self.log_stretch.copy_(torch.where(changed[:, None], parameters.to(self.log_stretch.dtype),
                                           self.log_stretch))
