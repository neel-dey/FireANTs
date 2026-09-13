# Block-matching affine registration

`BlockMatchingRegistration` estimates local scalar-image correspondences with
absolute normalized cross-correlation, then fits an affine using least trimmed
squares. It supports 2D and 3D images and exports the same physical fixed-to-moving
matrix format as `AffineRegistration`.

```python
from fireants.registration.block_matching import BlockMatchingRegistration

registration = BlockMatchingRegistration(
    scales=[4, 2, 1],
    iterations=[20, 10, 10],
    fixed_images=fixed,
    moving_images=moving,
    init_affine=rigid_matrix,
    backend="torch",
)
registration.optimize()
matrix = registration.get_affine_matrix()
moved = registration.evaluate(fixed, moving)
```

Use `backend="torch"` for the pure-Torch implementation on CPU or GPU. It does
not require compiled operations. Use `backend="cuda"` with CUDA images and an
updated `fireants_fused_ops` build for native kernels. Both dimensions have CUDA
block variance, block search, and affine least-squares fitting; neither uses a
CPU least-squares fallback. Pyramid filtering, resampling, and inlier selection
use Torch operations on the selected device. No NiftyReg installation or
runtime library is required.

The input must have one scalar channel. With `masked=True`, the last of two
channels is the mask, using values greater than 0.5 as foreground. Masking
selects reference samples for each matching direction; warped samples outside
the image are excluded. Constant patches and patches with insufficient valid
samples do not produce correspondences. Multi-channel descriptors are not
accepted by this implementation.

Blocks have width 4 and use integer search offsets from -3 to 3 voxels per axis.
`block_fraction=0.5` selects up to half of all grid blocks, ordered by variance,
subject to more than half of each block's samples being valid.
`inlier_fraction=0.5` retains the half of correspondences with smallest residuals
during robust fitting. Both fractions can be changed. `symmetric=True` fits both
directions and averages each matrix with the inverse estimate from the other
direction.

Least squares uses centered, normalized coordinates and float64 accumulation
to reduce numerical error from large physical coordinates. Invalid or rank-deficient fits
leave the last valid matrix in place for that pyramid level. `history` records
completed iterations and the reason for any rejected update. These checks do
not guarantee anatomical accuracy.

This is an affine refinement method with a finite local search range. Supply a
reasonable rigid initialization when the images have substantial displacement.
It follows the block-matching and least-trimmed-squares approach described by
[Modat et al.](https://doi.org/10.1117/1.JMI.1.2.024003), with FireANTs' Torch image
pyramid and independently implemented matching and fitting code.
