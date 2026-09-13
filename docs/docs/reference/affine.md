# Affine Image Matching

FireANTs supports multi-scale affine image matching between two images. 
Affine transformations are more flexible than rigid transformations, as they allow for scaling, rotation, and shearing.

**Initialization:** You can pass `init_rigid="cof"` to set the initial translation to \(c_m - c_f\) (center of moving minus center of fixed) and the linear part to identity. Pass a tensor (e.g. from rigid or moment matching) to use a custom initial affine.

Set `parameterization="polar"` to optimize rotation and symmetric log-stretch:

\[
y = R Q \exp(S)(x-c) + c + r u, \qquad S=S^T.
\]

Here `Q` is the initial orthogonal polar factor, `R` is an incremental rotation,
and `exp` is the matrix exponential. In 2D, rotation uses one angle and stretch
uses three parameters. In 3D, rotation uses a normalized quaternion and stretch
uses six parameters. Any reflection in the initialization stays in `Q`; its
determinant sign is preserved. Initialization must be numerically nonsingular.

With `around_center=True`, `c` is the fixed-image center. With
`normalize_translation=True`, `r` is the RMS physical half-extent of the fixed
image and `u` is dimensionless translation. Otherwise translation uses physical
units. For example, to refine an existing rigid registration:

```python
affine = AffineRegistration(
    scales=[4, 2, 1],
    iterations=[100, 100, 100],
    fixed_images=fixed,
    moving_images=moving,
    loss_type="cc",
    loss_params={"smooth_nr": 0.0},
    init_rigid=rigid.get_rigid_matrix().detach(),
    parameterization="polar",
    normalize_translation=True,
    optimizer_lr=0.003,
    keep_best=True,
    scale_bounds=(0.75, 4 / 3),
)
affine.optimize()
matrix = affine.get_affine_matrix()
```

The example disables NCC numerator smoothing. With intensities normalized to
`[0, 1]`, the default smoothing can assign high similarity to constant warped
regions. This caused affine collapse in the native-resolution brain experiments
with both parameterizations. Setting `smooth_nr=0` removed that failure at
learning rate `0.003` in the tested cases; it does not change the denominator
regularization. Large learning rates still failed some cases. This is a loss
setting, separate from the affine parameterization.

These scale bounds are an example; choose bounds appropriate for the expected
anatomical size differences. In polar mode they constrain the eigenvalues of
`S` to the logarithms of the requested bounds. This limits principal stretches
and condition number, but does not guarantee better anatomical alignment.
Unbounded polar transforms can still approach singularity, and large parameter
values can exceed floating-point range. `keep_best` selects by image loss,
which can favor an anatomically incorrect transform.

The default remains `parameterization="matrix"`. Both modes return physical
matrices through `get_affine_matrix()` and use the same warping and export APIs.

::: fireants.registration.affine.AffineRegistration
