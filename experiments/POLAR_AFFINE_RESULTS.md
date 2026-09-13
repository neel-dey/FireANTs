# Polar affine experiments, 2026-09-13

Polar affine refinement improved synthetic transform recovery and real abdominal
alignment, but it did not consistently outperform direct matrix optimization
with the same settings. Parameterization alone did not fix the native brain
failure. The NCC numerator smoothing was a larger source of instability in that
case. The implementation remains opt-in; neither the default parameterization
nor the default loss changed.

The branch is `experiment/polar-affine`, based on `c72d2ef`. It adds
`AffineRegistration(parameterization="polar")`, using an incremental rotation,
a fixed initial orthogonal factor, and a symmetric matrix exponential. It
supports 2D angles, 3D normalized quaternions, reflected initializations,
translation normalization, the existing principal stretch bounds, best-iterate
restoration, and physical matrix export. Initialization is detached and copied
so centering does not modify a caller's tensor.

**Primary results**

These comparisons use affine learning rate `0.003`, bounds `[0.75, 4/3]`,
normalized translation, centering, and best-iterate restoration. Each matrix
and polar comparison starts from the same rigid matrix. The brain result with
`smooth_nr=0` changes only the affine loss setting; the rigid algorithm and its
settings remain the same and are recomputed for that suite.

| Evaluation | Rigid | Matrix affine | Polar affine |
|---|---:|---:|---:|
| 8 synthetic 2D cases, mean point error in mm | 3.1653 | 0.0086 | 0.0086 |
| 8 synthetic 3D cases, mean point error in mm | 3.8419 | 0.0271 | 0.0268 |
| 8 abdominal MR/CT pairs, native 2 mm, mean organ Dice | 0.6036 | 0.7692 | 0.7695 |
| 3 brain pairs, native 1 mm, default NCC, mean brain-mask Dice | 0.9475 | 0.8798 | 0.8618 |
| Same brain cases, affine NCC `smooth_nr=0`, mean brain-mask Dice | 0.9475 | 0.9608 | 0.9608 |

Both affine parameterizations improved organ Dice over rigid registration on
all eight abdominal pairs. Per-pair native-resolution Dice:

| Pair | Rigid | Matrix affine | Polar affine |
|---|---:|---:|---:|
| 0001 | 0.4921 | 0.6959 | 0.6888 |
| 0002 | 0.6796 | 0.7436 | 0.7521 |
| 0003 | 0.6426 | 0.6999 | 0.6998 |
| 0004 | 0.7898 | 0.8506 | 0.8506 |
| 0005 | 0.4996 | 0.7940 | 0.7952 |
| 0006 | 0.4720 | 0.7203 | 0.7203 |
| 0007 | 0.5227 | 0.7808 | 0.7807 |
| 0008 | 0.7302 | 0.8681 | 0.8681 |

The longitudinal brain cases also have landmarks that were excluded from
optimization. Mean landmark errors in mm with affine `smooth_nr=0`:

| Case | Identity | Rigid | Matrix affine | Polar affine |
|---|---:|---:|---:|---:|
| BraTSReg 021 | 2.470 | 2.893 | 2.739 | 2.737 |
| BraTSReg 127 | 19.007 | 8.165 | 7.998 | 7.997 |

Refinement improves mean landmark error over rigid registration here, but does
not beat identity on case 021. Case 021's median landmark error also increases
slightly, from 2.593 mm after rigid registration to 2.601 mm after polar affine.
Brain-mask Dice alone is not enough to assess local anatomical alignment.

**The native brain failure**

For the inter-subject pair, the default NCC loss preferred an incorrect affine:

| Affine loss setting | Rigid Dice | Matrix Dice | Polar Dice |
|---|---:|---:|---:|
| Default numerator smoothing | 0.9151 | 0.7044 | 0.6502 |
| `smooth_nr=0` | 0.9151 | 0.9459 | 0.9459 |

The default normalized-window NCC adds `1e-5` to its numerator and denominator.
When the moving patch is constant, its covariance is zero, but numerator
smoothing still gives it a high similarity score. A deterministic probe with a
random `[0, 1]` fixed volume and a zero moving volume gives:

| NCC setting | Perfect match loss | Flat moving image loss |
|---|---:|---:|
| `smooth_nr=1e-5` | -1.0000 | -0.9172 |
| `smooth_nr=0` | -0.9987 | 0.0000 |

Lower loss is better. The existing `smooth_nr=0` option removes this reward
without changing denominator regularization. The loss implementation itself
was not modified. The effect depends on intensity scale and local variance;
this is evidence about these normalized inputs, not a universal loss default.

At the stress learning rate `0.1`, polar affine still failed the inter-subject
case even with `smooth_nr=0` and stretch bounds: Dice was 0.006. Bounds constrain
scaling, not translation or overlap. With rates `0.001` through `0.03`, both
parameterizations aligned all three brain cases with `smooth_nr=0` consistently.

A separate run from field-of-view centers reproduced severe collapse with
legacy matrix settings: Dice 0.316 and determinant `9.05e-5` at rate `0.01`.
Polar parameterization alone also failed there, with Dice 0.355. This reproduces
the failure qualitatively; it does not reproduce the old registerio pipeline
or its preprocessing bit for bit.

**Method and limits**

- Hardware: the two local RTX PRO 6000 Blackwell GPUs on laplace. PyTorch
  `2.13.0+cu130`, float32, PyTorch interpolation. The display GPU was excluded.
- Synthetic cases: analytically sampled asymmetric Gaussian mixtures with known
  rotation, stretch, shear, and translation. Fixed and moving arrays are sampled
  independently from the continuous phantom. Evaluation uses 2,000 physical
  points per case and no fitted image similarity. All generated target stretches
  were within the tested bounds.
- Brains: inter-subject BraTSReg 021 to 127 and the longitudinal T1 pairs for
  subjects 021 and 127, found under `reference-runs/brats` and `brats127`.
  Intensities are clipped and normalized using foreground percentiles. Masks
  participate in the masked NCC objective; their Dice is not a held-out metric.
  Longitudinal landmarks use physical LPS coordinates and do not participate in
  optimization.
- Abdomens: MR/CT pairs 0001 through 0008 from registerio's regression manifest.
  Body masks, CT window `[-450, 450]`, MR window `[0, 20000]`, and registerio's
  MIND-SSC descriptors with radius 1 and dilation 2. The objective is masked NCC
  on descriptors. Organ labels are used only for evaluation. Dice is computed
  per shared nonzero label, averaged per pair, then averaged across pairs.
- Real label evaluation uses SimpleITK nearest-neighbor resampling on the
  original fixed label grid. It does not use FireANTs' evaluation sampler.
- Rigid and affine each use scales `[4, 2, 1]`, 100 iterations per scale, Adam,
  centering, normalized translation, and best-iterate restoration. Rigid rate
  is `0.003`. Early stopping is disabled using `tolerance=inf`. The standard
  best-iterate implementation discards the final unmeasured update at each scale.
- Affine sweeps use rates `[0.001, 0.003, 0.01, 0.03, 0.1]`, with and without
  bounds `[0.75, 4/3]`. Native abdominal confirmation uses the fixed `0.003` rate
  and bounds. Brain kernels are `[9, 7, 5]` at 1 mm; abdominal kernels are
  `[13, 9, 7]`. Smaller preliminary suites use the settings in their JSON files.
- There are 839 recorded affine fits, excluding preliminary runs. The additional
  unmasked abdominal descriptor-MSE experiment often worsened alignment;
  changing parameterization did not repair that objective. Results are retained.
- No deformable stage was run. These measurements assess affine refinement,
  not the effect on a subsequent deformable registration. This is a small
  convenience sample, not evidence of general clinical performance.

At rate `0.003`, mean optimization time was 3.28 seconds for polar versus
2.68 seconds for matrix on native brains with `smooth_nr=0`, and 8.19 versus
7.55 seconds on native abdomens. Timings exclude feature extraction,
initialization, loading, and evaluation.

**Validation and artifacts**

Thirty-six tests passed: 30 new polar tests, four existing translation tests,
and the existing 2D and 3D affine keypoint tests. The new checks include numerical
gradients at identity, independent SciPy matrix-exponential agreement, fitting
known transforms, reflections, batch broadcasting, physical units, float
precisions, scale projection, best-iterate restoration, and SimpleITK transform
export. Three existing OASIS affine tests stopped during setup because
`tests/test_data/oasis_157_image.nii.gz` is absent. They were not changed or
reported as passes. Compilation and `git diff --check` passed.

- [All aggregate measurements](results/polar_affine/summary.csv)
- [Learning-rate curves](results/polar_affine/learning_rates.png)
- [Native brain failure contours](results/polar_affine/brains_native_alignment.png)
- [Native brain contours with `smooth_nr=0`](results/polar_affine/brains_native_nr0_alignment.png)
- [Native abdominal contours](results/polar_affine/abdomens_native_alignment.png)
- [Raw results and configurations](results/polar_affine/)

**Reproduction**

Run from the FireANTs root in an environment containing both FireANTs and
registerio dependencies. `PYTHONPATH=.:..` selects the checkout and the sibling
registerio package. Set `CUDA_VISIBLE_DEVICES` to a compute GPU UUID. These
commands write to a new output directory and leave the recorded results intact.

```bash
export PYTHONPATH=.:..
export CUDA_VISIBLE_DEVICES=GPU-d5542ac3-f909-10f9-af1e-e31f8e3eea66

python experiments/polar_affine.py --suite synthetic --seeds 0 1 2 3 4 5 6 7 \
  --output tests/test_results/polar_affine_rerun/synthetic.jsonl

python experiments/polar_affine.py --suite brains --masked --spacing 1 \
  --kernel 9 7 5 --data-root ../../reference-runs \
  --output tests/test_results/polar_affine_rerun/brains_native.jsonl

python experiments/polar_affine.py --suite brains --masked --spacing 1 \
  --kernel 9 7 5 --smooth-nr 0 --data-root ../../reference-runs \
  --output tests/test_results/polar_affine_rerun/brains_native_nr0.jsonl

python experiments/polar_affine.py --suite abdomens --masked --spacing 3 \
  --kernel 13 9 7 --data-root ../../reference-runs \
  --output tests/test_results/polar_affine_rerun/abdomens_masked.jsonl

python experiments/polar_affine.py --suite abdomens --masked --spacing 2 \
  --kernel 13 9 7 --rates .003 --bounds bounded --data-root ../../reference-runs \
  --output tests/test_results/polar_affine_rerun/abdomens_native.jsonl

python experiments/summarize_polar_affine.py \
  tests/test_results/polar_affine_rerun/synthetic.jsonl \
  tests/test_results/polar_affine_rerun/brains_native.jsonl \
  tests/test_results/polar_affine_rerun/brains_native_nr0.jsonl \
  tests/test_results/polar_affine_rerun/abdomens_masked.jsonl \
  --output-dir tests/test_results/polar_affine_rerun/summary
```

The environment used here was
`/autofs/cluster/dalcalab2/users/nd480/.local/share/mamba/envs/try-general-reg/bin/python`.
GPU UUID `GPU-7429bfab-8a89-dc43-99bf-b63d4470f962` was used for the abdominal
runs while the first compute GPU ran synthetic and brain experiments. CUDA
sampling gradients may vary slightly between runs; the raw values are not
bitwise reproducibility targets.
