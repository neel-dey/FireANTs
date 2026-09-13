# FireANTs Torch and CUDA block matching, 2026-09-13

The independent block matcher improved all 16 synthetic cases, all three brain
pairs, and all eight MR–CT abdominal pairs from the recorded rigid transforms.
Torch and CUDA produced identical exported matrices on all 27 cases. All
registration levels completed without a rejected fit. These measurements
support affine refinement on this sample; they do not establish that it will
improve every registration or the subsequent deformable stage.

`BlockMatchingRegistration` now provides both implementations in FireANTs.
Neither requires NiftyReg. The pure-Torch implementation works on CPU and GPU
without compiled operations. The CUDA implementation includes native kernels
for block variance, NCC matching, and least-squares fitting in both 2D and 3D.
Coordinate normalization, inlier selection, image pyramids, and resampling use
Torch on the selected device. There is no CPU least-squares fallback.

## Measurements

Both backends use scalar intensities, the same image pyramids, and the exact
same recorded rigid matrices. The L-BFGS column comes from the
[previous solver experiment](AFFINE_SOLVERS_RESULTS.md), using step size 1
and `smooth_nr=0` on real images.

| Evaluation | Rigid | L-BFGS, step 1 | Block matching, Torch and CUDA |
|---|---:|---:|---:|
| 8 synthetic 2D cases, mean point error, mm | 3.1653 | 0.0085 | 0.3188 |
| 8 synthetic 3D cases, mean point error, mm | 3.8419 | 0.0268 | 0.4685 |
| 3 brains, mean brain-mask Dice | 0.9475 | 0.9610 | 0.9590 |
| 8 abdomens, mean organ Dice | 0.6036 | 0.7564 | 0.7731 |

L-BFGS is substantially more accurate on these synthetic cases and slightly
better on brain-mask Dice. Block matching gives better mean abdominal Dice,
but not on every pair. This is a comparison of complete algorithms: abdominal
L-BFGS uses MIND descriptors, while block matching uses scalar intensities.
Tuned Adam with numerator smoothing previously reached abdominal Dice 0.7685
without scale bounds. Neither alternative is uniformly better than Adam.

The L-BFGS failure at step size 3 is a stress test with an unusually large
hyperparameter. It is not a reason to reject L-BFGS at ordinary settings.
The separate failures with default numerator smoothing at step size 1 still
show that its objective needs attention.

| Brain landmark evaluation | Rigid, mm | L-BFGS, mm | Block matching, mm |
|---|---:|---:|---:|
| BraTSReg 021, mean error | 2.893 | 2.738 | 2.817 |
| BraTSReg 127, mean error | 8.166 | 7.997 | 7.718 |

Both methods improve mean landmark error from rigid. Case 021 still does not
beat identity's 2.470 mm. Brain masks participate in registration, so their
Dice is not held out. Landmarks and abdominal organ labels are held out.

| Abdominal pair | Rigid Dice | Block-matching Dice |
|---|---:|---:|
| 0001 | 0.4921 | 0.7060 |
| 0002 | 0.6796 | 0.7586 |
| 0003 | 0.6426 | 0.6919 |
| 0004 | 0.7898 | 0.8500 |
| 0005 | 0.4996 | 0.8136 |
| 0006 | 0.4720 | 0.7331 |
| 0007 | 0.5227 | 0.7766 |
| 0008 | 0.7302 | 0.8548 |

Mean time per affine fit, including pyramids and optimization:

| Suite | Torch on GPU, seconds | Native CUDA, seconds |
|---|---:|---:|
| Synthetic 2D | 0.835 | 0.253 |
| Synthetic 3D | 1.595 | 0.243 |
| Native brains | 13.089 | 0.885 |
| Native abdomens | 12.426 | 0.830 |

These are single-run measurements on the two local RTX PRO 6000 Blackwell
compute GPUs, with four Torch CPU threads. Loading, feature preparation, rigid
initialization, and evaluation are excluded. Exporting the small matrix to CPU
is included and synchronizes the GPU. Native CUDA was about 15 times faster
than pure Torch on GPU for these real cases. The previous NiftyReg subprocess
timings included file I/O and output resampling and are not directly comparable.

## Implementation and verification

The algorithm uses width-4 blocks, integer search offsets from -3 to 3,
variance-based selection of up to 50% of all grid blocks, and least trimmed
squares retaining 50% of correspondences. Forward and reverse affine estimates
are averaged with the inverse estimate from the other direction. Scales are
`[4, 2, 1]`, with `[20, 10, 10]` iterations. No scale bounds are applied.

Both least-squares implementations center physical coordinates, normalize
source coordinates, and accumulate in float64. Torch solves the small normal
system using Torch operations. CUDA constructs the system with a reduction
kernel and solves it with pivoted elimination. Invalid or rank-deficient fits
stop that level and preserve its last valid matrix.

This follows the approach described by
[Modat et al.](https://doi.org/10.1117/1.JMI.1.2.024003), with independently
implemented matching and fitting and a shared Torch spatial image pyramid.
No NiftyReg source or binary is included or invoked by this implementation or
its benchmark. It is not a bitwise reproduction of NiftyReg. In particular,
its pyramid construction differs. Torch/CUDA agreement here describes these
recorded runs; floating-point differences can affect other cases.

All 53 focused tests passed with the rebuilt CUDA extension: 15 block-matching
checks, four solver checks, 30 polar checks, and four translation checks.
Without the extension or GPU, 46 passed and seven CUDA tests skipped.
The block-matching checks include an independent NumPy masked-NCC search,
known physical affine recovery, outliers, large physical origins, empty masks,
rank deficiency, both floating-point dtypes, full 2D/3D registration agreement,
and a nondefault CUDA stream. The complete fused-ops extension compiled with
CUDA 13.0 for architecture 12.0 against Torch 2.13.0+cu130.

- [Registration API](../docs/docs/reference/block_matching.md)
- [Raw measurements and run configurations](results/native_block_matching/)
- [Aggregate measurements](results/native_block_matching/summary.csv)
- [Per-case Torch/CUDA matrix agreement](results/native_block_matching/torch_cuda_agreement.json)
- [Data, preprocessing, and evaluation](POLAR_AFFINE_RESULTS.md)

## Reproduction

Run from the FireANTs root in the existing `try-general-reg` Python environment,
which provides the FireANTs and registerio dependencies. For the CUDA backend,
build the updated extension first. Ninja must be available on `PATH`.

```bash
cd fused_ops
CUDA_HOME=/usr/pubsw/packages/CUDA/13.0 TORCH_CUDA_ARCH_LIST=12.0 MAX_JOBS=4 \
  python setup.py build_ext \
  --build-lib ../tests/test_results/block_matching/extension \
  --build-temp ../tests/test_results/block_matching/build
cd ..
export PYTHONPATH=tests/test_results/block_matching/extension:.:..
export CUDA_VISIBLE_DEVICES=GPU-d5542ac3-f909-10f9-af1e-e31f8e3eea66

python -m pytest tests/test_block_matching.py tests/test_affine_solvers.py \
  tests/test_polar_affine.py tests/test_linear_translation.py

python experiments/native_block_matching.py --suite synthetic \
  --initial experiments/results/polar_affine/synthetic.jsonl \
  --output tests/test_results/block_matching/rerun_synthetic.jsonl
python experiments/native_block_matching.py --suite brains \
  --initial experiments/results/polar_affine/brains_native.jsonl \
  --output tests/test_results/block_matching/rerun_brains.jsonl
python experiments/native_block_matching.py --suite abdomens \
  --initial experiments/results/polar_affine/abdomens_native.jsonl \
  --output tests/test_results/block_matching/rerun_abdomens.jsonl
```

The benchmark defaults to both backends. For pure Torch without the extension,
use `PYTHONPATH=.:..` and `--backends torch`. The recorded abdominal runs used
the second compute GPU, `GPU-7429bfab-8a89-dc43-99bf-b63d4470f962`.
