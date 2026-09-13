# Affine solver comparison, 2026-09-13

GPU block matching is the stronger candidate for real 3D affine refinement in
this sample. It improved all three brain pairs and all eight MR–CT abdominal
pairs from the same rigid initializations, without collapsed transforms.
L-BFGS was accurate and fast on synthetic data and brains, but line search did
not prevent abdominal collapse when the loss favored it.

Both approaches remain experiments on `experiment/polar-affine`. This change
adds an L-BFGS experiment and runs NiftyReg's existing CUDA implementation;
it does not add a CPU block matcher or a new solver to the FireANTs public API.
The earlier polar parameterization remains opt-in.

## Measurements

The primary comparison uses direct matrix parameters, centered coordinates,
normalized translation, and no scale bounds for Adam and L-BFGS. Both use
`smooth_nr=0` on real images. Adam uses rate `0.003`; the displayed L-BFGS result
uses initial step size `1`, chosen before evaluation. The sweep also includes
`0.3` and `3`. NiftyReg uses scalar block NCC; it does not use FireANTs' loss or
its numerator smoothing setting.

| Evaluation | Rigid | Adam | L-BFGS | GPU block matching |
|---|---:|---:|---:|---:|
| 8 synthetic 2D cases, mean point error, mm | 3.1653 | 0.0086 | 0.0085 | 0.3918 |
| 8 synthetic 3D cases, mean point error, mm | 3.8419 | 0.0271 | 0.0268 | 0.5002 |
| 3 brains, mean brain-mask Dice | 0.9475 | 0.9608 | 0.9610 | 0.9612 |
| 8 abdomens, mean organ Dice | 0.6036 | 0.7569 | 0.7564 | 0.7693 |

All three affine configurations displayed above improved every case over its
rigid initialization on these metrics. This does not hold for every L-BFGS
step size or loss setting, as the failures below show.

The earlier successful abdominal Adam setting retains numerator smoothing.
With that setting, Adam reaches mean Dice `0.7685` without bounds, or `0.7692`
with the earlier bounds `[0.75, 4/3]`. GPU block matching's `0.7693` is therefore
close to tuned Adam overall. It is not uniformly more accurate per pair.

Per-pair abdominal Dice with the primary settings:

| Pair | Rigid | Adam | L-BFGS | GPU block matching |
|---|---:|---:|---:|---:|
| 0001 | 0.4921 | 0.6039 | 0.6083 | 0.6696 |
| 0002 | 0.6796 | 0.7333 | 0.7352 | 0.7621 |
| 0003 | 0.6426 | 0.7118 | 0.7006 | 0.6982 |
| 0004 | 0.7898 | 0.8502 | 0.8502 | 0.8500 |
| 0005 | 0.4996 | 0.7891 | 0.7891 | 0.8150 |
| 0006 | 0.4720 | 0.7194 | 0.7194 | 0.7438 |
| 0007 | 0.5227 | 0.7775 | 0.7784 | 0.7675 |
| 0008 | 0.7302 | 0.8698 | 0.8698 | 0.8479 |

For the previous inter-subject brain failure, Dice is `0.9151` after rigid,
`0.9464` after L-BFGS with `smooth_nr=0`, and `0.9465` after GPU block matching.
The longitudinal brain landmarks provide a separate anatomical measurement:

| Case | Rigid mean error, mm | Adam | L-BFGS | GPU block matching |
|---|---:|---:|---:|---:|
| BraTSReg 021 | 2.893 | 2.739 | 2.738 | 2.828 |
| BraTSReg 127 | 8.166 | 7.997 | 7.997 | 7.886 |

All improve the mean landmark error from rigid. Case 021 still does not beat
identity's `2.470` mm. Brain masks participate in optimization, so their Dice
is not a held-out measure. Landmarks and abdominal organ labels do not
participate in optimization.

Mean measured time per affine fit, in seconds:

| Suite | Adam | L-BFGS, step 1 | GPU block matching |
|---|---:|---:|---:|
| Synthetic 2D | 0.463 | 0.067 | 0.563 |
| Synthetic 3D | 0.349 | 0.075 | 0.681 |
| Native brains | 2.974 | 0.859 | 2.522 |
| Native abdomens | 7.224 | 3.209 | 1.903 |

Adam and L-BFGS timings cover pyramid construction and optimization, excluding
loading, features, initialization, and evaluation. NiftyReg timings cover the
subprocess, including input reads and final image resampling/writing; preparing
its input files is excluded. These are single-run measurements on the two local
RTX PRO 6000 Blackwell compute GPUs. CPU reference jobs shared the host with
other experiments. The timings are not a controlled CPU/GPU speedup benchmark.

## Failures and loss settings

L-BFGS with `smooth_nr=0` aligned all eight abdomens at step sizes `0.3` and
`1`. At step `3`, abdomen 0002 collapsed to organ Dice `0`. Its smallest
singular value was `0.000145`. The coarse-scale loss decreased from `-4.3885`
to `-8.4875`, so accepting only decreases in loss did not reject the collapse.

The masked objective evaluates the intersection of the fixed and warped
moving masks. For this failure, an independent nearest-neighbor resampling
measured just four intersecting body-mask voxels, compared with 1,663,126 after
rigid registration. Removing numerator smoothing does not fix an objective
whose evaluated region can shrink this much.

Retaining default numerator smoothing was worse for L-BFGS:

- On the previous inter-subject brain case, step `1` produced Dice `0.3159`,
  versus `0.9464` with numerator smoothing removed.
- Abdomen 0001 collapsed at all three step sizes. Mean abdominal Dice was
  `0.6818` at step `1`, versus `0.7685` for Adam on that same loss.
- Abdomen 0002 also collapsed at step `3`.

This supports changing the optimization method and checking the objective
together. It does not support unconstrained L-BFGS as a general replacement
for the existing affine optimizer. Block matching avoids optimizing this
whole-image masked objective: it estimates local correspondences and fits
the affine with least trimmed squares.

## CUDA implementation and checks

The GPU experiments use upstream
[NiftyReg source at `14cd5d58`](https://github.com/KCL-BMEIS/niftyreg/tree/14cd5d58ff9969497e2dd0d6762dbc050eaa4e0f),
version `2.0.0+12.14cd5d58`, compiled with CUDA 13 for architecture 12.0.
No NiftyReg source modifications were needed. Automatic GPU detection selected
the local architecture; disabling that check selected obsolete architectures
and failed with CUDA 13.

Block search runs on CUDA in both 2D and 3D. The 3D least-trimmed-squares affine
fit also runs on CUDA. Upstream's 2D affine fit still transfers the small
correspondence set to CPU. Its final cubic image resampling runs on CPU in
both dimensions. These results therefore do not establish an entirely GPU
resident 2D registration pipeline.

All 27 primary CUDA runs confirmed `Platform: CUDA` in their output. The same
upstream version was also run on CPU on all 27 cases. Maximum absolute
CPU/GPU metric differences were `0.00446` mm in synthetic mean point error,
`0.000947` brain Dice, and `0.006607` abdominal Dice. The maximum longitudinal
mean landmark-error difference was `0.0934` mm.

The matrices are not numerically equivalent: evaluated over a regular
5-by-5-by-5 grid spanning the fixed field of view, the largest per-case mean
CPU/GPU transform difference was `3.09` mm. The full differences are retained.
Repeating GPU abdomen 0001 produced an identical exported matrix.

The installed CPU NiftyReg 1.3.9 was the initial reference. It is retained in
the raw `synthetic.jsonl`, `brains.jsonl`, and `abdomens.jsonl` files. It differs
from the current implementation and slightly worsened abdomen 0002. The
headline comparison and plots use the current CUDA results instead.

Thirty-eight focused tests passed: four solver checks, thirty polar checks,
and four translation checks. The new solver checks test 2D/3D physical
coordinate conversion and image-based recovery of independently sampled
known transforms. Compilation checks and `git diff --check` passed.
There are 223 recorded affine fits, excluding preliminary runs. All expected
trial records are present; registration failures are retained in the metrics.

## Method and artifacts

All methods reuse the exact recorded rigid matrices; they do not rerun rigid
registration. The data and independent evaluation are described in the
[polar experiment report](POLAR_AFFINE_RESULTS.md). Brains use native 1 mm
images; abdomens use native 2 mm images. Adam and L-BFGS use MSE on synthetic
images. On real images they use the same masked
intensity NCC for brains and masked MIND-descriptor NCC for abdomens. NiftyReg
uses normalized scalar intensities and the same independent masks. This is a
comparison of complete algorithms on MR–CT, not a comparison with identical
features and losses.

Adam runs 100 iterations at each scale `[4, 2, 1]`, restoring its best measured
iterate. L-BFGS uses PyTorch's strong-Wolfe line search, up to 100 iterations
and 200 evaluations per scale, history size 12, gradient tolerance `1e-8`,
and change tolerance `1e-10`. Its history resets at each scale. It restores
the scale's initial parameters if the final loss is worse; that guard did
not activate in the recorded runs. There is no projection during line search.

NiftyReg uses symmetric affine refinement directly from the rigid matrix,
three pyramid levels, `-maxit 10`, and its default block/inlier percentages
of 50. Physical matrices are converted between SimpleITK LPS and NIfTI RAS.
Labels are evaluated using independent SimpleITK nearest-neighbor resampling.
Neither method was followed by deformable registration in this experiment.

The experiment helper also corrects the stored channel count after appending
a mask to MIND descriptors. It does not change the feature arrays.

- [Primary aggregate measurements](results/affine_solvers/comparison/summary.csv)
- [All solver and loss settings](results/affine_solvers/all_summary.csv)
- [Per-case comparison](results/affine_solvers/comparison/per_case.png)
- [Brain contours](results/affine_solvers/comparison/brains_alignment.png)
- [Abdominal contours](results/affine_solvers/comparison/abdomens_alignment.png)
- [L-BFGS failure with default NCC](results/affine_solvers/comparison/abdomens_default_ncc_alignment.png)
- [CPU/GPU differences](results/affine_solvers/cpu_cuda_agreement.json)
- [Collapse mask overlap](results/affine_solvers/abdomen_0002_overlap.json)
- [CUDA build details](results/affine_solvers/niftyreg_build.json)
- [Raw results and configurations](results/affine_solvers/)

## Reproduction

Run from the FireANTs root with both FireANTs and registerio dependencies.
The Python environment used was
`/autofs/cluster/dalcalab2/users/nd480/.local/share/mamba/envs/try-general-reg/bin/python`.
Use a compute GPU UUID, excluding the display GPU.

```bash
export PYTHONPATH=.:..
export CUDA_VISIBLE_DEVICES=GPU-d5542ac3-f909-10f9-af1e-e31f8e3eea66

git clone https://github.com/KCL-BMEIS/niftyreg.git tests/test_results/niftyreg_rerun/source
git -C tests/test_results/niftyreg_rerun/source checkout --detach 14cd5d58ff9969497e2dd0d6762dbc050eaa4e0f
cmake -S tests/test_results/niftyreg_rerun/source -B tests/test_results/niftyreg_rerun/build \
  -DUSE_CUDA=ON -DCHECK_GPU=ON -DCMAKE_CUDA_ARCHITECTURES=120 \
  -DCMAKE_CUDA_COMPILER=/usr/pubsw/packages/CUDA/13.0/bin/nvcc \
  -DCUDAToolkit_ROOT=/usr/pubsw/packages/CUDA/13.0 \
  -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=OFF -DUSE_CUDA_FMA=OFF
cmake --build tests/test_results/niftyreg_rerun/build --target reg_aladin -j 12

python experiments/affine_solvers.py --suite synthetic \
  --initial experiments/results/polar_affine/synthetic.jsonl \
  --niftyreg tests/test_results/niftyreg_rerun/build/reg-apps/reg_aladin --niftyreg-platform cuda \
  --work tests/test_results/affine_solver_rerun/synthetic \
  --output tests/test_results/affine_solver_rerun/synthetic.jsonl

python experiments/affine_solvers.py --suite brains --spacing 1 --kernel 9 7 5 \
  --initial experiments/results/polar_affine/brains_native.jsonl \
  --niftyreg tests/test_results/niftyreg_rerun/build/reg-apps/reg_aladin --niftyreg-platform cuda \
  --work tests/test_results/affine_solver_rerun/brains \
  --output tests/test_results/affine_solver_rerun/brains.jsonl

python experiments/affine_solvers.py --suite abdomens --spacing 2 --kernel 13 9 7 \
  --initial experiments/results/polar_affine/abdomens_native.jsonl \
  --niftyreg tests/test_results/niftyreg_rerun/build/reg-apps/reg_aladin --niftyreg-platform cuda \
  --work tests/test_results/affine_solver_rerun/abdomens \
  --output tests/test_results/affine_solver_rerun/abdomens.jsonl
```

Use `--methods adam lbfgs --smooth-nr 1e-5` for the additional default-NCC
comparison. Use `--methods niftyreg --niftyreg-platform cpu` for the matching
upstream CPU reference. Recorded configurations contain the exact arguments.

```bash
python experiments/summarize_affine_solvers.py \
  experiments/results/affine_solvers/synthetic.jsonl \
  experiments/results/affine_solvers/brains.jsonl \
  experiments/results/affine_solvers/abdomens.jsonl \
  --gpu-directory experiments/results/affine_solvers \
  --output-dir tests/test_results/affine_solver_rerun/summary
```
