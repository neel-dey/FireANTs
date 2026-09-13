"""Compare Adam, L-BFGS, and NiftyReg affine refinement from identical matrices."""

import argparse
import json
import logging
import os
from pathlib import Path
import subprocess
import time

import numpy as np
import SimpleITK as sitk
import torch

from fireants.interpolator import fireants_interpolator
from fireants.losses.cc import gaussian_1d
from fireants.registration.affine import AffineRegistration
from fireants.utils.globals import MIN_IMG_SIZE
from experiments.polar_affine import features, make_image, real_case, score, synthetic_case


def level_images(reg, scale):
    """Use the affine solver's image pyramid and mask downsampling."""
    fixed, moving = reg.fixed_images(), reg.moving_images()
    fixed_size = [max(int(s / scale), MIN_IMG_SIZE) for s in fixed.shape[2:]]
    moving_size = [max(int(s / scale), MIN_IMG_SIZE) for s in moving.shape[2:]]
    if scale == 1:
        return fixed, moving
    sigma = .5 * torch.tensor([s / n for s, n in zip(fixed.shape[2:], fixed_size)],
                              device=fixed.device, dtype=fixed.dtype)
    kernels = [gaussian_1d(s, truncated=2) for s in sigma]
    return tuple(reg._downsample_image_and_mask(
        array, size=size, mode=reg.fixed_images.interpolate_mode, gaussians=kernels,
        align_corners=True) for array, size in ((fixed, fixed_size), (moving, moving_size)))


def fit_lbfgs(reg, lr=1., iterations=100):
    """Run unconstrained strong-Wolfe L-BFGS with a new history at each level."""
    if reg.scale_bounds is not None:
        raise ValueError("The L-BFGS experiment requires scale_bounds=None")
    fixed_t2p = reg.fixed_images.get_torch2phy().to(reg.dtype)
    moving_p2t = reg.moving_images.get_phy2torch().to(reg.dtype)
    parameters = reg.optimized_parameters()
    traces = []
    for scale in reg.scales:
        if hasattr(reg.loss_fn, "set_current_scale_and_iterations"):
            reg.loss_fn.set_current_scale_and_iterations(scale, iterations)
        fixed, moving = level_images(reg, scale)
        optimizer = torch.optim.LBFGS(parameters, lr=lr, max_iter=iterations,
                                     max_eval=iterations * 2, history_size=12,
                                     tolerance_grad=1e-8, tolerance_change=1e-10,
                                     line_search_fn="strong_wolfe")
        evaluations = []

        def closure():
            optimizer.zero_grad()
            affine = (moving_p2t @ reg.get_affine_matrix() @ fixed_t2p)[:, :-1].contiguous()
            warped = fireants_interpolator(moving, affine=affine, out_shape=fixed.shape,
                                           mode="bilinear", align_corners=True)
            loss = reg.loss_fn(warped, fixed)
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite L-BFGS trial loss")
            loss.backward()
            evaluations.append(float(loss.detach()))
            return loss

        before = [p.detach().clone() for p in parameters]
        optimizer.step(closure)
        final_loss = float(closure().detach())
        restored = final_loss > evaluations[0] + 1e-7
        if restored:
            with torch.no_grad():
                for parameter, original in zip(parameters, before):
                    parameter.copy_(original)
            final_loss = evaluations[0]
        state = optimizer.state[parameters[0]]
        traces.append(dict(scale=scale, iterations=state.get("n_iter", 0),
                           evaluations=len(evaluations), initial_loss=evaluations[0],
                           final_loss=final_loss, restored_initial=restored,
                           trial_losses=evaluations))
    return traces


def lps_ras(matrix):
    dims = matrix.shape[0] - 1
    flip = np.eye(dims + 1)
    flip[0, 0] = flip[1, 1] = -1
    return flip @ matrix @ flip


def nifti_matrix(matrix):
    """Embed a 2D physical matrix in NIfTI's 3D RAS coordinates."""
    ras = lps_ras(matrix)
    if matrix.shape == (4, 4):
        return ras
    result = np.eye(4)
    result[:2, :2], result[:2, 3] = ras[:2, :2], ras[:2, 2]
    return result


def fit_niftyreg(case, init, binary, work, iterations=10, platform="legacy"):
    """Use scalar images; the installed block matcher reads one image channel."""
    work.mkdir(parents=True, exist_ok=True)
    dims = init.shape[0] - 1
    prepared = []
    for side in ("fixed", "moving"):
        image = case[side + "_batch"].images[0]
        array = image.array.detach().cpu().numpy()[0]
        if case["name"].startswith("abdomen"):
            native = case[side]
            clipped = sitk.Clamp(sitk.Cast(native, sitk.sitkFloat32),
                                 lowerBound=-450. if side == "fixed" else 0.,
                                 upperBound=450. if side == "fixed" else 20000.)
            # The feature stack's last channel contains the independent body mask.
            mask = make_image(array[-1], image.itk_image)
            scalar = features(clipped, "cpu", "intensity", mask, masked=True).images[0]
            data, mask_array, geometry = scalar.array[0, 0].numpy(), scalar.array[0, -1].numpy(), clipped
        else:
            data, geometry = array[0], image.itk_image
            mask_array = array[-1] if case["loss"].startswith("masked_") else None
        path = work / (side + ".nii.gz")
        sitk.WriteImage(make_image(data, geometry), str(path))
        mask_path = None
        if mask_array is not None:
            mask_path = work / (side + "_mask.nii.gz")
            sitk.WriteImage(sitk.Cast(make_image(mask_array, geometry), sitk.sitkUInt8), str(mask_path))
        prepared.append((path, mask_path))
    init_path, output_path = work / "initial_ras.txt", work / "affine_ras.txt"
    np.savetxt(init_path, nifti_matrix(init), fmt="%.12g")
    command = [str(binary), "-ref", str(prepared[0][0]), "-flo", str(prepared[1][0]),
               "-inaff", str(init_path), "-affDirect", "-ln", "3", "-lp", "3",
               "-maxit", str(iterations), "-aff", str(output_path), "-res", str(work / "warped.nii.gz")]
    command.extend(["-sym"] if platform == "legacy" else ["-platf", "1" if platform == "cuda" else "0"])
    for flag, (_, mask_path) in zip(("-rmask", "-fmask"), prepared):
        if mask_path is not None:
            command.extend((flag, str(mask_path)))
    env = dict(os.environ, OMP_NUM_THREADS="8")
    lib = str(binary.parent.parent / "lib")
    env["LD_LIBRARY_PATH"] = lib + (":" + env["LD_LIBRARY_PATH"] if env.get("LD_LIBRARY_PATH") else "")
    start = time.perf_counter()
    process = subprocess.run(command, env=env, capture_output=True, text=True, timeout=600)
    seconds = time.perf_counter() - start
    (work / "command.json").write_text(json.dumps(command, indent=2) + "\n")
    (work / "run.log").write_text(process.stdout + process.stderr)
    if process.returncode:
        raise RuntimeError(f"reg_aladin exited {process.returncode}: {process.stderr[-1500:]}")
    if platform == "cuda" and "Platform: CUDA" not in process.stdout:
        raise RuntimeError("NiftyReg did not confirm the requested CUDA platform")
    ras = np.loadtxt(output_path)
    if dims == 2:
        ras = ras[np.ix_([0, 1, 3], [0, 1, 3])]
    return lps_ras(ras), dict(seconds=seconds, command=command,
                              platform="CUDA" if platform == "cuda" else "CPU", omp_threads_requested=8)


def initial_rows(path):
    return {row["name"]: row for row in (json.loads(s) for s in path.read_text().splitlines())
            if row["method"] == "rigid"}


def run(case, args, previous, handle):
    init = np.asarray(previous[case["name"]]["matrix"])
    metadata = {key: case[key] for key in ("name", "paths", "truth", "loss") if key in case}
    metadata["smooth_nr"] = args.smooth_nr if case["loss"] != "mse" else None

    def record(method, matrix, **extra):
        row = dict(metadata, method=method, **score(case, matrix), **extra)
        if method == "niftyreg":
            row.update(loss="block_ncc", smooth_nr=None, features="scalar_intensity")
        handle.write(json.dumps(row) + "\n")
        handle.flush()
        print(json.dumps({key: row[key] for key in ("name", "method", "lr", "dice", "tre_mean_mm",
                                                   "seconds", "loss_value", "status") if key in row}), flush=True)

    record("rigid", init, reused_from=str(args.initial))
    common = dict(scales=[4, 2, 1], iterations=[args.iterations] * 3,
                  fixed_images=case["fixed_batch"], moving_images=case["moving_batch"],
                  loss_type=case["loss"], cc_kernel_size=args.kernel,
                  normalize_translation=True, around_center=True, tolerance=float("inf"),
                  keep_best=True, progress_bar=False,
                  init_rigid=torch.as_tensor(init, dtype=torch.float32, device=args.device)[None])
    for method in args.methods:
        try:
            if method == "niftyreg":
                matrix, info = fit_niftyreg(case, init, args.niftyreg, args.work / case["name"],
                                            args.block_iterations, args.niftyreg_platform)
                record(method, matrix, **info)
                continue
            for rate in (args.rates if method == "lbfgs" else [.003]):
                options = dict(common, loss_params={} if case["loss"] == "mse" else {"smooth_nr": args.smooth_nr})
                reg = AffineRegistration(**options, parameterization="matrix", optimizer_lr=.003)
                start = time.perf_counter()
                if method == "lbfgs":
                    trace = fit_lbfgs(reg, rate, args.iterations)
                else:
                    reg.optimize()
                    trace = None
                matrix = reg.get_affine_matrix().detach()[0].cpu().numpy()
                seconds = time.perf_counter() - start
                with torch.no_grad():
                    loss = float(reg.loss_fn(reg.evaluate(case["fixed_batch"], case["moving_batch"]), case["fixed_batch"]()))
                record(method, matrix, lr=rate, bounds=None, seconds=seconds,
                       loss_value=loss, levels=trace)
                del reg
        except (RuntimeError, FloatingPointError, subprocess.TimeoutExpired) as error:
            row = dict(metadata, method=method, status="error", error=str(error))
            handle.write(json.dumps(row) + "\n")
            handle.flush()
            print(json.dumps(row), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", choices=["synthetic", "brains", "abdomens"], required=True)
    parser.add_argument("--methods", nargs="+", choices=["adam", "lbfgs", "niftyreg"], default=["adam", "lbfgs", "niftyreg"])
    parser.add_argument("--rates", type=float, nargs="+", default=[.3, 1., 3.])
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--block-iterations", type=int, default=10)
    parser.add_argument("--smooth-nr", type=float, default=0.)
    parser.add_argument("--kernel", type=int, nargs="+", default=[9, 7, 5])
    parser.add_argument("--spacing", type=float, default=1.)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(8)))
    parser.add_argument("--cases", nargs="+")
    parser.add_argument("--data-root", type=Path, default=Path("../../reference-runs"))
    parser.add_argument("--initial", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--work", type=Path, default=Path("tests/test_results/affine_solvers/niftyreg"))
    parser.add_argument("--niftyreg", type=Path, default=Path("/usr/pubsw/packages/niftyreg/nifty_reg-1.3.9/local/bin/reg_aladin"))
    parser.add_argument("--niftyreg-platform", choices=["legacy", "cpu", "cuda"], default="legacy",
                        help="Use legacy CPU flags or select a platform in current NiftyReg")
    args = parser.parse_args()
    torch.set_num_threads(4)
    sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(4)
    logging.basicConfig(level=logging.WARNING)
    logging.getLogger("fireants.registration.abstract").setLevel(logging.WARNING)
    if args.device.startswith("cuda"):
        torch.cuda.set_device(args.device)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.with_suffix(".config.json").write_text(json.dumps(dict(vars(args), torch_version=torch.__version__),
                                                                default=str, indent=2) + "\n")
    previous = initial_rows(args.initial)
    with args.output.open("w") as handle:
        if args.suite == "synthetic":
            for dims in (2, 3):
                for seed in args.seeds:
                    run(synthetic_case(dims, seed, args.device), args, previous, handle)
        else:
            cases = args.cases or (["brain_inter", "brain_021", "brain_127"] if args.suite == "brains"
                                   else [f"abdomen_{i:04}" for i in range(1, 9)])
            for name in cases:
                run(real_case(name, args.data_root, args.device, args.spacing, masked=True), args, previous, handle)


if __name__ == "__main__":
    main()
