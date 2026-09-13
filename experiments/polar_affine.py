"""Compare matrix and polar affine refinement with independent geometry metrics.

Run from the repository root with PYTHONPATH=.:.. to use registerio's MIND-SSC
implementation for abdominal MR/CT. Labels and landmarks are used only to score.
"""

import argparse
import csv
import json
import logging
from pathlib import Path
import time

import numpy as np
import SimpleITK as sitk
import torch
from scipy.linalg import expm

from fireants.io import Image, BatchedImages
from fireants.registration.affine import AffineRegistration
from fireants.registration.rigid import RigidRegistration


def make_image(array, reference=None, spacing=None, origin=None):
    image = sitk.GetImageFromArray(array.astype(np.float32))
    if reference is not None:
        image.CopyInformation(reference)
    if spacing is not None:
        image.SetSpacing(tuple(spacing))
    if origin is not None:
        image.SetOrigin(tuple(origin))
    return image


def resample_spacing(image, spacing):
    size = np.asarray(image.GetSize())
    extent = (size - 1) * np.asarray(image.GetSpacing())
    new_size = np.maximum(2, np.round(extent / spacing).astype(int) + 1)
    return sitk.Resample(image, new_size.tolist(), sitk.Transform(3, sitk.sitkIdentity),
                         sitk.sitkLinear, image.GetOrigin(), (extent / (new_size - 1)).tolist(),
                         image.GetDirection(), 0., sitk.sitkFloat32)


def features(image, device, kind="intensity", mask=None, masked=False):
    values = sitk.GetArrayFromImage(image)
    foreground = values != 0
    if mask is not None:
        mask_on_image = sitk.Resample(mask, image, sitk.Transform(image.GetDimension(), sitk.sitkIdentity),
                                     sitk.sitkNearestNeighbor, 0.)
        foreground = sitk.GetArrayFromImage(mask_on_image) > 0
    low, high = np.percentile(values[foreground], [1, 99])
    values = np.clip((values - low) / max(high - low, 1e-6), 0., 1.)
    if masked:
        values *= foreground
    result = Image(make_image(values, image), device=device)
    if kind == "mind":
        from registerio.features.mindssc import mindssc
        with torch.no_grad():
            result.array = mindssc(result.array, radius=1, dilation=2)
        result.channels = result.array.shape[1]
    if masked:
        result.array *= torch.as_tensor(foreground, device=device)[None, None]
        result.array = torch.cat([result.array, torch.as_tensor(foreground, device=device)[None, None]], dim=1)
        result.channels = result.array.shape[1]
    return BatchedImages([result])


def synthetic_case(dims, seed, device):
    rng = np.random.default_rng(seed)
    size = 96 if dims == 2 else 64
    spacing = np.linspace(1.2, 1.8, dims)
    origin = np.arange(dims) * 30. + 40.
    extent = (size - 1) * spacing
    center = origin + extent / 2
    coords = np.stack(np.meshgrid(*[np.arange(size)] * dims, indexing="ij"), axis=-1)[..., ::-1]
    coords = coords * spacing + origin
    means = center + rng.uniform(-.24, .24, (10, dims)) * extent
    widths = rng.uniform(.035, .09, (10, dims)) * extent
    weights = rng.uniform(.4, 1.2, 10)

    def phantom(points):
        result = np.zeros(points.shape[:-1])
        for mean, width, weight in zip(means, widths, weights):
            result += weight * np.exp(-.5 * (((points - mean) / width) ** 2).sum(-1))
        return result.astype(np.float32)

    skew = rng.normal(0, .15, (dims, dims))
    rotation = expm(skew - skew.T)
    symmetric = rng.normal(0, .10, (dims, dims))
    symmetric = (symmetric + symmetric.T) / 2
    linear = rotation @ expm(symmetric)
    displacement = rng.uniform(-.055, .055, dims) * extent
    matrix = np.eye(dims + 1)
    matrix[:dims, :dims] = linear
    matrix[:dims, -1] = center + displacement - linear @ center
    moving_coords = (coords - center - displacement) @ np.linalg.inv(linear).T + center
    fixed = make_image(phantom(coords), spacing=spacing, origin=origin)
    moving = make_image(phantom(moving_coords), spacing=spacing, origin=origin)
    points = center + rng.uniform(-.3, .3, (2000, dims)) * extent
    return dict(name=f"synthetic_{dims}d_{seed}", fixed=fixed, moving=moving,
                fixed_batch=BatchedImages([Image(fixed, device=device)]),
                moving_batch=BatchedImages([Image(moving, device=device)]),
                points=points, target=points @ linear.T + matrix[:dims, -1],
                truth=matrix.tolist(), loss="mse")


def real_case(name, root, device, spacing, masked=False):
    feature_masks = (None, None)
    if name.startswith("abdomen_"):
        case = name.split("_")[1]
        folder = root / "abdomen/dataset"
        fixed_path = folder / f"imagesTr/AbdomenMRCT_{case}_0001.nii.gz"
        moving_path = folder / f"imagesTr/AbdomenMRCT_{case}_0000.nii.gz"
        fixed_labels = folder / f"labelsTr/AbdomenMRCT_{case}_0001.nii.gz"
        moving_labels = folder / f"labelsTr/AbdomenMRCT_{case}_0000.nii.gz"
        kind = "mind"
        loss = "masked_cc" if masked else "mse"
        if masked:
            feature_masks = tuple(sitk.ReadImage(str(folder / f"masksTr/AbdomenMRCT_{case}_{suffix}.nii.gz"))
                                  for suffix in ("0001", "0000"))
    else:
        folder = root / ("brats127" if name == "brain_127" else "brats")
        subject = "127" if name == "brain_127" else "021"
        fixed_path = folder / f"dataset/BraTSReg_{subject}_00_0000_t1.nii.gz"
        if name == "brain_inter":
            moving_path = root / "brats127/dataset/BraTSReg_127_00_0000_t1.nii.gz"
            moving_labels = root / "brats127/mask_00.nii.gz"
        else:
            day = "0148" if subject == "127" else "0214"
            moving_path = folder / f"dataset/BraTSReg_{subject}_01_{day}_t1.nii.gz"
            moving_labels = folder / "mask_01.nii.gz"
        fixed_labels = folder / "mask_00.nii.gz"
        kind = "intensity"
        loss = "masked_cc" if masked else "cc"
    fixed, moving = sitk.ReadImage(str(fixed_path)), sitk.ReadImage(str(moving_path))
    fixed_seg, moving_seg = sitk.ReadImage(str(fixed_labels)), sitk.ReadImage(str(moving_labels))
    if masked and name.startswith("brain"):
        feature_masks = (fixed_seg, moving_seg)
    fixed_grid, moving_grid = resample_spacing(fixed, spacing), resample_spacing(moving, spacing)
    if name.startswith("abdomen") and masked:
        fixed_grid = sitk.Clamp(fixed_grid, lowerBound=-450., upperBound=450.)
        moving_grid = sitk.Clamp(moving_grid, lowerBound=0., upperBound=20000.)
    result = dict(name=name + ("_masked" if masked else ""), fixed=fixed, moving=moving,
                  fixed_batch=features(fixed_grid, device, kind, feature_masks[0], masked),
                  moving_batch=features(moving_grid, device, kind, feature_masks[1], masked),
                  fixed_seg=fixed_seg, moving_seg=moving_seg, loss=loss,
                  paths=[str(p.resolve()) for p in (fixed_path, moving_path, fixed_labels, moving_labels)])
    if name in ("brain_021", "brain_127"):
        def landmarks(path):
            path = str(path).replace("_t1.nii.gz", "_landmarks.csv")
            with open(path) as handle:
                return {row[0]: np.asarray(row[1:], dtype=float) for row in list(csv.reader(handle))[1:]}
        fixed_points, moving_points = landmarks(fixed_path), landmarks(moving_path)
        shared = sorted(fixed_points.keys() & moving_points.keys())
        # BraTSReg landmark CSVs use physical LPS coordinates, as does SimpleITK.
        result["points"] = np.stack([fixed_points[key] for key in shared])
        result["target"] = np.stack([moving_points[key] for key in shared])
    return result


def score(case, matrix):
    dims = matrix.shape[0] - 1
    linear = matrix[:dims, :dims]
    scales = np.linalg.svd(linear, compute_uv=False)
    result = dict(determinant=float(np.linalg.det(linear)), singular_values=scales.tolist(),
                  condition=float(scales.max() / scales.min()), matrix=matrix.tolist())
    if "points" in case:
        moved = case["points"] @ linear.T + matrix[:dims, -1]
        errors = np.linalg.norm(moved - case["target"], axis=1)
        result.update(tre_mean_mm=float(errors.mean()), tre_median_mm=float(np.median(errors)),
                      tre_p95_mm=float(np.percentile(errors, 95)))
    if "fixed_seg" in case:
        transform = sitk.AffineTransform(dims)
        transform.SetMatrix(linear.ravel().tolist())
        transform.SetTranslation(matrix[:dims, -1].tolist())
        warped = sitk.Resample(case["moving_seg"], case["fixed_seg"], transform,
                              sitk.sitkNearestNeighbor, 0)
        fixed = sitk.GetArrayFromImage(case["fixed_seg"])
        moving = sitk.GetArrayFromImage(warped)
        labels = np.intersect1d(np.unique(fixed), np.unique(sitk.GetArrayFromImage(case["moving_seg"])))
        scores = {}
        for label in labels[labels != 0]:
            f, m = fixed == label, moving == label
            scores[str(int(label))] = float(2 * (f & m).sum() / max(1, f.sum() + m.sum()))
        result.update(dice=float(np.mean(list(scores.values()))), dice_per_label=scores)
    return result


def run_case(case, args, handle):
    fixed, moving = case["fixed_batch"], case["moving_batch"]
    dims = fixed.dims
    metadata = {key: case[key] for key in ("name", "paths", "truth", "loss") if key in case}
    metadata["initialization"] = args.initialization

    def record(method, matrix, **extra):
        row = dict(metadata, method=method, **extra, **score(case, matrix))
        handle.write(json.dumps(row) + "\n")
        handle.flush()
        concise = {key: value for key, value in row.items() if key in
                   ("name", "method", "lr", "bounds", "dice", "tre_mean_mm", "determinant", "seconds", "loss_value")}
        print(json.dumps(concise), flush=True)

    identity = np.eye(dims + 1)
    record("identity", identity)
    loss_params = {} if args.smooth_nr is None else {"smooth_nr": args.smooth_nr}
    common = dict(fixed_images=fixed, moving_images=moving, loss_type=case["loss"],
                  loss_params=loss_params.copy(), scales=[4, 2, 1], iterations=args.iterations,
                  cc_kernel_size=args.kernel[0] if len(args.kernel) == 1 else args.kernel,
                  optimizer="Adam", normalize_translation=True,
                  around_center=True, keep_best=True, tolerance=float("inf"), progress_bar=False)
    rigid = RigidRegistration(**{**common, "loss_params": {}}, optimizer_lr=.003, init_translation="cof")
    start = time.perf_counter()
    rigid.optimize()
    init = rigid.get_rigid_matrix().detach()
    record("rigid", init[0].cpu().numpy(), seconds=time.perf_counter() - start)
    if args.initialization == "cof":
        init = "cof"
    for bounds in ([None, (.75, 4/3)] if args.bounds == "both" else
                   [(.75, 4/3)] if args.bounds == "bounded" else [None]):
        for lr in args.rates:
            for method in (["matrix", "polar", "matrix_legacy"] if args.legacy else ["matrix", "polar"]):
                options = {**common, "loss_params": loss_params.copy()}
                if method == "matrix_legacy":
                    options.update(normalize_translation=False, keep_best=False)
                reg = AffineRegistration(**options, optimizer_lr=lr, init_rigid=init,
                                         parameterization="matrix" if method == "matrix_legacy" else method,
                                         scale_bounds=bounds)
                start = time.perf_counter()
                reg.optimize()
                matrix = reg.get_affine_matrix().detach()[0].cpu().numpy()
                seconds = time.perf_counter() - start
                with torch.no_grad():
                    loss = reg.loss_fn(reg.evaluate(fixed, moving), fixed()).item()
                record(method, matrix, lr=lr, bounds=bounds, seconds=seconds, loss_value=loss)
                del reg


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", choices=["synthetic", "brains", "abdomens"], required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rates", type=float, nargs="+", default=[.001, .003, .01, .03, .1])
    parser.add_argument("--iterations", type=int, nargs=3, default=[100, 100, 100])
    parser.add_argument("--bounds", choices=["both", "bounded", "none"], default="both")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3])
    parser.add_argument("--cases", nargs="+")
    parser.add_argument("--spacing", type=float, default=2.)
    parser.add_argument("--kernel", type=int, nargs="+", default=[7])
    parser.add_argument("--masked", action="store_true")
    parser.add_argument("--initialization", choices=["rigid", "cof"], default="rigid")
    parser.add_argument("--legacy", action="store_true",
                        help="Also test matrix optimization with physical translation and no best-iterate restoration")
    parser.add_argument("--smooth-nr", type=float,
                        help="Override NCC numerator smoothing for affine refinement; keep the rigid initialization unchanged")
    args = parser.parse_args()
    logging.basicConfig(level=logging.WARNING)
    logging.getLogger("fireants.registration.abstract").setLevel(logging.WARNING)
    torch.set_num_threads(4)
    sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(4)
    torch.manual_seed(0)
    np.random.seed(0)
    if args.device.startswith("cuda"):
        torch.cuda.set_device(args.device)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    metadata = dict(vars(args), torch_version=torch.__version__)
    if args.device.startswith("cuda"):
        metadata["gpu"] = torch.cuda.get_device_name(args.device)
    args.output.with_suffix(".config.json").write_text(json.dumps(metadata, default=str, indent=2) + "\n")
    with args.output.open("w") as handle:
        if args.suite == "synthetic":
            for dims in (2, 3):
                for seed in args.seeds:
                    run_case(synthetic_case(dims, seed, args.device), args, handle)
        else:
            cases = args.cases or (["brain_inter", "brain_021", "brain_127"] if args.suite == "brains"
                                   else [f"abdomen_{i:04}" for i in range(1, 9)])
            for name in cases:
                run_case(real_case(name, args.data_root, args.device, args.spacing, args.masked), args, handle)


if __name__ == "__main__":
    main()
