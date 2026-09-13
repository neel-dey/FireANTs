"""Compare FireANTs Torch and CUDA block matching from recorded rigid matrices."""

import argparse
import json
from pathlib import Path
import time

import numpy as np
import SimpleITK as sitk
import torch

from experiments.polar_affine import features, make_image, real_case, score, synthetic_case
from fireants.registration.block_matching import BlockMatchingRegistration


def scalar_batches(case, device):
    if not case["name"].startswith("abdomen"):
        return case["fixed_batch"], case["moving_batch"]
    results = []
    for side in ("fixed", "moving"):
        image = case[side + "_batch"].images[0]
        mask = make_image(image.array[0, -1].cpu().numpy(), image.itk_image)
        clipped = sitk.Clamp(sitk.Cast(case[side], sitk.sitkFloat32),
                             lowerBound=-450. if side == "fixed" else 0.,
                             upperBound=450. if side == "fixed" else 20000.)
        results.append(features(clipped, device, "intensity", mask, masked=True))
    return tuple(results)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", choices=["synthetic", "brains", "abdomens"], required=True)
    parser.add_argument("--backends", nargs="+", choices=["torch", "cuda"], default=["torch", "cuda"])
    parser.add_argument("--initial", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=Path("../../reference-runs"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(8)))
    parser.add_argument("--cases", nargs="+")
    parser.add_argument("--iterations", type=int, nargs=3, default=[20, 10, 10])
    args = parser.parse_args()
    torch.set_num_threads(4)
    sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(4)
    previous = {r["name"]: r for r in map(json.loads, args.initial.read_text().splitlines()) if r["method"] == "rigid"}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.with_suffix(".config.json").write_text(json.dumps(dict(vars(args), torch_version=torch.__version__), default=str, indent=2) + "\n")
    if args.suite == "synthetic":
        cases = (synthetic_case(d, seed, args.device) for d in (2, 3) for seed in args.seeds)
    else:
        names = args.cases or (["brain_inter", "brain_021", "brain_127"] if args.suite == "brains"
                               else [f"abdomen_{i:04}" for i in range(1, 9)])
        cases = (real_case(name, args.data_root, args.device, 1. if args.suite == "brains" else 2., masked=True) for name in names)
    with args.output.open("w") as handle:
        for case in cases:
            fixed, moving = scalar_batches(case, args.device)
            initial = np.asarray(previous[case["name"]]["matrix"])
            metadata = {key: case[key] for key in ("name", "paths", "truth") if key in case}
            row = dict(metadata, method="rigid", **score(case, initial))
            handle.write(json.dumps(row) + "\n")
            for backend in args.backends:
                reg = BlockMatchingRegistration([4, 2, 1], args.iterations, fixed, moving,
                    init_affine=torch.tensor(initial, device=args.device, dtype=torch.float32)[None],
                    backend=backend, masked=case["loss"].startswith("masked_"), progress_bar=False)
                if args.device.startswith("cuda"):
                    torch.cuda.synchronize()
                start = time.perf_counter()
                reg.optimize()
                matrix = reg.get_affine_matrix()[0].cpu().numpy()
                seconds = time.perf_counter() - start
                row = dict(metadata, method="block_" + backend, seconds=seconds,
                           history=reg.history, **score(case, matrix))
                handle.write(json.dumps(row) + "\n")
                handle.flush()
                print(json.dumps({key: row[key] for key in ("name", "method", "seconds", "dice", "tre_mean_mm", "history") if key in row}), flush=True)


if __name__ == "__main__":
    main()
