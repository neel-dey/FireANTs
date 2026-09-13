"""Aggregate solver results and plot per-case changes from rigid registration."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import SimpleITK as sitk


def read(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def summarize(path):
    rows = read(path)
    rigid = {row["name"]: row for row in rows if row["method"] == "rigid"}
    groups = {}
    for row in rows:
        suite = path.stem
        if row["name"].startswith("synthetic"):
            suite = "_".join(row["name"].split("_")[:2])
        groups.setdefault((suite, row["method"], row.get("lr")), []).append(row)
    results = []
    for (suite, method, rate), runs in groups.items():
        valid = [row for row in runs if row.get("status") != "error"]
        result = dict(source=path.stem, suite=suite, method=method, lr=rate,
                      count=len(valid), errors=len(runs) - len(valid))
        for metric in ("dice", "tre_mean_mm", "tre_median_mm", "seconds", "loss_value"):
            values = [row[metric] for row in valid if metric in row]
            if values:
                result[metric] = float(np.mean(values))
        if valid:
            result["min_singular_value"] = min(min(row["singular_values"]) for row in valid)
            result["max_singular_value"] = max(max(row["singular_values"]) for row in valid)
            result["nonpositive_determinants"] = sum(row["determinant"] <= 0 for row in valid)
        for metric, direction in (("dice", 1), ("tre_mean_mm", -1)):
            changes = [direction * (row[metric] - rigid[row["name"]][metric])
                       for row in valid if metric in row]
            if changes:
                result[metric + "_improved_over_rigid"] = sum(change > 0 for change in changes)
                result[metric + "_worst_signed_change"] = min(changes)
        if method == "lbfgs" and valid:
            result["mean_evaluations"] = np.mean([sum(level["evaluations"] for level in row["levels"])
                                                  for row in valid])
            result["restored_levels"] = sum(level["restored_initial"] for row in valid for level in row["levels"])
        results.append(result)
    return results


def plot(files, output):
    selections = []
    for path in files:
        rows = read(path)
        if path.stem.startswith("synthetic"):
            for dims in (2, 3):
                selections.append((f"Synthetic {dims}D", [r for r in rows if f"_{dims}d_" in r["name"]], "tre_mean_mm"))
        else:
            selections.append((path.stem.replace("_", " "), rows, "dice"))
    figure, axes = plt.subplots(1, len(selections), figsize=(5 * len(selections), 4.5), squeeze=False)
    for ax, (title, rows, metric) in zip(axes[0], selections):
        names = list(dict.fromkeys(row["name"] for row in rows))
        for method, rate, label in (("rigid", None, "Rigid"), ("adam", .003, "Adam 0.003"),
                                    ("lbfgs", 1., "L-BFGS 1"), ("niftyreg", None, "Block matching")):
            values = {r["name"]: r[metric] for r in rows if r["method"] == method
                      and r.get("lr") == rate and metric in r}
            if values:
                ax.plot(range(len(names)), [values.get(name, np.nan) for name in names], marker="o", label=label)
        ax.set_title(title)
        ax.set_xticks(range(len(names)), [name.removesuffix("_masked").split("_")[-1] for name in names])
        ax.set_xlabel("Case")
        ax.set_ylabel("Mean point error (mm)" if metric == "tre_mean_mm" else "Mean Dice")
        if metric == "tre_mean_mm":
            ax.set_yscale("log")
        ax.grid(alpha=.2)
    axes[0, -1].legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(output, dpi=160)
    plt.close(figure)


def contours(path, output):
    rows = read(path)
    name = rows[0]["name"]
    selected = [r for r in rows if r["name"] == name and "paths" in r and
                (r["method"] in ("rigid", "adam", "niftyreg") or
                 r["method"] == "lbfgs" and r["lr"] == 1.)]
    if not selected:
        return
    fixed_path, _, fixed_label_path, moving_label_path = selected[0]["paths"]
    fixed = sitk.GetArrayFromImage(sitk.ReadImage(fixed_path))
    reference = sitk.ReadImage(fixed_label_path)
    fixed_labels = sitk.GetArrayFromImage(reference)
    moving_labels = sitk.ReadImage(moving_label_path)
    axis = 1 if name.startswith("abdomen") else 0
    index = int(np.median(np.argwhere(fixed_labels > 0)[:, axis]))
    fixed_slice = np.take(fixed, index, axis=axis)
    label_slice = np.take(fixed_labels, index, axis=axis)
    lo, hi = np.percentile(fixed_slice, [2, 98])
    figure, axes = plt.subplots(1, len(selected), figsize=(3.5 * len(selected), 4.7), squeeze=False)
    for ax, row in zip(axes[0], selected):
        matrix = np.asarray(row["matrix"])
        transform = sitk.AffineTransform(3)
        transform.SetMatrix(matrix[:3, :3].ravel().tolist())
        transform.SetTranslation(matrix[:3, 3].tolist())
        warped = sitk.Resample(moving_labels, reference, transform, sitk.sitkNearestNeighbor, 0.)
        moved_slice = np.take(sitk.GetArrayFromImage(warped), index, axis=axis)
        ax.imshow(fixed_slice, cmap="gray", origin="lower", vmin=lo, vmax=hi)
        for label in np.unique(label_slice):
            if label == 0:
                continue
            for array, color in ((label_slice, "#4de5ea"), (moved_slice, "#ff813d")):
                mask = array == label
                if mask.any() and not mask.all():
                    ax.contour(mask, levels=[.5], colors=[color], linewidths=.8)
        label = "NiftyReg " + row["platform"] if row["method"] == "niftyreg" else row["method"]
        ax.set_title(f"{label}\nDice {row['dice']:.3f}")
        ax.axis("off")
    figure.suptitle(f"{name}: fixed labels cyan; moved labels orange")
    figure.tight_layout(rect=(0, 0, 1, .90))
    figure.savefig(output, dpi=160)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("files", type=Path, nargs="+")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--gpu-directory", type=Path,
                        help="Replace CPU block-matching rows with the corresponding recorded CUDA rows")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.gpu_directory:
        combined_dir = args.output_dir / "inputs"
        combined_dir.mkdir(exist_ok=True)
        combined_files = []
        for path in args.files:
            base = [dict(row, source_results=str(path)) for row in read(path) if row["method"] != "niftyreg"]
            names = {row["name"] for row in base}
            gpu_path = args.gpu_directory / (path.stem.split("_")[0] + "_niftyreg_cuda.jsonl")
            base.extend(dict(row, source_results=str(gpu_path)) for row in read(gpu_path)
                        if row["method"] == "niftyreg" and row["name"] in names)
            combined = combined_dir / path.name
            combined.write_text("".join(json.dumps(row) + "\n" for row in base))
            combined_files.append(combined)
        args.files = combined_files
    rows = [row for path in args.files for row in summarize(path)]
    keys = list(dict.fromkeys(key for row in rows for key in row))
    with (args.output_dir / "summary.csv").open("w") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    plot(args.files, args.output_dir / "per_case.png")
    for path in args.files:
        contours(path, args.output_dir / (path.stem + "_alignment.png"))


if __name__ == "__main__":
    main()
