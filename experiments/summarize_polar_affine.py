"""Summarize affine experiments without selecting runs by evaluation labels."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import SimpleITK as sitk


def summarize(path):
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    rigid = {row["name"]: row for row in rows if row["method"] == "rigid"}
    groups = {}
    for row in rows:
        key = (row["method"], bool(row.get("bounds")), row.get("lr", 0.))
        groups.setdefault(key, []).append(row)
    summary = []
    for (method, bounded, rate), runs in groups.items():
        result = dict(suite=path.stem, method=method, bounded=bounded, lr=rate, count=len(runs))
        for metric in ("dice", "tre_mean_mm", "tre_median_mm", "seconds", "condition", "loss_value"):
            values = [row[metric] for row in runs if metric in row]
            if values:
                result[metric] = float(np.mean(values))
        result["min_singular_value"] = min(min(row["singular_values"]) for row in runs)
        result["max_singular_value"] = max(max(row["singular_values"]) for row in runs)
        result["nonpositive_determinants"] = sum(row["determinant"] <= 0 for row in runs)
        result["extreme_scale_runs"] = sum(min(row["singular_values"]) < .5 or
                                            max(row["singular_values"]) > 2 for row in runs)
        if "dice" in result:
            differences = [row["dice"] - rigid[row["name"]]["dice"] for row in runs]
            result["dice_change_from_rigid"] = float(np.mean(differences))
            result["dice_worse_than_rigid"] = sum(delta < -.001 for delta in differences)
        if "tre_mean_mm" in result:
            result["tre_worse_than_rigid"] = sum(row["tre_mean_mm"] > rigid[row["name"]]["tre_mean_mm"] + .1
                                                for row in runs if "tre_mean_mm" in row)
            result["tre_max_mm"] = max(row["tre_mean_mm"] for row in runs if "tre_mean_mm" in row)
        summary.append(result)
    return summary


def alignment_figure(path, output, rate):
    """Draw label contours on the native image grid for the first real pair."""
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    name = rows[0]["name"]
    selected = [row for row in rows if row["name"] == name and
                (row["method"] in ("identity", "rigid") or
                 row["method"] in ("matrix", "polar") and row.get("lr") == rate and row.get("bounds"))]
    if len(selected) != 4 or "paths" not in selected[0]:
        return
    fixed_path, _, fixed_label_path, moving_label_path = selected[0]["paths"]
    reference = sitk.ReadImage(fixed_path)
    fixed = sitk.GetArrayFromImage(reference)
    fixed_labels_image = sitk.ReadImage(fixed_label_path)
    fixed_labels = sitk.GetArrayFromImage(fixed_labels_image)
    moving_labels = sitk.ReadImage(moving_label_path)
    axis = 1 if name.startswith("abdomen") else 0
    occupied = np.argwhere(fixed_labels > 0)
    index = int(np.median(occupied[:, axis]))
    fixed_slice = np.take(fixed, index, axis=axis)
    fixed_seg_slice = np.take(fixed_labels, index, axis=axis)
    lo, hi = np.percentile(fixed_slice, [2, 98])
    figure, axes = plt.subplots(1, 4, figsize=(14, 4.5))
    for ax, row in zip(axes, selected):
        matrix = np.asarray(row["matrix"])
        transform = sitk.AffineTransform(3)
        transform.SetMatrix(matrix[:3, :3].ravel().tolist())
        transform.SetTranslation(matrix[:3, 3].tolist())
        moved = sitk.Resample(moving_labels, fixed_labels_image, transform, sitk.sitkNearestNeighbor, 0.)
        moved_slice = np.take(sitk.GetArrayFromImage(moved), index, axis=axis)
        ax.imshow(fixed_slice, cmap="gray", vmin=lo, vmax=hi, origin="lower")
        for label in np.unique(fixed_seg_slice):
            if label == 0:
                continue
            for segmentation, color in ((fixed_seg_slice, "#4de5ea"), (moved_slice, "#ff813d")):
                mask = segmentation == label
                if mask.any() and not mask.all():
                    ax.contour(mask, levels=[.5], colors=[color], linewidths=.8)
        ax.set_title(f"{row['method']}\nDice {row['dice']:.3f}")
        ax.axis("off")
    figure.suptitle(f"{name}: fixed labels cyan; moved labels orange; affine LR {rate}")
    figure.tight_layout(rect=(0, 0, 1, .90))
    figure.savefig(output, dpi=160)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("files", type=Path, nargs="+")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--qc-rate", type=float, default=.003)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summaries = [summarize(path) for path in args.files]
    rows = [row for summary in summaries for row in summary]
    keys = list(dict.fromkeys(key for row in rows for key in row))
    with (args.output_dir / "summary.csv").open("w") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    figure, axes = plt.subplots(1, len(summaries), figsize=(5 * len(summaries), 4.5), squeeze=False)
    colors = {"matrix": "#3264a8", "polar": "#c55524", "matrix_legacy": "#777777"}
    for ax, path, summary in zip(axes[0], args.files, summaries):
        metric = "tre_mean_mm" if path.stem.startswith("synthetic") else "dice"
        for method in colors:
            for bounded in (False, True):
                selected = sorted([row for row in summary if row["method"] == method and row["bounded"] == bounded],
                                  key=lambda row: row["lr"])
                if not selected:
                    continue
                label = method + (", bounded" if bounded else ", unbounded")
                ax.plot([row["lr"] for row in selected], [row[metric] for row in selected],
                        marker="o", linestyle="-" if bounded else "--", color=colors[method], label=label)
        baseline = next(row for row in summary if row["method"] == "rigid")
        ax.axhline(baseline[metric], color="black", linewidth=1, label="rigid")
        ax.set_xscale("log")
        if metric == "tre_mean_mm":
            ax.set_yscale("log")
        ax.set_xlabel("Affine learning rate")
        ax.set_ylabel("Mean point error (mm)" if metric == "tre_mean_mm" else "Mean Dice")
        ax.set_title(path.stem.replace("_", " "))
        ax.grid(alpha=.2)
    axes[0, -1].legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(args.output_dir / "learning_rates.png", dpi=180)
    plt.close(figure)
    for path in args.files:
        if not path.stem.startswith("synthetic"):
            alignment_figure(path, args.output_dir / (path.stem + "_alignment.png"), args.qc_rate)


if __name__ == "__main__":
    main()
