from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import numpy as np
from mlflow.tracking import MlflowClient


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize Grad-CAM adversarial experiment runs from local MLflow."
    )
    parser.add_argument("--tracking-db", type=Path, default=Path("mlflow.db"))
    parser.add_argument("--experiment-name", default="gradcam-adversarial-imagenet")
    parser.add_argument("--output-dir", type=Path, default=Path("analysis"))
    return parser.parse_args()


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def analyze(args: argparse.Namespace) -> None:
    mlflow.set_tracking_uri(f"sqlite:///{args.tracking_db.resolve()}")
    client = MlflowClient()
    experiment = client.get_experiment_by_name(args.experiment_name)
    if experiment is None:
        raise ValueError(f"MLflow experiment not found: {args.experiment_name}")
    runs = client.search_runs([experiment.experiment_id], max_results=100000)
    trial_runs = [
        run
        for run in runs
        if "layer" in run.data.params and "success" in run.data.metrics
    ]
    if not trial_runs:
        raise ValueError(f"No trial runs found in MLflow experiment {args.experiment_name}.")

    per_image: dict[tuple[str, float, str, str], list[Any]] = defaultdict(list)
    all_images_by_model: dict[str, set[str]] = defaultdict(set)
    for run in trial_runs:
        params, metrics = run.data.params, run.data.metrics
        model = params["model"]
        confidence = float(params["confidence"])
        layer = params["layer"]
        image_id = params["image_id"]
        all_images_by_model[model].add(image_id)
        succeeded = int(metrics["success"])
        if succeeded:
            per_image[(model, confidence, layer, image_id)].append(
                {
                    "linf": metrics["linf"],
                    "changed_pixel_fraction": metrics["changed_pixel_fraction"],
                }
            )

    # Select the best mask fraction per image, separately by perturbation objective.
    by_layer: dict[tuple[str, float, str], dict[str, list[float]]] = defaultdict(
        lambda: {"linf": [], "changed_pixel_fraction": []}
    )
    for (model, confidence, layer, _image_id), results in per_image.items():
        by_layer[(model, confidence, layer)]["linf"].append(min(row["linf"] for row in results))
        by_layer[(model, confidence, layer)]["changed_pixel_fraction"].append(
            min(row["changed_pixel_fraction"] for row in results)
        )

    summary_rows: list[dict[str, Any]] = []
    for (model, confidence, layer), values in sorted(by_layer.items()):
        summary_rows.append(
            {
                "model": model,
                "confidence": confidence,
                "layer": layer,
                "successful_images": len(values["linf"]),
                "median_best_linf": float(np.median(values["linf"])),
                "median_best_changed_pixel_fraction": float(
                    np.median(values["changed_pixel_fraction"])
                ),
                "success_rate": len(values["linf"]) / len(all_images_by_model[model]),
            }
        )
    if not summary_rows:
        raise ValueError("No successful trials were found; no best layer can be ranked.")

    best_rows = []
    for model in sorted({row["model"] for row in summary_rows}):
        for confidence in sorted(
            {row["confidence"] for row in summary_rows if row["model"] == model}
        ):
            candidates = [
                row
                for row in summary_rows
                if row["model"] == model and row["confidence"] == confidence
            ]
            for metric, label in (
                ("median_best_linf", "linf"),
                ("median_best_changed_pixel_fraction", "changed_pixel_fraction"),
            ):
                winner = min(candidates, key=lambda row: (-row["success_rate"], row[metric]))
                best_rows.append(
                    {
                        "model": model,
                        "confidence": confidence,
                        "objective": label,
                        "best_layer": winner["layer"],
                        "median_value": winner[metric],
                        "successful_images": winner["successful_images"],
                        "success_rate": winner["success_rate"],
                    }
                )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "layer_summary.csv", summary_rows)
    write_csv(args.output_dir / "best_layers.csv", best_rows)
    for metric, ylabel, filename in (
        ("median_best_linf", "Median minimum L-infinity perturbation", "architecture_linf.png"),
        (
            "median_best_changed_pixel_fraction",
            "Median minimum changed pixel fraction",
            "architecture_changed_pixels.png",
        ),
    ):
        figure, axis = plt.subplots(figsize=(9, 6))
        for model in sorted({row["model"] for row in summary_rows}):
            model_rows = [row for row in summary_rows if row["model"] == model]
            for layer in sorted({row["layer"] for row in model_rows}):
                layer_rows = sorted(
                    (row for row in model_rows if row["layer"] == layer),
                    key=lambda row: row["confidence"],
                )
                axis.plot(
                    [row["confidence"] for row in layer_rows],
                    [row[metric] for row in layer_rows],
                    marker="o",
                    alpha=0.75,
                    label=f"{model}: {layer}",
                )
        axis.set_xlabel("Requested target confidence")
        axis.set_ylabel(ylabel)
        axis.grid(True, alpha=0.3)
        axis.legend(fontsize="x-small", bbox_to_anchor=(1.02, 1), loc="upper left")
        figure.tight_layout()
        figure.savefig(args.output_dir / filename, dpi=150, bbox_inches="tight")
        plt.close(figure)


def main() -> None:
    args = parse_args()
    analyze(args)


if __name__ == "__main__":
    main()
