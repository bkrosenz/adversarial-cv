from __future__ import annotations

import argparse
import csv
import hashlib
import logging
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import torch
from PIL import Image
from torch import nn
from torchvision import models

from adversarial_cv.attacks import GradCAM, gradcam_overlay, targeted_attack, top_fraction_mask

LOGGER = logging.getLogger("adversarial_cv")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sweep Grad-CAM layers for targeted ImageNet classifier attacks."
    )
    parser.add_argument("--images", type=Path, required=True, help="Image file or directory.")
    parser.add_argument(
        "--models",
        nargs="+",
        default=["vgg16"],
        help="torchvision classification model names (default: vgg16).",
    )
    parser.add_argument(
        "--layers",
        nargs="*",
        help="Exact module names to test; default is every Conv2d layer.",
    )
    parser.add_argument("--target-class", type=int, help="ImageNet class index; default is top non-predicted class.")
    parser.add_argument(
        "--confidences",
        nargs="+",
        type=float,
        default=[0.5, 0.75, 0.9],
        help="Required target-class probabilities to sweep.",
    )
    parser.add_argument(
        "--mask-fractions",
        nargs="+",
        type=float,
        default=[0.05, 0.15, 0.3],
        help="Fractions of highest Grad-CAM pixels allowed to change.",
    )
    parser.add_argument("--max-images", type=int, default=1, help="Maximum images per model; 0 means no limit.")
    parser.add_argument("--max-epsilon", type=float, default=0.15, help="Maximum L-infinity perturbation in [0,1] pixel space.")
    parser.add_argument("--binary-search-steps", type=int, default=6)
    parser.add_argument("--attack-steps", type=int, default=30)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or another torch device.")
    parser.add_argument("--tracking-db", type=Path, default=Path("mlflow.db"))
    parser.add_argument("--artifact-dir", type=Path, default=Path("mlruns"))
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    parser.add_argument("--experiment-name", default="gradcam-adversarial-imagenet")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def image_paths(source: Path, limit: int) -> list[Path]:
    if source.is_file():
        paths = [source]
    elif source.is_dir():
        paths = sorted(path for path in source.rglob("*") if path.suffix.lower() in IMAGE_EXTENSIONS)
    else:
        raise FileNotFoundError(f"Image source does not exist: {source}")
    paths = [path for path in paths if path.suffix.lower() in IMAGE_EXTENSIONS]
    if not paths:
        raise ValueError(f"No supported images found in {source}")
    return paths if limit == 0 else paths[:limit]


def load_model(name: str, device: torch.device) -> tuple[nn.Module, Any, Any, str]:
    try:
        weights = models.get_model_weights(name).DEFAULT
        model = models.get_model(name, weights=weights)
    except (AttributeError, KeyError, ValueError) as error:
        raise ValueError(f"Unknown torchvision model or unavailable DEFAULT weights: {name}") from error
    model = model.to(device).eval()
    transforms = weights.transforms()
    mean = torch.tensor(transforms.mean, device=device).view(1, 3, 1, 1)
    std = torch.tensor(transforms.std, device=device).view(1, 3, 1, 1)

    def normalize(image: torch.Tensor) -> torch.Tensor:
        return (image - mean) / std

    return model, transforms, normalize, f"{name}:{weights.name}"


def load_image(path: Path, transforms: Any, device: torch.device) -> torch.Tensor:
    with Image.open(path) as image:
        normalized = transforms(image.convert("RGB")).unsqueeze(0).to(device)
    mean = torch.tensor(transforms.mean, device=device).view(1, 3, 1, 1)
    std = torch.tensor(transforms.std, device=device).view(1, 3, 1, 1)
    return (normalized * std + mean).clamp(0, 1)


def prediction_name(class_id: int, categories: list[str]) -> str:
    return categories[class_id] if class_id < len(categories) else str(class_id)


def layer_modules(model: nn.Module, requested: list[str] | None) -> list[tuple[str, nn.Module]]:
    named_layers = dict(model.named_modules())
    if requested:
        missing = [name for name in requested if name not in named_layers]
        if missing:
            raise ValueError(f"Unknown layer name(s): {', '.join(missing)}")
        selected = [(name, named_layers[name]) for name in requested]
        invalid = [name for name, layer in selected if not isinstance(layer, nn.Conv2d)]
        if invalid:
            raise ValueError(f"Grad-CAM layer(s) must be Conv2d modules: {', '.join(invalid)}")
        return selected
    return [(name, layer) for name, layer in model.named_modules() if isinstance(layer, nn.Conv2d)]


def save_visualization(
    clean: torch.Tensor,
    adversarial: torch.Tensor,
    cam: torch.Tensor,
    output: Path,
    title: str,
) -> None:
    clean_np = clean[0].detach().cpu().permute(1, 2, 0).numpy()
    adv_np = adversarial[0].detach().cpu().permute(1, 2, 0).numpy()
    difference = np.abs(adv_np - clean_np)
    figure, axes = plt.subplots(1, 4, figsize=(14, 4))
    axes[0].imshow(clean_np)
    axes[0].set_title("Original")
    axes[1].imshow(gradcam_overlay(clean, cam))
    axes[1].set_title("Grad-CAM")
    axes[2].imshow(adv_np)
    axes[2].set_title("Adversarial")
    axes[3].imshow(difference / max(float(difference.max()), 1e-8))
    axes[3].set_title("Absolute difference")
    for axis in axes:
        axis.axis("off")
    figure.suptitle(title)
    figure.tight_layout()
    figure.savefig(output, dpi=140)
    plt.close(figure)


def save_raw_data(
    output: Path,
    clean: torch.Tensor,
    adversarial: torch.Tensor,
    cam: torch.Tensor,
    mask: torch.Tensor,
) -> None:
    np.savez_compressed(
        output,
        original=clean[0].detach().cpu().numpy().astype(np.float32),
        adversarial=adversarial[0].detach().cpu().numpy().astype(np.float32),
        gradcam=cam[0, 0].detach().cpu().numpy().astype(np.float32),
        mask=mask[0, 0].detach().cpu().numpy().astype(np.uint8),
    )


def summarize(results: list[dict[str, Any]], output_dir: Path) -> None:
    if not results:
        return
    csv_path = output_dir / "results.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)

    successful = [result for result in results if result["success"]]
    if not successful:
        LOGGER.warning("No attack reached its requested confidence; saved raw trials only.")
        return
    for metric, ylabel, filename in (
        ("linf", "Minimum L-infinity perturbation", "confidence_vs_linf.png"),
        ("changed_pixel_fraction", "Changed pixel fraction", "confidence_vs_changed_pixels.png"),
    ):
        figure, axis = plt.subplots(figsize=(8, 5))
        for model_name in sorted({row["model"] for row in successful}):
            confidences = sorted(
                {row["confidence"] for row in successful if row["model"] == model_name}
            )
            values = []
            for confidence in confidences:
                subset = [
                    row[metric]
                    for row in successful
                    if row["model"] == model_name and row["confidence"] == confidence
                ]
                values.append(float(np.median(subset)))
            axis.plot(confidences, values, marker="o", label=model_name)
        axis.set_xlabel("Requested target confidence")
        axis.set_ylabel(ylabel)
        axis.grid(True, alpha=0.3)
        axis.legend()
        figure.tight_layout()
        figure.savefig(output_dir / filename, dpi=140)
        plt.close(figure)


def run(args: argparse.Namespace) -> None:
    if args.max_images < 0:
        raise ValueError("--max-images cannot be negative.")
    if any(not 0 < confidence < 1 for confidence in args.confidences):
        raise ValueError("Every confidence must be strictly between 0 and 1.")
    if any(not 0 < fraction <= 1 for fraction in args.mask_fractions):
        raise ValueError("Every mask fraction must be in (0, 1].")

    torch.manual_seed(args.seed)
    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available() else
        "cpu" if args.device == "auto" else args.device
    )
    tracking_db = args.tracking_db.resolve()
    artifact_dir = args.artifact_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(f"sqlite:///{tracking_db}")
    experiment = mlflow.set_experiment(
        args.experiment_name,
        artifact_location=artifact_dir.as_uri(),
    )
    all_results: list[dict[str, Any]] = []
    layer_failures: list[dict[str, str]] = []
    skipped_images: list[dict[str, str]] = []
    model_names = args.models

    with mlflow.start_run(run_name="experiment-sweep") as parent_run:
        mlflow.log_params(
            {
                "models": ",".join(model_names),
                "confidences": ",".join(map(str, args.confidences)),
                "mask_fractions": ",".join(map(str, args.mask_fractions)),
                "max_epsilon": args.max_epsilon,
                "binary_search_steps": args.binary_search_steps,
                "attack_steps": args.attack_steps,
                "seed": args.seed,
                "device": str(device),
            }
        )
        for model_name in model_names:
            model, transforms, normalize, weights_name = load_model(model_name, device)
            categories = list(models.get_model_weights(model_name).DEFAULT.meta["categories"])
            layers = layer_modules(model, args.layers)
            if not layers:
                raise ValueError(f"Model {model_name} has no Conv2d layers to inspect.")
            for image_path in image_paths(args.images, args.max_images):
                clean = load_image(image_path, transforms, device)
                with torch.no_grad():
                    initial_probabilities = model(normalize(clean)).softmax(dim=1)[0]
                original_class = int(initial_probabilities.argmax())
                target_class = (
                    args.target_class
                    if args.target_class is not None
                    else int(initial_probabilities.topk(2).indices[-1])
                )
                if not 0 <= target_class < initial_probabilities.numel():
                    raise ValueError(f"Target class index {target_class} is outside the model output.")
                if target_class == original_class:
                    skipped = {
                        "model": model_name,
                        "weights": weights_name,
                        "image_id": hashlib.sha256(
                            str(image_path.resolve()).encode()
                        ).hexdigest()[:12],
                        "image": str(image_path),
                        "original_class": str(original_class),
                        "target_class": str(target_class),
                        "reason": "Target class equals the clean-image prediction.",
                    }
                    skipped_images.append(skipped)
                    LOGGER.warning(
                        "Skipping %s / %s: target class %d is already predicted.",
                        model_name,
                        image_path,
                        target_class,
                    )
                    with mlflow.start_run(
                        run_name=f"{model_name}-{skipped['image_id']}-same-class",
                        nested=True,
                    ):
                        mlflow.log_params(
                            {
                                "model": model_name,
                                "weights": weights_name,
                                "image_id": skipped["image_id"],
                                "image": str(image_path),
                                "original_class": original_class,
                                "target_class": target_class,
                            }
                        )
                        mlflow.set_tag("status", "skipped_same_class")
                    continue
                image_id = hashlib.sha256(str(image_path.resolve()).encode()).hexdigest()[:12]
                LOGGER.info(
                    "%s | %s: %s (%d) -> %s (%d)",
                    model_name,
                    image_path,
                    prediction_name(original_class, categories),
                    original_class,
                    prediction_name(target_class, categories),
                    target_class,
                )

                for layer_name, layer in layers:
                    try:
                        cam = GradCAM(model, layer)(clean, target_class, normalize)
                    except ValueError as error:
                        failure = {
                            "model": model_name,
                            "weights": weights_name,
                            "image_id": image_id,
                            "image": str(image_path),
                            "layer": layer_name,
                            "target_class": str(target_class),
                            "error": str(error),
                        }
                        layer_failures.append(failure)
                        LOGGER.warning(
                            "Skipping %s / %s / %s: Grad-CAM unavailable: %s",
                            model_name,
                            image_path,
                            layer_name,
                            error,
                        )
                        with mlflow.start_run(
                            run_name=f"{model_name}-{layer_name}-{image_id}-gradcam-error",
                            nested=True,
                        ):
                            mlflow.log_params(
                                {
                                    "model": model_name,
                                    "weights": weights_name,
                                    "image_id": image_id,
                                    "image": str(image_path),
                                    "layer": layer_name,
                                    "target_class": target_class,
                                }
                            )
                            mlflow.set_tag("status", "gradcam_error")
                            mlflow.set_tag("error", str(error))
                        continue
                    for fraction in args.mask_fractions:
                        mask = top_fraction_mask(cam, fraction)
                        for confidence in args.confidences:
                            result = targeted_attack(
                                model=model,
                                clean=clean,
                                target_class=target_class,
                                confidence=confidence,
                                mask=mask,
                                normalize=normalize,
                                max_epsilon=args.max_epsilon,
                                binary_search_steps=args.binary_search_steps,
                                attack_steps=args.attack_steps,
                            )
                            record = {
                                "model": model_name,
                                "image_id": image_id,
                                "image": str(image_path),
                                "layer": layer_name,
                                "original_class": original_class,
                                "target_class": target_class,
                                "confidence": confidence,
                                "mask_fraction": fraction,
                                "success": int(result.succeeded),
                                "target_probability": result.target_confidence,
                                "adversarial_prediction": result.prediction,
                                "linf": result.linf,
                                "l1_mean": result.l1,
                                "l2": result.l2,
                                "changed_pixels": result.changed_pixels,
                                "changed_pixel_fraction": result.changed_pixel_fraction,
                                "changed_channels": result.channel_values_changed,
                            }
                            all_results.append(record)
                            with mlflow.start_run(
                                run_name=f"{model_name}-{layer_name}-{image_id}-c{confidence:g}-m{fraction:g}",
                                nested=True,
                            ):
                                mlflow.log_params(
                                    {
                                        "model": model_name,
                                        "image_id": image_id,
                                        "image": str(image_path),
                                        "layer": layer_name,
                                        "original_class": original_class,
                                        "target_class": target_class,
                                        "confidence": confidence,
                                        "mask_fraction": fraction,
                                        "max_epsilon": args.max_epsilon,
                                    }
                                )
                                mlflow.log_metrics(
                                    {
                                        "success": int(result.succeeded),
                                        "target_probability": result.target_confidence,
                                        "adversarial_prediction": result.prediction,
                                        "linf": result.linf,
                                        "l1_mean": result.l1,
                                        "l2": result.l2,
                                        "changed_pixels": result.changed_pixels,
                                        "changed_pixel_fraction": result.changed_pixel_fraction,
                                        "changed_channels": result.channel_values_changed,
                                    }
                                )
                                mlflow.set_tag("original_label", prediction_name(original_class, categories))
                                mlflow.set_tag("target_label", prediction_name(target_class, categories))
                                mlflow.set_tag("success", str(result.succeeded).lower())
                                artifact_root = args.output_dir / model_name / image_id / layer_name.replace("/", "_")
                                artifact_root.mkdir(parents=True, exist_ok=True)
                                figure_path = artifact_root / f"c{confidence:g}_m{fraction:g}.png"
                                raw_path = artifact_root / f"c{confidence:g}_m{fraction:g}.npz"
                                save_visualization(
                                    clean,
                                    result.adversarial,
                                    cam,
                                    figure_path,
                                    f"{model_name} | {layer_name} | target {target_class} @ {confidence:.2f}",
                                )
                                save_raw_data(raw_path, clean, result.adversarial, cam, mask)
                                mlflow.log_artifact(str(figure_path), artifact_path="visualizations")
                                mlflow.log_artifact(str(raw_path), artifact_path="raw")
                del model
                if device.type == "cuda":
                    torch.cuda.empty_cache()

        summarize(all_results, args.output_dir)
        if layer_failures:
            failures_path = args.output_dir / "layer_failures.csv"
            with failures_path.open("w", newline="", encoding="utf-8") as file:
                writer = csv.DictWriter(file, fieldnames=list(layer_failures[0]))
                writer.writeheader()
                writer.writerows(layer_failures)
            mlflow.log_artifact(str(failures_path), artifact_path="aggregate")
        if skipped_images:
            skipped_path = args.output_dir / "skipped_images.csv"
            with skipped_path.open("w", newline="", encoding="utf-8") as file:
                writer = csv.DictWriter(file, fieldnames=list(skipped_images[0]))
                writer.writeheader()
                writer.writerows(skipped_images)
            mlflow.log_artifact(str(skipped_path), artifact_path="aggregate")
        if all_results:
            mlflow.log_artifact(str(args.output_dir / "results.csv"), artifact_path="aggregate")
            for figure_name in ("confidence_vs_linf.png", "confidence_vs_changed_pixels.png"):
                figure_path = args.output_dir / figure_name
                if figure_path.exists():
                    mlflow.log_artifact(str(figure_path), artifact_path="aggregate")
            mlflow.log_metrics(
                {
                    "trial_count": len(all_results),
                    "success_rate": sum(row["success"] for row in all_results) / len(all_results),
                }
            )
        mlflow.set_tag("experiment_id", experiment.experiment_id)
        mlflow.set_tag("parent_run_id", parent_run.info.run_id)
        mlflow.log_dict(
            {
                "tracking_uri": mlflow.get_tracking_uri(),
                "artifact_directory": str(artifact_dir),
                "output_directory": str(args.output_dir.resolve()),
            },
            "run_manifest.json",
        )


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    run(args)


if __name__ == "__main__":
    main()
