from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


@dataclass
class AttackResult:
    adversarial: torch.Tensor
    succeeded: bool
    target_confidence: float
    prediction: int
    linf: float
    l1: float
    l2: float
    changed_pixels: int
    changed_pixel_fraction: float
    channel_values_changed: int


class GradCAM:
    """Compute a class-conditional Grad-CAM map for one model layer."""

    def __init__(self, model: nn.Module, layer: nn.Module):
        self.model = model
        self.layer = layer

    def __call__(
        self,
        image: torch.Tensor,
        target_class: int,
        normalize: Any,
    ) -> torch.Tensor:
        activation: torch.Tensor | None = None

        def capture_activation(_module: nn.Module, _inputs: tuple, output: torch.Tensor) -> None:
            nonlocal activation
            if not isinstance(output, torch.Tensor) or output.ndim != 4:
                raise ValueError("Grad-CAM requires a layer with a 4D feature-map output.")
            activation = output

        handle = self.layer.register_forward_hook(capture_activation)
        try:
            logits = self.model(normalize(image))
        finally:
            handle.remove()
        if activation is None:
            raise RuntimeError("The selected layer did not run during the model forward pass.")

        gradients = torch.autograd.grad(logits[:, target_class].sum(), activation)[0]
        weights = gradients.mean(dim=(2, 3), keepdim=True)
        cam = F.relu((weights * activation).sum(dim=1, keepdim=True))
        cam = F.interpolate(cam, size=image.shape[-2:], mode="bilinear", align_corners=False)
        minimum = cam.amin(dim=(2, 3), keepdim=True)
        maximum = cam.amax(dim=(2, 3), keepdim=True)
        span = maximum - minimum
        if torch.any(span <= torch.finfo(cam.dtype).eps):
            raise ValueError("Grad-CAM is constant for this layer and target class.")
        return (cam - minimum) / span


def top_fraction_mask(cam: torch.Tensor, fraction: float) -> torch.Tensor:
    if not 0 < fraction <= 1:
        raise ValueError("Mask fractions must be in the interval (0, 1].")
    flat = cam.flatten(start_dim=2)
    count = max(1, int(np.ceil(flat.shape[-1] * fraction)))
    indices = flat.topk(count, dim=-1).indices
    mask = torch.zeros_like(flat)
    mask.scatter_(-1, indices, 1)
    return mask.view_as(cam)


def _predict(
    model: nn.Module,
    image: torch.Tensor,
    target_class: int,
    normalize: Any,
) -> tuple[float, int]:
    with torch.no_grad():
        probabilities = model(normalize(image)).softmax(dim=1)[0]
    return float(probabilities[target_class]), int(probabilities.argmax())


def _attack_at_budget(
    model: nn.Module,
    clean: torch.Tensor,
    target_class: int,
    confidence: float,
    mask: torch.Tensor,
    normalize: Any,
    budget: float,
    steps: int,
) -> tuple[torch.Tensor, float, int, bool]:
    perturbation = torch.zeros_like(clean)
    best_image = clean
    best_confidence, best_prediction = _predict(model, clean, target_class, normalize)
    best_succeeded = best_prediction == target_class and best_confidence >= confidence
    step_size = max((2 * budget / steps), 1e-6)

    for _ in range(steps):
        perturbation.requires_grad_(True)
        candidate = (clean + perturbation).clamp(0, 1)
        logits = model(normalize(candidate))
        loss = F.cross_entropy(logits, torch.tensor([target_class], device=clean.device))
        gradient = torch.autograd.grad(loss, perturbation)[0]
        with torch.no_grad():
            perturbation = perturbation - step_size * gradient.sign() * mask
            perturbation.clamp_(-budget, budget)
            perturbation.mul_(mask)
            candidate = (clean + perturbation).clamp(0, 1)
            probability, prediction = _predict(model, candidate, target_class, normalize)
            succeeded = prediction == target_class and probability >= confidence
            if probability > best_confidence or succeeded:
                best_image = candidate.detach()
                best_confidence = probability
                best_prediction = prediction
                best_succeeded = succeeded
            if succeeded:
                break

    return best_image, best_confidence, best_prediction, best_succeeded


def targeted_attack(
    model: nn.Module,
    clean: torch.Tensor,
    target_class: int,
    confidence: float,
    mask: torch.Tensor,
    normalize: Any,
    max_epsilon: float,
    binary_search_steps: int = 8,
    attack_steps: int = 40,
) -> AttackResult:
    if not 0 < confidence < 1:
        raise ValueError("Target confidence must be strictly between 0 and 1.")
    if max_epsilon <= 0 or binary_search_steps < 1 or attack_steps < 1:
        raise ValueError("Epsilon and iteration counts must be positive.")

    low, high = 0.0, max_epsilon
    best_image = clean.detach()
    best_confidence, best_prediction = _predict(model, clean, target_class, normalize)
    best_succeeded = best_prediction == target_class and best_confidence >= confidence

    if not best_succeeded:
        image, probability, prediction, succeeded = _attack_at_budget(
            model, clean, target_class, confidence, mask, normalize, high, attack_steps
        )
        if probability > best_confidence or succeeded:
            best_image, best_confidence, best_prediction = image, probability, prediction
        best_succeeded = succeeded
        if succeeded:
            best_image = image

        for _ in range(binary_search_steps):
            budget = (low + high) / 2
            image, probability, prediction, succeeded = _attack_at_budget(
                model, clean, target_class, confidence, mask, normalize, budget, attack_steps
            )
            if succeeded:
                high = budget
                best_image, best_confidence, best_prediction = image, probability, prediction
                best_succeeded = True
            else:
                low = budget
                if not best_succeeded and probability > best_confidence:
                    best_image, best_confidence, best_prediction = image, probability, prediction

    difference = (best_image - clean).detach().abs()
    per_pixel_difference = difference.amax(dim=1, keepdim=True)
    changed = per_pixel_difference > (1 / 255)
    changed_pixels = int(changed.sum().item())
    pixel_total = int(changed.numel())
    return AttackResult(
        adversarial=best_image.detach(),
        succeeded=best_succeeded,
        target_confidence=best_confidence,
        prediction=best_prediction,
        linf=float(difference.max().item()),
        l1=float(difference.mean().item()),
        l2=float(torch.linalg.vector_norm(difference).item()),
        changed_pixels=changed_pixels,
        changed_pixel_fraction=changed_pixels / pixel_total,
        channel_values_changed=int((difference > (1 / 255)).sum().item()),
    )


def gradcam_overlay(image: torch.Tensor, cam: torch.Tensor) -> np.ndarray:
    source = image[0].detach().cpu().permute(1, 2, 0).numpy()
    heatmap = cam[0, 0].detach().cpu().numpy()
    heatmap = np.clip(heatmap, 0, 1)
    colors = np.stack((heatmap, np.zeros_like(heatmap), 1 - heatmap), axis=-1)
    return np.clip(0.55 * source + 0.45 * colors, 0, 1)
