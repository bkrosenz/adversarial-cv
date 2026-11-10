# Grad-CAM-guided adversarial image experiments

This project studies targeted adversarial examples for pretrained torchvision
ImageNet classifiers. It compares convolutional layers by restricting
perturbations to the highest-scoring pixels in a target-class Grad-CAM map, then
measures the minimum perturbation found while requiring the target class to be
the predicted label at a requested softmax confidence.

The code operates on image files supplied by the researcher and local
torchvision models. It does not send images or results to a remote tracking
service. Downloaded model weights are managed by torchvision.

## Setup

Use Python 3.10 or later. Install PyTorch and torchvision using the instructions
for your CPU/CUDA platform at [pytorch.org](https://pytorch.org/get-started/locally/),
then install this project:

```bash
python -m pip install -e .
```

## Run a small experiment

Use a local image (or a directory containing images) and sweep architectures,
Grad-CAM layers, confidence targets, and allowed mask area:

```bash
adversarial-cv-run \
  --images /path/to/imagenet/validation \
  --models vgg16 resnet50 \
  --max-images 5 \
  --confidences 0.5 0.75 0.9 0.99 \
  --mask-fractions 0.02 0.05 0.1 0.2 \
  --tracking-db mlflow.db \
  --artifact-dir mlruns \
  --output-dir results
```

Start with one model, one image, and a few layers when checking a new setup.
The default layer set is every `Conv2d` module; this, multiple images, many
confidence levels, and a dense mask-fraction sweep can take substantial time.
Use `--layers features.28 features.30` (for example) to select exact module
names. `--target-class` fixes the incorrect target class index; if omitted, the
second-highest-scoring class for each clean image is used. Models use their
torchvision `DEFAULT` ImageNet weights and weight-specific preprocessing.

Choose a CUDA device with `--device cuda` or let `--device auto` select CUDA
when available. `--max-epsilon` is a maximum per-channel L-infinity budget in
un-normalized `[0, 1]` RGB space; the attack binary-searches this budget. Each
attack uses targeted cross-entropy PGD with a fixed Grad-CAM-derived spatial
mask. An example succeeds only when the target is the top prediction and its
softmax probability reaches the requested confidence.

## Tracking and artifacts

MLflow uses a local SQLite tracking database (`mlflow.db` by default) and a
local artifact directory (`mlruns`). The top-level run represents a sweep;
each image/model/layer/mask/confidence trial is a nested run. Trial runs record
parameters, success and perturbation metrics, an original/Grad-CAM/adversarial/
difference visualization, and compressed NumPy arrays containing the original
and adversarial RGB tensors, Grad-CAM map, and mask. The sweep run also records
`results.csv`, confidence plots, and a run manifest. Human-readable outputs are
written to `results/`.

Open the local tracking UI with:

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns
```

## Analyze architecture and confidence effects

After one or more runs, rank the best layer independently for each model and
confidence, using either changed-pixel fraction or L-infinity magnitude as the
objective:

```bash
adversarial-cv-analyze \
  --tracking-db mlflow.db \
  --experiment-name gradcam-adversarial-imagenet \
  --output-dir analysis
```

The analyzer writes `layer_summary.csv`, `best_layers.csv`, and plots of
confidence versus each perturbation objective. The summary first selects the
best mask fraction per image/layer/confidence, then reports the median across
successful images. Best-layer ranking prioritizes the fraction of input images
with a successful attack, then minimizes the selected perturbation objective.
Compare architectures by their best-layer names, objective values, success
rates, and how these change as the requested confidence rises.

## Interpretation and limitations

- The pixel-count metric counts spatial pixels whose maximum channel change is
  greater than `1/255`; L-infinity, mean absolute change, and L2 are reported
  separately. These thresholds are in RGB `[0, 1]` space.
- The Grad-CAM map is computed once from the clean image for the chosen target
  and layer. The mask is fixed during optimization. This tests where a layer's
  target-class explanation points, not whether recomputing Grad-CAM on each
  iterate would improve the attack.
- The reported minimum is the best successful budget found by finite-step
  projected gradient descent and binary search, not a proof of a global
  minimum. A trial that does not reach the confidence target within the
  configured budget is marked unsuccessful; it must not be interpreted as a
  measured minimum.
- Compare multiple images and target classes before drawing conclusions about
  architecture. If target classes are not specified, the second-ranked class
  is a reproducible but potentially easier-than-average target.
- The default sweep is intended for exploration, not a statistically powered
  ImageNet benchmark. Record the model weights, image selection, seed, device,
  search steps, and confidence/mask grids when reporting results.

## Tests

```bash
python -m pytest
```
