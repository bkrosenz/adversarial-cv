import torch
from torch import nn

from adversarial_cv.attacks import targeted_attack, top_fraction_mask


def test_top_fraction_mask_selects_expected_number_of_pixels() -> None:
    cam = torch.tensor([[[[0.0, 0.1], [0.8, 1.0]]]])

    mask = top_fraction_mask(cam, 0.5)

    assert int(mask.sum()) == 2
    assert mask[0, 0, 1, 0] == 1
    assert mask[0, 0, 1, 1] == 1


def test_targeted_attack_reaches_target_within_budget() -> None:
    model = nn.Sequential(nn.Flatten(), nn.Linear(12, 2))
    with torch.no_grad():
        model[1].weight.zero_()
        model[1].bias[:] = torch.tensor([1.0, 0.0])
        model[1].weight[1, 0] = 20.0
    clean = torch.zeros(1, 3, 2, 2)
    mask = torch.ones(1, 1, 2, 2)

    result = targeted_attack(
        model=model,
        clean=clean,
        target_class=1,
        confidence=0.8,
        mask=mask,
        normalize=lambda image: image,
        max_epsilon=0.15,
        binary_search_steps=4,
        attack_steps=20,
    )

    assert result.succeeded
    assert result.prediction == 1
    assert result.target_confidence >= 0.8
    assert result.linf <= 0.15
