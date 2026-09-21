"""H11: the calibration KGE is computed per cell and weighted by observation count."""
import math
from types import SimpleNamespace

import pytest
import torch

from hython.losses import CellKGELoss, KGELoss, compute_kge_per_cell, compute_kge_torch
from hython.trainer.base import AbstractTrainer


def _inverted_cells():
    """Two cells at different levels, each with its timing exactly inverted."""
    t = torch.linspace(0, 2 * math.pi, 50)
    wave = torch.sin(t)
    true = torch.stack([0.1 + 0.02 * wave, 0.4 + 0.02 * wave])
    pred = torch.stack([0.1 - 0.02 * wave, 0.4 - 0.02 * wave])
    return true, pred


def test_inverted_timing_is_bad_per_cell_but_good_pooled():
    true, pred = _inverted_cells()
    pooled = -KGELoss()(true.flatten(), pred.flatten())
    per_cell = -CellKGELoss()(true, pred)
    assert pooled > 0.9
    assert per_cell == pytest.approx(-1.0, abs=1e-3)


def test_matches_the_pooled_kge_of_each_cell():
    torch.manual_seed(0)
    true = torch.rand(4, 30) + 0.1
    pred = true + 0.1 * torch.randn(4, 30)
    kge, n = compute_kge_per_cell(true, pred)
    for c in range(4):
        assert kge[c] == pytest.approx(compute_kge_torch(true[c], pred[c]).item(), abs=1e-5)
    assert n.tolist() == [30] * 4


def test_weighted_by_observation_count():
    torch.manual_seed(1)
    true = torch.rand(2, 40) + 0.1
    pred = true.clone()
    pred[1] = pred[1] + 0.2 * torch.randn(40)
    true[0, 10:] = torch.nan  # cell 0: 10 observations, cell 1: 40
    kge, n = compute_kge_per_cell(true, pred)
    assert n.tolist() == [10, 40]
    expected = (kge[0] * 10 + kge[1] * 40) / 50
    assert -CellKGELoss()(true, pred) == pytest.approx(expected.item(), abs=1e-6)


def test_cells_below_two_observations_are_dropped():
    true = torch.full((3, 10), torch.nan)
    true[0] = torch.rand(10)
    true[1, 0] = 0.3  # one observation
    pred = torch.rand(3, 10)
    kge, n = compute_kge_per_cell(true, pred)
    assert n.tolist() == [10]
    assert torch.isfinite(CellKGELoss()(true, pred))


def test_no_valid_cell_gives_zero_loss_with_a_graph():
    true = torch.full((2, 5), torch.nan)
    pred = torch.rand(2, 5, requires_grad=True)
    loss = CellKGELoss()(true, pred)
    assert loss.item() == 0.0
    loss.backward()
    assert torch.all(pred.grad == 0)


def test_gradient_finite_with_missing_values_and_constant_prediction():
    torch.manual_seed(2)
    true = torch.rand(3, 20)
    true[:, ::3] = torch.nan
    pred = torch.rand(3, 20)
    pred[2] = 0.25  # constant prediction in one cell
    pred.requires_grad_(True)
    CellKGELoss()(true, pred).backward()
    assert torch.all(torch.isfinite(pred.grad))


def _batch_loss(loss_fn, prediction, target, valid_mask, scale=False):
    cfg = SimpleNamespace(
        loss_fn=loss_fn,
        model_head_layer="regression",
        data_loss_scale_proportional_valid_target_timesteps=scale,
    )
    return AbstractTrainer._compute_batch_loss(
        SimpleNamespace(cfg=cfg), prediction, target, valid_mask, {"vwc": 1.0}
    )


def test_trainer_passes_cells_not_a_flat_vector():
    true, pred = _inverted_cells()
    target = true[..., None].clone()
    target[0, ::2, 0] = torch.nan
    valid_mask = ~target.isnan()
    prediction = {"y_hat": pred[..., None]}

    loss = _batch_loss(CellKGELoss(), prediction, target, valid_mask)
    assert loss == pytest.approx(CellKGELoss()(target[..., 0], pred).item(), abs=1e-6)
    assert loss > 0.9  # per cell the timing is inverted: KGE about -1

    # The pooled loss still goes through the old flattening path
    pooled = _batch_loss(KGELoss(), prediction, target, valid_mask)
    assert pooled < -0.9


def test_trainer_keeps_the_valid_fraction_scaling():
    true, pred = _inverted_cells()
    target = true[..., None].clone()
    target[:, :25, 0] = torch.nan
    valid_mask = ~target.isnan()
    prediction = {"y_hat": pred[..., None]}
    plain = _batch_loss(CellKGELoss(), prediction, target, valid_mask)
    scaled = _batch_loss(CellKGELoss(), prediction, target, valid_mask, scale=True)
    assert scaled == pytest.approx(plain.item() * 0.5, abs=1e-6)
