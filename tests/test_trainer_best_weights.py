"""RNNDistributedTrainer.train() ends with the weights of the best validation
epoch, in memory and in the final save - not those of the last epoch."""
from types import MethodType, SimpleNamespace

import pytest
import torch

trainer_mod = pytest.importorskip("hython.itwinai.trainer")
Trainer = trainer_mod.RNNDistributedTrainer

VAL_LOSSES = [0.5, 0.1, 0.3, 0.4]   # best at epoch 1


class _Tracker:
    def __init__(self, **kwargs):
        pass

    def add_epoch_time(self, *args):
        pass


def _fake(val_losses, patience=None):
    model = torch.nn.Linear(1, 1, bias=False)
    torch.nn.init.zeros_(model.weight)
    saved = []
    epoch = {"i": 0}

    def train_valid_epoch(model, train_loader, val_loader, device):
        # in place, as an optimizer step is: weight = epoch + 1
        with torch.no_grad():
            model.weight.add_(1.0)
        loss = torch.tensor(val_losses[epoch["i"]])
        epoch["i"] += 1
        return loss, {"vwc": {}}, loss, {"vwc": {}}

    fake = SimpleNamespace(
        model=model,
        epochs=len(val_losses),
        profiler=None,
        train_loader=None,
        val_loader=None,
        model_logger="local",
        config=SimpleNamespace(target_variables=["vwc"],
                               early_stopping_patience=patience,
                               early_stopping_min_delta=0.0),
        strategy=SimpleNamespace(is_main_worker=True, is_distributed=False,
                                 name="none", device=lambda: "cpu",
                                 gather=lambda x, dst_rank=0: [x]),
        hython_trainer=SimpleNamespace(
            train_valid_epoch=train_valid_epoch,
            lr_scheduler=SimpleNamespace(step=lambda loss: None)),
        model_api=SimpleNamespace(
            get_model_log_names=lambda: {"model": "Net"},
            log_model=lambda name, item: saved.append(item.weight.detach().clone())),
        log=lambda **kwargs: None,
    )
    for name in ("_early_stop_agreed", "set_epoch", "logging"):
        setattr(fake, name, MethodType(getattr(Trainer, name), fake))
    return fake, saved


@pytest.fixture(autouse=True)
def _no_side_effects(monkeypatch):
    monkeypatch.setattr(trainer_mod, "EpochTimeTracker", _Tracker)
    monkeypatch.setattr(trainer_mod, "train", SimpleNamespace(report=lambda d: None))


def test_restores_best_epoch():
    fake, saved = _fake(VAL_LOSSES)
    Trainer.train(fake)
    assert fake.model.weight.item() == 2.0          # after epoch 1
    assert saved[-1].item() == 2.0                  # the final save too


def test_early_stop_keeps_best():
    fake, saved = _fake([0.5, 0.1, 0.3, 0.4, 0.6, 0.7], patience=2)
    Trainer.train(fake)
    assert fake.model.weight.item() == 2.0
    assert saved[-1].item() == 2.0
