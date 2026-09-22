"""The training branches of RNNDistributedTrainer start from saved weights
only with `model_load_pretrained: true`, and refuse to fall back silently."""
from types import SimpleNamespace

import pytest
import torch

trainer_mod = pytest.importorskip("hython.itwinai.trainer")
from hython.models import ModelLogAPI  # noqa: E402

load = trainer_mod.RNNDistributedTrainer._load_pretrained_model


def _fake(tmp_path, flag):
    uri = tmp_path / "Net.pt"
    config = SimpleNamespace(
        model_load_pretrained=flag,
        model_logger={"Net": {"logger": "local", "model_component": "model",
                              "model_name": "Net", "model_uri": str(uri),
                              "log": True, "load": True}},
    )
    return SimpleNamespace(config=config, model_api=ModelLogAPI(config),
                           model=torch.nn.Linear(3, 2)), uri


def test_loads_saved_weights(tmp_path):
    fake, uri = _fake(tmp_path, True)
    saved = torch.nn.Linear(3, 2)
    torch.save(saved.state_dict(), uri)
    load(fake)
    assert torch.equal(fake.model.weight, saved.weight)


def test_missing_weights_raise(tmp_path):
    fake, _ = _fake(tmp_path, True)
    with pytest.raises(FileNotFoundError):
        load(fake)


def test_off_by_default(tmp_path):
    fake, uri = _fake(tmp_path, False)
    before = fake.model.weight.clone()
    torch.save(torch.nn.Linear(3, 2).state_dict(), uri)
    load(fake)
    assert torch.equal(fake.model.weight, before)
