"""Does the surrogate learn d(vwc)/d(theta)? (H1, the half that needs no wflow)

`test_pool_archive.py` shows the *dataset* pairs the right rows. This shows the
rest of the path works: the statics reach the model, they differ per run, and
a gradient from the loss can move the prediction in response to them. If any of
that is broken the model can only ever output one value per cell, calibration
has nothing to optimise, and nothing else in the suite would notice.

The archive here is synthetic, with `vwc` a clean declining function of
`KsatVer` - fast soil drains, so it holds less water. That makes the test a
statement about plumbing and gradient flow, **not** about hydrology. Whether
the surrogate can pick up the real, noisier wflow signal is the wflow-based H1
test, and this does not replace it.

CPU, seeded, a few seconds.
"""

import json

import numpy as np
import pytest
import torch

POOL_SIZE = 24
RUNS = 2
NTIME = 140
SEQ_LEN = 10
KSAT_FACTOR = 3.0

STATIC_VARS = ["KsatVer", "thetaS", "SoilThickness", "Slope"]
FORCING_VARS = ["precip", "pet", "temp"]

# vwc = BASE - SLOPE * log10(KsatVer) + a little weather, so run 1 (KsatVer x 3)
# sits a known distance below run 0: SLOPE * log10(3).
BASE, SLOPE, WEATHER, NOISE = 0.45, 0.06, 0.004, 0.002
TRUE_GAP = SLOPE * np.log10(KSAT_FACTOR)


def _run(run, pool, time, xr):
    n = len(pool)
    sr = np.random.default_rng(1)
    ksat = sr.uniform(10, 500, n).astype("float32") * (KSAT_FACTOR ** run)

    static = xr.Dataset(
        {v: ("cell", sr.uniform(0.1, 10, n).astype("float32")) for v in STATIC_VARS}
        | {"KsatVer": ("cell", ksat),
           "mask_missing": ("cell", np.zeros(n, bool)),
           "mask_lake": ("cell", np.zeros(n, bool))}
    )

    # identical forcing in every run - only KsatVer, and so vwc, may move
    fr = np.random.default_rng(0)
    forcing = {v: fr.uniform(0, 5, (len(time), n)).astype("float32")
               for v in FORCING_VARS}
    nr = np.random.default_rng(100 + run)
    vwc = (BASE
           - SLOPE * np.log10(ksat)[None, :]
           + WEATHER * forcing["precip"]
           + nr.normal(0, NOISE, (len(time), n))).astype("float32")

    dynamic = xr.Dataset({v: (("time", "cell"), a) for v, a in forcing.items()}
                         | {"vwc": (("time", "cell"), vwc)})

    for ds in (static, dynamic):
        ds.coords.update({
            "lat": ("cell", pool[:, 0] * 0.01 + 45.0),
            "lon": ("cell", pool[:, 1] * 0.01 + 10.0),
            "lat_i": ("cell", pool[:, 0].astype("int32")),
            "lon_i": ("cell", pool[:, 1].astype("int32")),
            "run": ("cell", np.full(n, run, "int32")),
            "cycle": ("cell", np.zeros(n, "int32")),
            "member": ("cell", np.full(n, run, "int32")),
        })
    return static, dynamic.assign_coords(time=time)


@pytest.fixture(scope="module")
def archive_dir(tmp_path_factory):
    xr = pytest.importorskip("xarray")
    tmp = tmp_path_factory.mktemp("sensitivity")
    pool = np.stack([np.arange(POOL_SIZE) % 7, np.arange(POOL_SIZE) % 11], axis=1)
    time = xr.date_range("2017-01-01", periods=NTIME, freq="D")

    manifest = []
    for r in range(RUNS):
        static, dynamic = _run(r, pool, time, xr)
        manifest.append(dict(run=r, cycle=0, member=r,
                             static=f"s{r}.nc", output=f"o{r}.nc"))
        attrs = dict(pool_seed=42, pool_size=POOL_SIZE, runs=json.dumps(manifest))
        static.attrs.update(attrs); dynamic.attrs.update(attrs)
        mode = dict(mode="w") if r == 0 else dict(mode="a", append_dim="cell")
        static.chunk({"cell": POOL_SIZE}).to_zarr(tmp / "static.zarr", **mode)
        dynamic.chunk({"cell": POOL_SIZE, "time": -1}).to_zarr(tmp / "dynamic.zarr", **mode)
    return tmp


@pytest.fixture(scope="module")
def cfg(archive_dir):
    from hydra.utils import instantiate
    from omegaconf import OmegaConf

    return instantiate(OmegaConf.create(f"""
experiment_name: t
experiment_run: t
run_dir: {archive_dir}
seq_length: {SEQ_LEN}
data_lazy_load: false
train_temporal_range: ["2017-01-01", "2017-05-20"]
train_downsampler: null
dynamic_downsampler: null
static_inputs: {STATIC_VARS}
dynamic_inputs: {FORCING_VARS}
target_variables: [vwc]
mask_variables: [mask_missing, mask_lake]
scaling_use_cached: false
scaling_static_range: null
preprocessor: null
model: CudaLSTM
hidden_size: 32
dropout: 0.0
lstm_layers: 1
lstm_batch_norm: false
model_head_layer: regression
model_head_activation: linear
model_head_kwargs: {{}}
predict_steps: 0
data_source:
  file:
    static_inputs: {archive_dir}/static.zarr
    dynamic_inputs: {archive_dir}/dynamic.zarr
    target_variables: {archive_dir}/dynamic.zarr
scaler:
  static_inputs:
    lazy: false
    variant:
      - _target_: hython.scaler.MinMax01Scaler
        variable: {STATIC_VARS}
  dynamic_inputs:
    lazy: false
    variant:
      - _target_: hython.scaler.MinMax01Scaler
        variable: {FORCING_VARS}
  target_variables: null
"""))


def _x(sample):
    """Concatenate statics onto the forcing, as `RNNTrainer.epoch_step` does."""
    xd, xs = sample["xd"], sample["xs"]
    return torch.cat([xd, xs.unsqueeze(-2).expand(*xd.shape[:-1], xs.shape[-1])], -1)


@pytest.fixture(scope="module")
def trained(cfg):
    """A small LSTM trained briefly on the two-run archive."""
    from torch.utils.data import DataLoader

    from hython.datasets import WflowSBM_Pool
    from hython.models.cudnnLSTM import CudaLSTM
    from hython.scaler import Scaler

    torch.manual_seed(0)
    ds = WflowSBM_Pool(cfg, Scaler(cfg), True, "train")
    loader = DataLoader(ds, batch_size=256, shuffle=True, generator=torch.Generator().manual_seed(0))

    model = CudaLSTM(cfg)
    opt = torch.optim.Adam(model.parameters(), lr=0.01)
    for _ in range(6):
        for batch in loader:
            opt.zero_grad()
            pred = model(_x(batch))["y_hat"]
            torch.nn.functional.mse_loss(pred, batch["y"]).backward()
            opt.step()
    model.eval()
    return model, ds


def _per_run_prediction(model, ds):
    """Mean predicted vwc for run 0 and run 1, over the same base cells."""
    n_time = ds.time_size - SEQ_LEN
    out = []
    for run in range(RUNS):
        idx = [(run * POOL_SIZE + c) * n_time + t
               for c in range(POOL_SIZE) for t in range(0, n_time, 7)]
        batch = torch.utils.data.default_collate([ds[i] for i in idx])
        with torch.no_grad():
            out.append(model(_x(batch))["y_hat"].mean().item())
    return out


def test_prediction_responds_to_the_parameter(trained):
    """The point of the whole archive: same cell, same weather, different theta,
    different prediction."""
    model, ds = trained
    run0, run1 = _per_run_prediction(model, ds)
    assert not np.isclose(run0, run1, atol=TRUE_GAP / 4), (
        f"surrogate gives the same vwc for KsatVer and KsatVer x {KSAT_FACTOR}: "
        f"{run0:.4f} vs {run1:.4f}"
    )


def test_the_response_has_the_right_sign_and_size(trained):
    """Higher KsatVer drains faster, so run 1 must sit below run 0, by roughly
    the gap built into the data."""
    model, ds = trained
    run0, run1 = _per_run_prediction(model, ds)
    gap = run0 - run1
    assert gap > 0, f"run 1 should be drier, got run0={run0:.4f} run1={run1:.4f}"
    assert gap == pytest.approx(TRUE_GAP, rel=0.5), f"{gap:.4f} vs {TRUE_GAP:.4f}"


def test_gradient_reaches_the_static_inputs(cfg):
    """A direct check, independent of whether training converged: d(pred)/d(xs)
    must be non-zero. If the statics were detached or overwritten, calibration
    would silently optimise nothing."""
    from hython.datasets import WflowSBM_Pool
    from hython.models.cudnnLSTM import CudaLSTM
    from hython.scaler import Scaler

    torch.manual_seed(0)
    ds = WflowSBM_Pool(cfg, Scaler(cfg), True, "train")
    batch = torch.utils.data.default_collate([ds[i] for i in range(8)])
    batch["xs"] = batch["xs"].clone().requires_grad_(True)

    CudaLSTM(cfg)(_x(batch))["y_hat"].sum().backward()

    assert batch["xs"].grad is not None
    assert batch["xs"].grad.abs().max() > 0
