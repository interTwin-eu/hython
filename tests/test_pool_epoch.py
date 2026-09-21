"""Per-epoch resampling wiring for the pool dataset (H3).

The downsampler's own behaviour is covered in `test_downsampler.py`. What is
checked here is the plumbing around it: that `set_epoch` reaches the dataset,
that the sample index is rebuilt from the full axes rather than from the
previous epoch's subset, and that validation stays put.

`WflowSBM_Pool.__init__` opens zarr stores, so these build the object with
`object.__new__` and set only the attributes `build_sample_index` reads. That
keeps the test on index arithmetic, with no data files.
"""

import numpy as np
import pytest
import torch
import xarray as xr

from hython.datasets.wflow_sbm import WflowSBM_Pool
from hython.sampler.downsampler import PoolDownsampler

POOL_SIZE = 50
RUNS = 3
N_TIME = 100
SEQ_LEN = 10


def make_dataset(downsampler, period="train"):
    ds = object.__new__(WflowSBM_Pool)
    ds.downsampler = downsampler
    ds.period = period
    ds.seq_len = SEQ_LEN
    ds.xs = xr.Dataset(coords={"cell": np.arange(POOL_SIZE * RUNS)})
    ds.xd = xr.Dataset(coords={"time": np.arange(N_TIME)})
    # `__init__` captures these before the index build, and before the xarray
    # objects are replaced by tensors.
    ds.cell_size = ds.xs.sizes["cell"]
    ds.time_size = len(ds.xd.time)
    ds.build_sample_index()
    return ds


def pool_downsampler(split="train", resample=True):
    return PoolDownsampler(
        pool_size=POOL_SIZE,
        runs=RUNS,
        rows_target=30,
        seed=10,
        split=split,
        resample_each_epoch=resample,
    )


def test_set_epoch_changes_which_cells_are_used():
    ds = make_dataset(pool_downsampler())
    first = ds.cell_linear_index.copy()
    ds.set_epoch(1)
    assert not np.array_equal(first, ds.cell_linear_index)


def test_rebuild_does_not_shrink_the_selection():
    """The bug this guards: re-running `sampling_idx` on its own output.

    `sampling_idx` returns a subset, so feeding the previous epoch's index back
    in would take a subset of a subset and the training set would decay to
    nothing over a run of 100 epochs.
    """
    ds = make_dataset(pool_downsampler())
    sizes = []
    for epoch in range(20):
        ds.set_epoch(epoch)
        sizes.append(len(ds.cell_linear_index))
    assert len(set(sizes)) == 1, sizes
    assert len(ds.spacetime_index) == sizes[0] * len(ds.time_index)


def test_every_epoch_keeps_run_alignment():
    """Each chosen base cell must still appear in all of its runs."""
    ds = make_dataset(pool_downsampler())
    for epoch in range(5):
        ds.set_epoch(epoch)
        base, counts = np.unique(ds.cell_linear_index % POOL_SIZE, return_counts=True)
        assert set(counts.tolist()) == {RUNS}


def test_validation_does_not_move_between_epochs():
    ds = make_dataset(pool_downsampler(split="valid", resample=False), period="valid")
    first = ds.cell_linear_index.copy()
    for epoch in range(5):
        ds.set_epoch(epoch)
        np.testing.assert_array_equal(first, ds.cell_linear_index)


def test_train_and_valid_stay_disjoint_every_epoch():
    train = make_dataset(pool_downsampler())
    valid = make_dataset(pool_downsampler(split="valid", resample=False), period="valid")
    for epoch in range(5):
        train.set_epoch(epoch)
        valid.set_epoch(epoch)
        overlap = np.intersect1d(
            train.cell_linear_index % POOL_SIZE, valid.cell_linear_index % POOL_SIZE
        )
        assert overlap.size == 0


def test_numpy_index_matches_the_old_tuple_list():
    """H9: the NumPy build must give exactly what `itertools.product` gave -
    same pairs, same cell-major order, same integer dtype."""
    import itertools

    ds = make_dataset(pool_downsampler())
    for epoch in range(3):
        ds.set_epoch(epoch)
        old = np.array(list(itertools.product(
            ds.cell_linear_index.tolist(), ds.time_index.tolist()
        )))
        np.testing.assert_array_equal(ds.spacetime_index, old)
        assert ds.spacetime_index.dtype == old.dtype
        assert ds.spacetime_index.shape == (len(ds.cell_linear_index) * len(ds.time_index), 2)


def test_set_epoch_is_a_noop_without_a_downsampler():
    ds = make_dataset(None)
    before = ds.cell_linear_index.copy()
    ds.set_epoch(3)
    np.testing.assert_array_equal(before, ds.cell_linear_index)


def test_same_epoch_gives_the_same_draw():
    """A resumed or repeated run must see the same cells at the same epoch."""
    a, b = make_dataset(pool_downsampler()), make_dataset(pool_downsampler())
    a.set_epoch(7)
    b.set_epoch(7)
    np.testing.assert_array_equal(a.cell_linear_index, b.cell_linear_index)


def test_set_epoch_works_after_the_arrays_become_tensors():
    """`__init__` replaces `xs`/`xd` with torch tensors at the end, so by the
    time the first epoch fires they have no `.sizes` and no `.time`.

    A real run hit this: `build_sample_index` read `self.xs.sizes["cell"]` and
    raised `'Tensor' object has no attribute 'sizes'` on epoch 1. The other
    tests here missed it because their stubs stay xarray for ever.
    """
    ds = make_dataset(pool_downsampler())
    before = len(ds.cell_linear_index)

    # what the tail of `__init__` does
    ds.xs = torch.zeros(16, POOL_SIZE * RUNS)
    ds.xd = torch.zeros(N_TIME, 3, POOL_SIZE * RUNS)

    ds.set_epoch(1)
    assert len(ds.cell_linear_index) == before


# ==== the trainer side of the wiring


def test_trainer_set_epoch_reaches_the_datasets():
    """`set_epoch` used to forward only to `loader.sampler`, and only when the
    strategy was distributed. On one GPU that branch never runs, so per-epoch
    resampling silently never fired. It must reach the datasets regardless.
    """
    from hython.itwinai.trainer import RNNDistributedTrainer

    class Dataset:
        def __init__(self): self.seen = []
        def set_epoch(self, epoch): self.seen.append(epoch)

    class Loader:
        def __init__(self, ds): self.dataset = ds

    class Strategy:
        is_distributed = False

    trainer = object.__new__(RNNDistributedTrainer)
    trainer.profiler = None
    trainer.strategy = Strategy()
    train_ds, val_ds = Dataset(), Dataset()
    trainer.train_loader, trainer.val_loader = Loader(train_ds), Loader(val_ds)

    for epoch in range(3):
        trainer.set_epoch(epoch)

    assert train_ds.seen == [0, 1, 2]
    assert val_ds.seen == [0, 1, 2]


def test_trainer_set_epoch_tolerates_a_dataset_without_one():
    """Every other dataset class in wflow_sbm.py has no `set_epoch`."""
    from hython.itwinai.trainer import RNNDistributedTrainer

    class Loader:
        dataset = object()

    class Strategy:
        is_distributed = False

    trainer = object.__new__(RNNDistributedTrainer)
    trainer.profiler = None
    trainer.strategy = Strategy()
    trainer.train_loader = trainer.val_loader = Loader()

    trainer.set_epoch(0)   # must not raise
