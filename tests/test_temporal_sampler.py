"""The temporal samplers the itwinai trainer puts on each loader (H8).

Validation must visit the same start days every epoch, or its loss moves with
the draw and not only with the model - and early stopping, the learning-rate
scheduler and the multicycle loop all read it. Training must still draw new
days every epoch, and both must repeat exactly from a seed without touching
numpy's global state.

The samplers only read a handful of attributes from the dataset, so a stub
stands in for it: no data files.
"""

import itertools
from types import SimpleNamespace

import numpy as np
import pytest

from hython.sampler import (
    DistributedTemporalDynamicDownsampler,
    RandomTemporalDynamicDownsampler,
    SequentialTemporalDynamicDownsampler,
)
from hython.utils import generate_time_idx

N_CELLS = 7
TIME_SIZE = 60
SEQ_LEN = 10
N_START = TIME_SIZE - SEQ_LEN


class Stub(SimpleNamespace):
    def __len__(self):
        return N_CELLS * N_START


def stub():
    return Stub(
        seq_len=SEQ_LEN,
        time_size=TIME_SIZE,
        cell_coords=np.zeros((N_CELLS, 2)),
        cell_linear_index=np.arange(N_CELLS),
    )


def epochs(sampler, n=4):
    return [list(sampler) for _ in range(n)]


def dd(**kw):
    return {"frac_time": 0.3, "seed": 10, **kw}


# == VALIDATION


def test_validation_days_are_the_same_every_epoch():
    runs = epochs(SequentialTemporalDynamicDownsampler(stub(), dd()))
    assert all(r == runs[0] for r in runs)


def test_validation_is_in_order():
    idx = list(SequentialTemporalDynamicDownsampler(stub(), dd()))
    assert idx == sorted(idx)


def test_validation_uses_frac_time_valid_when_set():
    s = SequentialTemporalDynamicDownsampler(stub(), dd(frac_time_valid=0.5))
    assert len(s) == int(N_START * 0.5) * N_CELLS
    assert len(list(s)) == len(s)


@pytest.mark.parametrize("frac", [None, 1.0])
def test_validation_can_take_every_start_day(frac):
    idx = list(SequentialTemporalDynamicDownsampler(stub(), dd(frac_time_valid=frac)))
    assert idx == list(range(N_CELLS * N_START))


def test_validation_falls_back_to_frac_time():
    """Configs without the new key keep their validation size."""
    s = SequentialTemporalDynamicDownsampler(stub(), dd())
    assert len(s) == int(N_START * 0.3) * N_CELLS


def test_validation_ignores_frac_time_valid_for_training():
    s = RandomTemporalDynamicDownsampler(stub(), dd(frac_time_valid=None))
    assert len(s) == int(N_START * 0.3) * N_CELLS


# == TRAINING


def test_training_days_change_between_epochs():
    runs = epochs(RandomTemporalDynamicDownsampler(stub(), dd()))
    assert len({tuple(r) for r in runs}) == len(runs)


def test_training_repeats_from_the_seed():
    a = epochs(RandomTemporalDynamicDownsampler(stub(), dd(seed=3)))
    b = epochs(RandomTemporalDynamicDownsampler(stub(), dd(seed=3)))
    c = epochs(RandomTemporalDynamicDownsampler(stub(), dd(seed=4)))
    assert a == b
    assert a != c


def test_training_draws_without_repeats_by_default():
    s = RandomTemporalDynamicDownsampler(stub(), dd())
    list(s)
    assert len(set(s.time_indices.tolist())) == len(s.time_indices)


def test_replacement_still_works():
    s = RandomTemporalDynamicDownsampler(stub(), dd(), replacement=True)
    idx = list(s)
    assert len(idx) == len(s)
    assert max(idx) < N_CELLS * N_START


# == BOTH


@pytest.mark.parametrize("cls", [RandomTemporalDynamicDownsampler,
                                 SequentialTemporalDynamicDownsampler])
def test_does_not_touch_global_numpy_state(cls):
    before = np.random.get_state()[1].copy()
    list(cls(stub(), dd()))
    np.testing.assert_array_equal(before, np.random.get_state()[1])


@pytest.mark.parametrize("cls", [RandomTemporalDynamicDownsampler,
                                 SequentialTemporalDynamicDownsampler])
def test_every_cell_gets_the_same_days(cls):
    s = cls(stub(), dd())
    idx = np.array(list(s))
    cells, days = idx // N_START, idx % N_START
    np.testing.assert_array_equal(np.unique(cells), np.arange(N_CELLS))
    per_cell = [tuple(sorted(days[cells == c])) for c in range(N_CELLS)]
    assert len(set(per_cell)) == 1
    assert len(idx) == len(s)


def test_index_values_match_generate_time_idx():
    """The layout `spacetime_index` expects: `c * n_start + day`. Validation
    keeps that order; training holds the same indices, shuffled."""
    s = SequentialTemporalDynamicDownsampler(stub(), dd())
    assert list(s) == generate_time_idx(s.time_indices, N_START, SEQ_LEN, N_CELLS)

    s = RandomTemporalDynamicDownsampler(stub(), dd())
    idx = list(s)
    assert sorted(idx) == sorted(generate_time_idx(s.time_indices, N_START, SEQ_LEN, N_CELLS))


def test_training_batches_mix_cells():
    """The loader keeps a sampler's order, so the sampler must shuffle: in
    cell-major order the first batch would be one cell only."""
    s = RandomTemporalDynamicDownsampler(stub(), dd())
    first_batch = np.array(list(s)[:10])
    assert len(np.unique(first_batch // N_START)) > 1
    assert list(s) != sorted(list(s))


def test_index_points_at_the_right_pair_of_the_dataset_index():
    s = SequentialTemporalDynamicDownsampler(stub(), dd())
    pairs = list(itertools.product(range(N_CELLS), range(N_START)))
    for i in list(s):
        cell, day = pairs[i]
        assert day in s.time_indices


# == DISTRIBUTED


def dist(shuffle):
    return DistributedTemporalDynamicDownsampler(
        stub(), dd(), shuffle=shuffle, num_replicas=1, rank=0
    )


def test_distributed_validation_is_fixed():
    runs = epochs(dist(shuffle=False))
    assert all(r == runs[0] for r in runs)


def test_distributed_training_redraws():
    runs = epochs(dist(shuffle=True))
    assert len({tuple(r) for r in runs}) == len(runs)
