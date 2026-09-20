"""Downsampler behaviour for the multicycle archive (H3), and what H6 must fix.

Pure index arithmetic on synthetic arrays: no data files, no configs, no model.
Runs in milliseconds and is unaffected by the modules that currently fail to
import.

The `RandomDownsampler` tests are marked `xfail(strict=True)`. They state the
behaviour H6 is meant to deliver, so they do not break the suite today, and
they turn into failures the moment H6 lands - which is the prompt to drop the
marker.
"""

import numpy as np
import pytest

from hython.sampler.downsampler import RandomDownsampler

POOL_SIZE = 50
RUNS = 3
SPACE = np.arange(POOL_SIZE * RUNS)
TIME = np.arange(100)


def runs_per_base_cell(rows, pool_size=POOL_SIZE):
    """How many runs each chosen base cell appears in."""
    base = np.asarray(rows) % pool_size
    return {int(c): int((base == c).sum()) for c in np.unique(base)}


# ==== RandomDownsampler -- H6, and why H3 needs a different class


@pytest.mark.xfail(strict=True, reason="H6: __init__ calls the global np.random.seed")
def test_does_not_touch_global_numpy_state():
    np.random.seed(999)
    before = np.random.get_state()[1].copy()
    RandomDownsampler(frac_time=None, frac_space=0.5, seed=10)
    np.testing.assert_array_equal(np.random.get_state()[1], before)


@pytest.mark.xfail(
    strict=True,
    reason="H6: seeding is global, so the draw depends on when __init__ runs "
           "relative to sampling_idx, not only on the seed",
)
def test_draw_does_not_depend_on_interleaving():
    """Same seed must mean same rows, however construction and use interleave.

    Today it does not. `__init__` reseeds the *global* RNG, so building both
    downsamplers before drawing (which is what hydra's `instantiate` does,
    since it builds the whole config tree first) gives a different answer from
    building and drawing one at a time.
    """
    a = RandomDownsampler(None, 0.5, seed=10)
    first, _ = a.sampling_idx([SPACE, TIME])
    b = RandomDownsampler(None, 0.5, seed=10)
    second, _ = b.sampling_idx([SPACE, TIME])

    c = RandomDownsampler(None, 0.5, seed=10)
    d = RandomDownsampler(None, 0.5, seed=10)      # built before either draws
    third, _ = c.sampling_idx([SPACE, TIME])
    fourth, _ = d.sampling_idx([SPACE, TIME])

    # Built one at a time, each __init__ reseeds immediately before its draw,
    # so these agree.
    np.testing.assert_array_equal(first, second)
    # Built together, the second __init__ is a no-op (same seed) and the first
    # draw advances the shared global RNG, so the second draw lands elsewhere.
    np.testing.assert_array_equal(third, fourth)


def test_random_downsampler_breaks_run_alignment():
    """Characterisation, and the reason H3 needs its own downsampler.

    Run against a stacked cell axis, `RandomDownsampler` picks rows
    independently, so a base cell lands in some runs and not others. The
    surrogate then sees that cell with one theta instead of several, which is
    the failure the archive exists to prevent.

    This guards the config: if anyone puts RandomDownsampler back on the
    multicycle path, the new-downsampler tests below stop being satisfied.
    """
    rows, _ = RandomDownsampler(None, 0.5, seed=10).sampling_idx([SPACE, TIME])
    counts = runs_per_base_cell(rows)
    assert set(counts.values()) != {RUNS}, (
        "RandomDownsampler kept every base cell in every run; if this now holds, "
        "the stacked-axis argument in H3 needs revisiting"
    )


def test_train_valid_are_not_disjoint_by_construction():
    """Pre-H3 baseline: nothing keeps train and valid apart, and how badly
    depends on interleaving.

    Both downsamplers take `${sampling_seed}` in the config. Built one at a
    time they draw *identically* - complete contamination. Built together, as
    hydra does, they overlap only at the chance rate. Neither is disjoint,
    which is what H3 requires, and the fact that the answer moves with
    construction order is the H6 defect above.
    """
    space = np.arange(336466)
    mk = lambda: RandomDownsampler(None, 0.01, seed=10)

    tr, va = mk(), mk()                                   # hydra's order
    a, _ = tr.sampling_idx([space, TIME])
    b, _ = va.sampling_idx([space, TIME])
    together = len(set(a) & set(b))

    tr = mk(); c, _ = tr.sampling_idx([space, TIME])      # one at a time
    va = mk(); d, _ = va.sampling_idx([space, TIME])
    apart = len(set(c) & set(d))

    chance = len(a) * len(b) / len(space)
    assert apart == len(c), "built one at a time, train and valid are identical"
    assert 0 < together < 4 * chance, f"built together, overlap {together} ~ chance {chance:.0f}"


# ==== the pool-aware downsampler -- H3, red until the class exists

NEW_DOWNSAMPLER = None
try:  # the class H3 introduces; name it here once it exists
    from hython.sampler.downsampler import PoolDownsampler as NEW_DOWNSAMPLER
except ImportError:
    pass

needs_h3 = pytest.mark.skipif(
    NEW_DOWNSAMPLER is None,
    reason="H3: PoolDownsampler does not exist yet",
)


@needs_h3
def test_every_chosen_base_cell_appears_in_every_run():
    rows, _ = NEW_DOWNSAMPLER(
        pool_size=POOL_SIZE, runs=RUNS, rows_target=30, seed=10
    ).sampling_idx([SPACE, TIME])
    assert set(runs_per_base_cell(rows).values()) == {RUNS}


@needs_h3
def test_row_count_follows_the_target():
    target = 30
    rows, _ = NEW_DOWNSAMPLER(
        pool_size=POOL_SIZE, runs=RUNS, rows_target=target, seed=10
    ).sampling_idx([SPACE, TIME])
    places = target // RUNS
    assert len(rows) == places * RUNS


@needs_h3
def test_train_and_valid_never_overlap():
    kw = dict(pool_size=POOL_SIZE, runs=RUNS, rows_target=30, seed=10)
    tr, _ = NEW_DOWNSAMPLER(split="train", **kw).sampling_idx([SPACE, TIME])
    va, _ = NEW_DOWNSAMPLER(split="valid", **kw).sampling_idx([SPACE, TIME])
    assert set(np.asarray(tr) % POOL_SIZE).isdisjoint(np.asarray(va) % POOL_SIZE)


@needs_h3
def test_shrinking_places_nests_inside_the_previous_choice():
    """A2(b2) shrinks `places` as runs accumulate. No cell may join training
    for the first time late in the run, so each cycle's choice must sit inside
    the previous one.

    This is a property of the *cycle-level* choice, so it is asserted with
    `resample_each_epoch=False`. With resampling on the per-epoch draw is
    deliberately random, and the nesting question stops mattering: over many
    epochs the model covers the whole train pool rather than a fixed prefix.
    """
    kw = dict(pool_size=POOL_SIZE, rows_target=40, seed=10,
              resample_each_epoch=False)
    big, _ = NEW_DOWNSAMPLER(runs=2, **kw).sampling_idx([SPACE, TIME])
    small, _ = NEW_DOWNSAMPLER(runs=4, **kw).sampling_idx([SPACE, TIME])
    assert set(np.asarray(small) % POOL_SIZE) <= set(np.asarray(big) % POOL_SIZE)


@needs_h3
def test_split_is_stable_across_every_other_setting():
    """(ii) The train/valid partition must not move with runs, target, seed or
    epoch - only `split_seed` may change it."""
    base = NEW_DOWNSAMPLER(pool_size=POOL_SIZE, runs=2, rows_target=40, seed=10)
    for kw in (dict(runs=7), dict(rows_target=10), dict(seed=999)):
        other = NEW_DOWNSAMPLER(
            pool_size=POOL_SIZE, **{**dict(runs=2, rows_target=40, seed=10), **kw}
        )
        np.testing.assert_array_equal(base.base_cells(), other.base_cells())
    base.set_epoch(5)
    np.testing.assert_array_equal(
        base.base_cells(),
        NEW_DOWNSAMPLER(pool_size=POOL_SIZE, runs=2, rows_target=40, seed=10).base_cells(),
    )


@needs_h3
def test_same_seed_same_draw():
    kw = dict(pool_size=POOL_SIZE, runs=RUNS, rows_target=30, seed=10)
    a, _ = NEW_DOWNSAMPLER(**kw).sampling_idx([SPACE, TIME])
    b, _ = NEW_DOWNSAMPLER(**kw).sampling_idx([SPACE, TIME])
    np.testing.assert_array_equal(a, b)


@needs_h3
def test_per_epoch_resampling_changes_cells_but_keeps_alignment():
    kw = dict(pool_size=POOL_SIZE, runs=RUNS, rows_target=30, seed=10)
    d = NEW_DOWNSAMPLER(**kw)
    d.set_epoch(0); a, _ = d.sampling_idx([SPACE, TIME])
    d.set_epoch(1); b, _ = d.sampling_idx([SPACE, TIME])
    assert not np.array_equal(a, b), "cells must change between epochs"
    for rows in (a, b):
        assert set(runs_per_base_cell(rows).values()) == {RUNS}


@needs_h3
def test_validation_is_never_resampled():
    d = NEW_DOWNSAMPLER(pool_size=POOL_SIZE, runs=RUNS, rows_target=30,
                        seed=10, split="valid")
    d.set_epoch(0); a, _ = d.sampling_idx([SPACE, TIME])
    d.set_epoch(1); b, _ = d.sampling_idx([SPACE, TIME])
    np.testing.assert_array_equal(a, b)
