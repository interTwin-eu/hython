"""Early stopping in the epoch loop (H7).

Two things have to hold. The run must stop when the validation loss stops
improving, and every worker must stop on the *same* epoch - a worker leaving a
collective on its own hangs the rest. Only the main worker holds the gathered
validation loss, so the decision is made there and shared.

The agreement logic is tested directly here; that the loop actually breaks is
checked by a real pipeline run (see `test_it_stops_a_real_run`, skipped
without the archive and a GPU).
"""

import pytest

from hython.itwinai.trainer import RNNDistributedTrainer


class Strategy:
    def __init__(self, distributed=False, others=()):
        self.is_distributed = distributed
        self._others = list(others)
        self.gathered = None

    def allgather_obj(self, obj):
        self.gathered = obj
        return [obj] + self._others


def trainer(flag=None, strategy=None):
    t = object.__new__(RNNDistributedTrainer)
    # `strategy` is a property whose setter replaces anything that is not a
    # real TorchDistributedStrategy with a detected one, silently. Assigning
    # `t.strategy` here would swap the stub for a non-distributed strategy and
    # the distributed cases below would pass without testing anything.
    t._strategy = strategy or Strategy()
    if flag is not None:
        t._stop_requested = flag
    return t


def test_does_not_stop_before_anything_has_happened():
    """`_stop_requested` is not set until the first epoch finishes."""
    assert trainer()._early_stop_agreed() is False


def test_stops_when_requested():
    assert trainer(flag=True)._early_stop_agreed() is True


def test_does_not_stop_while_still_improving():
    assert trainer(flag=False)._early_stop_agreed() is False


def test_every_worker_agrees_when_the_main_one_stops():
    """The main worker decides; the others must follow, or they hang."""
    s = Strategy(distributed=True, others=[False, False, False])
    assert trainer(flag=True, strategy=s)._early_stop_agreed() is True


def test_a_worker_that_did_not_decide_still_stops():
    """A non-main worker holds False and learns the answer from the gather."""
    s = Strategy(distributed=True, others=[True, False, False])
    assert trainer(flag=False, strategy=s)._early_stop_agreed() is True


def test_no_gather_when_not_distributed():
    """One process must not touch a collective: there is nobody to meet."""
    s = Strategy(distributed=False)
    trainer(flag=True, strategy=s)._early_stop_agreed()
    assert s.gathered is None


def test_nobody_stops_when_nobody_asked():
    s = Strategy(distributed=True, others=[False, False])
    assert trainer(flag=False, strategy=s)._early_stop_agreed() is False
