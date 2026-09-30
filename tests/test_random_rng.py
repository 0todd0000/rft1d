'''
Tests for the optional "rng" argument of the random field generators.
'''

import numpy as np
import pytest
import rft1d


FNS = [
    lambda rng: rft1d.randn1d(5, 101, 15.0, rng=rng),
    lambda rng: rft1d.randn1d(5, 101, 0, rng=rng),
    lambda rng: rft1d.randn1d(5, 101, np.inf, rng=rng),
    lambda rng: rft1d.randn1d(5, 101, 60.0, pad=True, rng=rng),
    lambda rng: rft1d.multirandn1d(5, 101, 3, 15.0, rng=rng),
    lambda rng: rft1d.multirandn1d(5, 101, 3, np.inf, rng=rng),
]


@pytest.mark.parametrize('fn', FNS)
def test_same_seed_same_fields(fn):
    assert np.array_equal(fn(0), fn(0))
    assert not np.array_equal(fn(0), fn(1))


@pytest.mark.parametrize('fn', FNS)
def test_seed_does_not_touch_global_state(fn):
    np.random.seed(123)
    state = np.random.get_state()[1].copy()
    fn(0)
    assert np.array_equal(np.random.get_state()[1], state)


@pytest.mark.parametrize('fn', FNS)
def test_generator_and_randomstate_accepted(fn):
    a = fn(np.random.default_rng(7))
    b = fn(np.random.default_rng(7))
    assert np.array_equal(a, b)
    c = fn(np.random.RandomState(7))
    d = fn(np.random.RandomState(7))
    assert np.array_equal(c, d)


def test_default_matches_previous_behaviour():
    # rng=None uses numpy's global state exactly as before
    np.random.seed(0)
    a = rft1d.randn1d(5, 101, 15.0)
    np.random.seed(0)
    b = rft1d.randn1d(5, 101, 15.0, rng=np.random.RandomState(0))
    assert np.array_equal(a, b)


def test_generator_class_continues_one_sequence():
    g0 = rft1d.random.Generator1D(4, 101, 10.0, rng=3)
    g1 = rft1d.random.Generator1D(4, 101, 10.0, rng=3)
    s0 = [g0.generate_sample() for _ in range(3)]
    s1 = [g1.generate_sample() for _ in range(3)]
    assert all(np.array_equal(a, b) for a, b in zip(s0, s1))
    assert not np.array_equal(s0[0], s0[1])


@pytest.mark.parametrize('bad', ['0', 1.5, True, [0]])
def test_bad_rng_rejected(bad):
    with pytest.raises(ValueError):
        rft1d.randn1d(2, 11, 3.0, rng=bad)
