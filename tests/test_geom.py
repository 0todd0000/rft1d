'''
Tests for rft1d.geom:  labeling, resel counts and cluster metrics.

Expected values are worked out by hand (or taken from the docstring examples)
so that each test states what the function is meant to return.
'''

import numpy as np
import pytest
import rft1d
from rft1d import geom


# ------------------------------------------------------------------ bwlabel
def test_bwlabel_clusters():
    b    = np.array([0, 1, 1, 0, 1, 0, 1, 1], dtype=bool)
    L, n = geom.bwlabel(b)
    assert n == 3
    assert L.tolist() == [0, 1, 1, 0, 2, 0, 3, 3]


def test_bwlabel_merge_wrapped():
    b    = np.array([1, 0, 0, 1], dtype=bool)
    assert geom.bwlabel(b)[1] == 2
    L, n = geom.bwlabel(b, merge_wrapped=True)
    assert n == 1
    assert L.tolist() == [1, 0, 0, 1]


def test_bwlabel_matches_scipy():
    from scipy import ndimage
    rng  = np.random.default_rng(0)
    for _ in range(20):
        b     = rng.random(101) > 0.6
        L, n  = geom.bwlabel(b)
        Ls, ns = ndimage.label(b)
        assert n == ns
        assert np.array_equal(L, Ls)


# ------------------------------------------------------------------ resels
def test_resel_counts_docstring_values():
    b = np.zeros(101)
    assert geom.resel_counts(b, fwhm=10.0) == (1, 10.0)
    assert geom.resel_counts(b, fwhm=10.0, element_based=True) == (1, pytest.approx(10.1))
    b[25:55] = 1
    r = geom.resel_counts(b, fwhm=10.0)
    assert r[0] == 2 and r[1] == pytest.approx(6.9)
    r = geom.resel_counts(b, fwhm=10.0, element_based=True)
    assert r[0] == 2 and r[1] == pytest.approx(7.1)


def test_resel_conversions_docstring_values():
    b = np.zeros(101)
    b[25:55] = 1
    assert geom.resels2fwhm((1, 10.0), 101) == pytest.approx(10.0)
    assert geom.resels2fwhm((2, 6.9), 71) == pytest.approx(10.0)
    assert geom.resels2fwhm_masked((2, 6.9), b) == pytest.approx(10.0)
    assert geom.resels2nelements((1, 10.0), 10.0) == 100
    assert geom.resels2nelements((2, 6.9), 10.0) == 69
    assert geom.resels2nnodes((1, 10.0), 10.0) == 101
    assert geom.resels2nnodes((2, 6.9), 10.0) == 71


@pytest.mark.parametrize('fwhm', [5.0, 12.5, 30.0])
def test_resel_counts_round_trip(fwhm):
    b = np.zeros(101)
    r = geom.resel_counts(b, fwhm=fwhm)
    assert geom.resels2fwhm(r, 101) == pytest.approx(fwhm)
    assert geom.resels2nnodes(r, fwhm) == 101


def test_estimate_fwhm_recovers_smoothness():
    y = rft1d.randn1d(2000, 101, 20.0, rng=0)
    assert geom.estimate_fwhm(y) == pytest.approx(20.0, rel=0.05)


# ------------------------------------------------------------------ cluster metrics
Y = np.array([0, 0, 1, 1, 1, 0, 0, 0, 2, 2, 0], dtype=float)


def test_cluster_extents_without_interpolation():
    # extent = distance between the first and last suprathreshold node
    calc = geom.ClusterMetricCalculator()
    assert calc.cluster_extents(Y, 0.5, interp=False) == [2, 1]


def test_cluster_extents_with_interpolation():
    # linear interpolation of the threshold crossings:
    # cluster 1 crosses 0.5 at 1.5 and 4.5 (extent 3.0);
    # cluster 2 crosses 0.5 at 7.25 and 9.75 (extent 2.5)
    calc = geom.ClusterMetricCalculator()
    assert calc.cluster_extents(Y, 0.5, interp=True) == pytest.approx([3.0, 2.5])


def test_cluster_extents_wrap():
    y    = np.array([1, 1, 0, 0, 0, 1], dtype=float)
    calc = geom.ClusterMetricCalculator()
    assert len(calc.cluster_extents(y, 0.5, interp=False, wrap=False)) == 2
    assert len(calc.cluster_extents(y, 0.5, interp=False, wrap=True)) == 1


def test_cluster_counts():
    calc = geom.ClusterMetricCalculator()
    assert calc.nUpcrossings(Y, 0.5) == 2
    assert calc.nSuprathresholdNodes(Y, 0.5) == 5
