'''
Geometry module

This module contains functions for computing various geomtric characteristics
of 1D fields and upcrossings.

The implementation is split across private submodules;  every public name is
re-exported here, so "rft1d.geom.<name>" is unchanged.
'''

# Copyright (C) 2026  Todd Pataky


# NOTE:  "log" and "eps" were incidental module-level names in the single-file
# version of this module;  they are re-exported so that e.g. "rft1d.geom.eps"
# keeps working, and can be dropped at the next release.
from math import log

from ._clusters import ClusterMetricCalculator, ClusterMetricCalculatorInitialized
from ._labeling import bwlabel
from ._resels import (eps, estimate_fwhm, resel_counts, resels2fieldsize,
                      resels2fwhm, resels2fwhm_masked, resels2nelements,
                      resels2nnodes)
from ._upcrossing import Upcrossing


__all__ = ['bwlabel', 'ClusterMetricCalculator',
           'ClusterMetricCalculatorInitialized', 'estimate_fwhm',
           'resel_counts', 'resels2fieldsize', 'resels2fwhm',
           'resels2fwhm_masked', 'resels2nelements', 'resels2nnodes',
           'Upcrossing']
