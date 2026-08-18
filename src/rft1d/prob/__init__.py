'''
Random Field Theory expectations and probabilities.
The core RFT computations are conducted inside **prob.rft**, and the
**RFTCalculator** class serves as a high-level interface to **prob.rft**

The implementation is split across private submodules;  every public name is
re-exported here, so "rft1d.prob.<name>" is unchanged.
'''

# Copyright (C) 2026  Todd Pataky


# NOTE:  the math functions below were incidental module-level names in the
# single-file version of this module;  they are re-exported so that e.g.
# "rft1d.prob.sqrt" keeps working, and can be dropped at the next release.
from math import exp, log, pi, sqrt

from ._calculators import (RFTCalculator, RFTCalculatorResels, _Expected,
                           _Probability)
from ._constants import FOUR_LOG2, SQRT_2, SQRT_4LOG2, TWO_PI, eps
from ._ec_density import (ec_density, ec_density_F, ec_density_T,
                          ec_density_X2, ec_density_Z)
from ._rft import isf, p_bonferroni, poisson_cdf, rft


__all__ = ['ec_density', 'ec_density_F', 'ec_density_T', 'ec_density_X2',
           'ec_density_Z', 'isf', 'p_bonferroni', 'poisson_cdf', 'rft',
           'RFTCalculator', 'RFTCalculatorResels',
           'FOUR_LOG2', 'SQRT_2', 'SQRT_4LOG2', 'TWO_PI', 'eps']


rftcalc  =  RFTCalculator()   #instantiated only for auto-doc generation
expected = _Expected(None)    #instantiated only for auto-doc generation
p        = _Probability(None) #instantiated only for auto-doc generation
