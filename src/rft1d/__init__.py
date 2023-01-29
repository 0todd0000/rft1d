
__version__  = '0.2.1'   # (2023.01.29)

__all__ = ['data', 'distributions', 'geom', 'prob', 'random']

from . import data
from . import distributions
from . import geom
from . import prob
from . import random

randn1d      = random.randn1d
multirandn1d = random.multirandn1d

chi2         = distributions.chi2
f            = distributions.f
norm         = distributions.norm
t            = distributions.t
T2           = distributions.T2
