
__version__ = '0.2.6'  # 2026-08-18

__all__ = ['distributions', 'geom', 'prob', 'random']

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
