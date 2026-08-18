'''
Euler characteristic (EC) densities.

The EC density of a statistic field is the expected EC per unit volume at a
given field height;  the RFT expectations in "_rft.py" are assembled from
these densities and the search volume's resel counts.
'''

# Copyright (C) 2026  Todd Pataky

# NOTE:  scipy is imported inside the functions that need it (rather than
# at module scope) to keep "import rft1d" fast

from math import exp

import numpy as np

from ._constants import FOUR_LOG2, SQRT_4LOG2, TWO_PI


def ec_density_Z(z):
    from scipy import stats
    # note: "1 - cdf" (rather than "sf") is retained to match SPM
    ec0d        = 1 - stats.norm.cdf(z)
    ec1d        = SQRT_4LOG2 / TWO_PI   *  exp(-0.5*(z*z))
    return [ec0d, ec1d]


def ec_density_T(z, df):
    '''
    Reference:  Worsley KJ et al. (1996) Hum Brain Mapp 4:58-73
    Reference:  Worsley KJ et al. (2004) [Eqn.2 and Table 2]
    '''
    from scipy import stats
    from scipy.special import gammaln
    v    = float(df[1])
    a    = FOUR_LOG2
    b    = np.exp((gammaln((v+1)/2) - gammaln(v/2)))
    c    = (1+z**2/v)**((1-v)/2)
    EC   = []
    EC.append(  stats.t.sf(z,v)  )  #dim: 0
    EC.append(  a**0.5 / TWO_PI * c  )   #dim: 1
    return EC

def ec_density_F(z, df):
    from scipy import stats
    from scipy.special import gammaln
    if z<0:
        return [1, np.inf]    #to bypass warnings in critical threshold calculation
    k,v  = map(float, df)
    k    = max(k, 1.0)        #stats.f.cdf will return nan if k is less than 1
    a    = FOUR_LOG2/TWO_PI
    b    = gammaln(v/2) + gammaln(k/2)
    EC   = []
    EC.append(  1 - stats.f.cdf(z, k, v)  )
    EC.append(  a**0.5 * np.exp(gammaln((v+k-1)/2)-b)*2**0.5 *(k*z/v)**(0.5*(k-1))*(1+k*z/v)**(-0.5*(v+k-2))  )
    return EC

def ec_density_X2(z, df):
    from scipy import stats
    from scipy.special import gammaln
    v    = float(df[1])
    a    = FOUR_LOG2 / TWO_PI
    b    = z ** ((v-1)/2)  * np.exp(-z/2 -gammaln(v/2))  /  (2**((v-2)/2))
    EC   = []
    EC.append(  1 - stats.chi2.cdf(z,v)  )
    EC.append(  a**0.5 * b  )
    return EC


def ec_density(STAT, z, df):
    if STAT=='Z':
        return ec_density_Z(z)
    if STAT=='T':
        return ec_density_T(z, df)
    elif STAT=='F':
        return ec_density_F(z, df)
    elif STAT=='X2':
        return ec_density_X2(z, df)
    elif STAT=='T2':
        p,m  = map(float,df)
        df_F = p, m - p + 1
        zz   = z * ( (m-p+1)/(p*m) )
        return ec_density_F(zz, df_F)
    else:
        raise(ValueError('Statistic must be one of: ["Z", "T", "X2", "F", "T2"]'))
