'''
Random Field Theory probabilities and expectations.

The core computation is "rft", which is based on "spm_P_RF.m" and "spm_P.m"
from the spm8 and spm12 Matlab packages (http://www.fil.ion.ucl.ac.uk/spm/).
'''

# Copyright (C) 2025  Todd Pataky

# NOTE:  scipy is imported inside the functions that need it (rather than
# at module scope) to keep "import rft1d" fast

from math import pi, sqrt

import numpy as np

from ._constants import eps
from ._ec_density import ec_density


def _as_float(x):
    if isinstance(x, (int,float)):
        x = float(x)
    elif isinstance(x, np.ndarray):
        x = np.asarray(x, dtype=float)
    return x


def p_bonferroni(STAT, z, df, Q, n=1):
    '''
    Bonferroni correction.

    When fields are very rough a Bonferroni correction might be less severe than
    the RFT threshold. This function yields Bonferroni-corrected p values based
    on the number of field nodes *Q*.

    :Parameters:

        *STAT* --- test statistic (one of:  "Z", "T", "F", "X2", "T2")

        *z* --- field height

        *df* --- degrees of freedom [df{interest} df{error}]

        *Q* --- number of field nodes (used for Bonferroni comparison)

        *n* --- number of test statistic fields in conjunction

    :Returns:

        The probability of exceeding the specified height.

    :Example:

        >>> rft1d.prob.p_bonferroni('Z', 3.1, None, 101) #yields 0.098

    '''
    from scipy import stats
    if STAT=='Z':
        # note: "1 - cdf" (rather than "sf") is retained to match SPM
        p     = 1 - stats.norm.cdf(z)
    elif STAT=='T':
        p     = stats.t.sf(z, df[1])
    elif STAT=='F':
        p     = stats.f.sf(z, df[0], df[1])
    elif STAT=='X2':
        p     = stats.chi2.sf(z, df[1])
    elif STAT=='T2':
        a,m   = map(float,df)
        v0,v1 = a, m - a + 1
        zz    = z * ( (m-a+1)/(a*m) )
        p     = stats.f.sf(zz, v0, v1)
    else:
        raise( ValueError('Statistic must be one of: ["Z", "T", "X2", "F", "T2"]') )
    p         = Q * (p**n)
    return min(p, 1)


def _replaceWithBonferroniIfPossible(STAT, P, c, csize, z, df, Q, n=1):
    if (csize is None) or (z is None) or (Q is None) or (n is None):
        return P
    if (c==1) & (csize==0) :
        Pbonf  = p_bonferroni(STAT, z, df, Q, n)
        P      =  min(P, Pbonf)
    return P

def _replaceWith0DpValueIfPossible(STAT, P, c, csize, z, df, Q, n=1):
    if (c>1) or (csize>0):
        return P
    p = p_bonferroni(STAT, z, df, 1, n)
    if p>P:
        return p
    else:
        return P


def poisson_cdf(a, b):
    # return stats.poisson.cdf(a, b)
    # returns zero when b<0 to matches spm8 results
    from scipy import stats
    if b <= 0:
        p   = 0.0
    else:
        p   = stats.poisson.cdf(a, b)
    return p


def rft(c, k, STAT, Z, df, R, n=1, Q=None, expectations_only=False, version='spm12', _0d_check=True):
    '''
    Random Field Theory probabilities and expectations using unified Euler Characteristic (EC) theory.
    This code is based on "spm_P_RF.m" and "spm_P.m" from the spm8 and spm12 Matlab packages
    which are available from: http://www.fil.ion.ucl.ac.uk/spm/

    :Parameters:

        *c* --- number of clusters

        *k* --- cluster extent (resels)

        *STAT* --- test statistic (one of:  "Z", "T", "F", "X2", "T2")

        *Z* --- field height

        *df* --- degrees of freedom [df{interest} df{error}]

        *R* --- resel counts (0D counts, 1D counts) defining search volume

        *n* --- number of test statistic fields in conjunction

        *Q* --- number of field nodes (used for Bonferroni comparison)

        *expectations_only* --- if True only expectations will be returned

        *version* --- "spm8" or "spm12" (see below)

    :Returns:

        *P*  --- corrected P value

        *p*  --- uncorrected P value

        *Ec* --- expected number of upcrossings {c}

        *Ek* --- expected resels per upcrossing {k}

        *EN* --- expected excursion set resels

        NOTE!  If expectations_only==True, then only (Ec,Ek,EN) are returned.

    :Examples:

        >>> P,p,Ec,Ek,EN = rft1d.prob.rft(1, 0, 'T', 2.1, [1,8], [1,10])

    :Notes:

        1. The spm8 and spm12 Matlab functions on which this code is based were
        developed by K.Friston and other members of the Wellcome Trust Centre for
        Neuroimaging. This function makes minor modifications to those procedures
        to take advantage of the simplicity of the 1D case.

        2. Results for the spm8 and spm12 versions can be obtained via the
        keyword "version". When expected ECs approach zero, the spm8 and spm12
        results will diverge slightly, due to a minor modification in spm12:
        In spm8: "EC = EC + eps".
        In spm12: "EC = np.array([max(ec,eps) for ec in EC])"

        3. Setting *c* and *k* in particular manners will yield important
        probabilities. Consider these three cases:

        (a)  rft(1, 0, STAT, Z, R)  --- field maximum
        (b)  rft(1, k, STAT, u, R)  --- cluster-based inference
        (c)  rft(c, k, STAT, u, R)  --- set-based inference

        (a) is the probability that Gaussian fields will produce 1 upcrossing
        with an extent of 0. Thus this pertains to the maximum of height of
        Gaussian fields, and can be used, for example, for critical threshold
        computations.

        (b) is the probability that Gaussian fields, when thresholded at *u*,
        will produce 1 upcrossing with an extent of *k*. This is used for
        cluster-level inference (i.e. p values for individual upcrossings).

        (c) is the probability that Gaussian fields, when thresholded at *u*,
        will produce *c* upcrossings with a minimum extent of *k*. This is
        used for set-level inference (i.e. p values for the entire result).

        .. warning:: Set-based inference (c) is more powerful than cluster-based inference (b), but unlike (b) it has no localizing information; it is a global p value pertaining to the entire excursion set en masse. It will thus always be lower than (b).

        4. If Q==None, then no Bonferroni check is made. If Q!=None, the RFT
        correction will be compared to Bonferroni correction, and the less
        severe correction will be returned. This will only have an effect
        for very rough fields, for example: when then second resel count
        approaches 0.5*Q.

    :References:

        1. Hasofer AM (1978) Upcrossings of random fields. Suppl Adv Appl
           Prob 10:14-21.
        2. Friston KJ et al (1994) Assessing the significance of focal
           activations using their spatial extent. Human Brain Mapping 1:
           210-220.
        3. Worsley KJ et al (1996) A unified statistical approach for
           determining significant signals in images of cerebral
           activation. Human Brain Mapping 4:58-73.
    '''
    c        = _as_float(c)
    k        = _as_float(k)
    Z        = _as_float(Z)
    D        = float(len(R))  #dimensionality
    if R[1]==0:  #infinitely smooth field
        R = R[0], eps  #to make the results numerically stable
    R        = np.asarray(R, dtype=float)
    EC       = ec_density(STAT, Z, df)
    if version=='spm8':
        EC   = EC + eps
    elif version=='spm12':
        EC   = np.array([max(ec,eps) for ec in EC])
    else:
        raise( ValueError('rft1d error:  unknown version "%s" (version must be "spm8" or "spm12")'%str(version)) )
    if n==1:  #take a shortcut (Edit TCP 2014.08.11) -- about 9 times faster than the fast version below
        EM   = R*EC
        EN   = EC[0]*R[-1]
    else:
        from scipy.special import gamma
        # from math import gamma
        ### SLOW CODE -- but useful for D>1  (following spm8)
        # P = np.linalg.matrix_power( np.triu(linalg.toeplitz(EC*G)), n )
        # P = P[0,]
        ### FASTER CODE (Edit TCP 2013.12.02) -- in 1D case this is about 25 times faster than using np.linalg.matrix_power
        # a,b   = EC*G
        # P     = a**n, n*b*a**(n-1)
        G    = sqrt(pi) / (gamma(0.5*np.arange(1,D+1)))
        a,b  = EC*G
        P    = a**n, n*b*a**(n-1)
        EM   = R/G*P
        EN   = P[0]*R[-1]
    ### expected maxima and resels per cluster:
    Ec       = EM.sum()   #previously "Em"
    Ek       = EN/EM[-1]  #previously "En"
    if expectations_only:
        return Ec,Ek,EN
    ### compute probabilities:  first P{n>k}
    D       -= 1
    if (k==0) or (D==0):
        p    = 1.0
    else:
        # from math import gamma
        from scipy.special import gamma
        beta = (gamma(0.5*D+1)/Ek) **(2/D)
        p    = np.exp( -beta*(k**(2/D)) )
    #Poisson clumping heuristic (for multiple clusters)
    if p==0:
        P    = 0
    else:
        P    = 1 - poisson_cdf(c-1, (Ec + eps)*p)
    #Non-implemented cases:
    if version=='spm8':  #non-implemented flags are removed in spm12; rft1d validates all cases (see ./rft1d/examples/val*)
        if STAT in ['T','X2']:
            if (k>0) and (n>1):
                P,p   = None, None
        elif STAT=='F':
            if k>0:
                P,p   = None, None
    P        = _replaceWithBonferroniIfPossible(STAT, P, c, k, Z, df, Q, n)
    if _0d_check:
        P    = _replaceWith0DpValueIfPossible(STAT, P, c, k, Z, df, Q, n)
    return P, p, Ec, Ek, EN


################################
# Crtical threshold computations
################################

def _approx_threshold(STAT, alpha, df, resels, n):
    from scipy import stats
    # if two_tailed:
    #     alpha   = 0.5*alpha
    a   = (alpha/sum(resels))**(1.0/n)
    if STAT=='Z':
        zstar = stats.norm.isf(a)
    elif STAT=='T':
        zstar = stats.t.isf(a, df[1])
    elif STAT=='X2':
        zstar = stats.chi2.isf(a, df[1])
    elif STAT=='F':
        zstar = stats.f.isf(a, df[0], df[1])
    elif STAT=='T2':
        p,m   = map(float,df)
        df_F  = p, m - p + 1
        fstar = stats.f.isf(a, df_F[0], df_F[1])
        zstar = fstar / ( (m-p+1)/(p*m) )
    else:
        raise ValueError('Statistic must be one of: "Z", "T", "X2", "F", "T2"')
    return zstar

def isf(STAT, alpha, df, resels, n, Q=None, version='spm12'):
    '''
    Inverse survival function
    '''
    from scipy.optimize import fmin
    if isinstance(alpha, (int,float)):
        alpha       = [alpha]
    zstar = []
    for aaa in alpha:
        z0    = _approx_threshold(STAT, aaa, df, resels, n)
        fn    = lambda x : (rft(1, 0, STAT, x[0], df, resels, n, Q, False, version)[0] - aaa)**2
        zzz   = fmin(fn, z0, xtol=1e-9, disp=0)[0]
        zstar.append(zzz)
    return np.asarray(zstar)
