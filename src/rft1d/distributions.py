r'''
SciPy-like interface to 1D RFT distributions.
Currently implemented distributions include:

    * Gaussian
    * Student's t
    * :math:`\chi^2`
    * Fisher-Snedecor F
    * Hotelling's T\ :sup:`2`


All distributions share the following functions:

:Methods:

    **isf** --- RFT inverse survival function (critical height)

    **isf0d** --- Common (0D) inverse survival function (critical height)

    **p_cluster** --- RFT cluster-level inference

    **p_set** --- RFT set-level inference

    **sf** --- RFT survival function

    **sf0d** --- Common (0D) survival function.


:Basic use:

    All distributions can be accessed directly from **rft1d** as follows:

        >>> height = 3.0
        >>> nodes = 101
        >>> FWHM = 10.0
        >>> rft1d.norm.sf(height, nodes, FWHM)
        >>> rft1d.norm.sf0d(height)

:Unbroken and broken fields:

    If the field is unbroken (i.e. continuous) between the start and the
    end of the field, then the "nodes" argument to all methods should be
    an integer representing the number of field nodes:

        >>> height = 3.0
        >>> nodes = 101
        >>> FWHM = 10.0
        >>> rft1d.norm.sf(height, nodes, FWHM)

    However, if the field is broken (i.e. piecewise continuous), then the
    "nodes" argument should be a 1D mask:  a boolean array (size: nodes)
    where False specifies nodes which are masked out:

        >>> height = 3.0
        >>> nodes = np.array([True]*20 + [False]*30 + [True]*51)
        >>> FWHM = 10.0
        >>> rft1d.norm.sf(height, nodes, FWHM)


:Very rough fields and the Bonferroni correction:

    When the fields are very rough (e.g. *FWHM* < 2),
    the Bonferroni correction may be less severe than the RFT correction.
    While both are vaild, it is generally best to use the less-severe
    threshold to maintain statistical power.
    To adopt the less-severe threshold use the keyword argument
    *withBonf* as follows:

        >>> rft1d.norm.sf(3, 101, 1.5, withBonf=False) #yields 0.17932
        >>> rft1d.norm.sf(3, 101, 1.5, withBonf=True)  #yields 0.13634

    By default the *withBonf* argument is False.

        >>> rft1d.norm.sf(3, 101, 1.5)  #yields 0.17932

    For smooth fields the keyword argument will have no effect:

        >>> rft1d.norm.sf(3, 101, 5.0, withBonf=False) #yields 0.05845
        >>> rft1d.norm.sf(3, 101, 5.0, withBonf=True)  #yields 0.05845


:Very smooth fields:

    When Gaussian fields become very smooth, they start to behave like
    Gaussian scalars. Theoretically RFT results are equivalent to scalar
    results when smoothness is infinite. Thus raising the *FWHM* value
    systematically will cause the RFT results to converge to typical 0D results:

        >>> scipy.stats.norm.sf(2)  #yields 0.02275
        >>> rft1d.norm.sf(2, 101, 10.0) #yields 0.31710
        >>> rft1d.norm.sf(2, 101, 100.0) #yields 0.05693
        >>> rft1d.norm.sf(2, 101, 1000.0) #yields 0.02599
        >>> rft1d.norm.sf(2, 101, 10000.0) #yields 0.02284
        >>> rft1d.norm.sf(2, 101, 100000.0) #yields 0.02275

    Setting the smoothness to infinite will return the same result:

        >>> rft1d.norm.sf(2, 101, np.inf) #yields 0.02275

'''

# Copyright (C) 2025  Todd Pataky


# NOTE:  scipy is imported inside the functions that need it (rather than at
# module scope) to keep "import rft1d" fast;  see also prob.py and random.py

import types

import numpy as np

from . prob import RFTCalculator, RFTCalculatorResels


__all__ = ['add_docstrings', 'Gaussian', 'StudentsT', 'Chi2',
           'FisherSnedecorF', 'HotellingsT2',
           'norm', 't', 'chi2', 'f', 'T2']


###############################################################################
# Docstring templates
#
# Each RFT method exists in two forms:  one parameterized by (nodes, FWHM) and
# one parameterized by resel counts.  The two share all of their documentation
# apart from the search-volume parameters, so the docstrings are written once
# here and expanded below.  Distribution-specific substitutions (DISTFLAG and
# the DOFFLAGs) are applied per-class by "add_docstrings".
###############################################################################

_SEARCHVOLFLAG = 'SEARCHVOLFLAG'
_NNODESFLAG    = 'NNODESFLAG'

_SEARCHVOL_NODES = '''*nodes* -- number of field nodes (int)  OR a binary field (boolean array)

            *FWHM* -- field smoothness (float)'''

_SEARCHVOL_RESELS = '''*resels* -- resolution element counts'''

_NNODES = '''
            *nNodes* --- number of field nodes (int)  (must be specified if "withBonf" is True)'''


_DOC_ISF = '''
        RFT inverse survival function.
        (see also the survival function: **rft1d.DISTFLAG.sf**)

        :Parameters:

            *alpha* -- upper tail probability (float;  0 < alpha < 1)

            *df* -- degrees of freedom (int or float)

            SEARCHVOLFLAG

            *withBonf* -- use a Bonferroni correction if less severe than the RFT correction (bool)
NNODESFLAG

        :Returns:

            Quantile corresponding to upper-tail probability alpha.
            Equivalently: critical threshold at a Type I error rate of alpha.

        :Examples:

            >>> rft1d.DISTFLAG.isf(0.05,DOFFLAG 101, 10.0)
        '''

_DOC_P_CLUSTER = '''
        RFT cluster-level inference.

        Probability that 1D Gaussian fields with a smoothness of *FWHM* would produce
        an upcrossing of extent *k* when thresholded at *u*.
        For set-specific probabilities use **rft1d.DISTFLAG.p_set**

        .. warning:: The threshold *u* should generally be chosen objectively. One possibility is to calculate the *alpha*-based critical threshold using the inverse survival function: **rft1d.DISTFLAG.isf**

        :Parameters:

            *k* -- cluster extent (resels)

            *u* -- threshold

            *df* -- degrees of freedom (int or float)

            SEARCHVOLFLAG

            *withBonf* -- use a Bonferroni correction if less severe than the RFT correction (bool)
NNODESFLAG

        :Returns:

            Cluster-specific probability value.

        :Examples:

            >>> rft1d.DISTFLAG.p_cluster(0.5, 3.0,DOFFLAG 101, 15.0)
        '''

_DOC_P_SET = '''
        RFT set-level inference.

        Probability that 1D Gaussian fields with a smoothness of *FWHM* would produce
        at least *c* upcrossings with a minimum extent of *k* when thresholded at *u*.
        This probability pertains to the entire excursion set.
        For cluster-specific probabilities use **rft1d.DISTFLAG.p_cluster**

        .. warning:: The threshold *u* should generally be chosen objectively. One possibility is to calculate the *alpha*-based critical threshold using the inverse survival function: **rft1d.DISTFLAG.isf**

        :Parameters:

            *c* -- number of upcrossings

            *k* -- minimum cluster extent (resels)

            *u* -- threshold

            *df* -- degrees of freedom (int or float)

            SEARCHVOLFLAG

            *withBonf* -- use a Bonferroni correction if less severe than the RFT correction (bool)
NNODESFLAG

        :Returns:

            Set-specific probability value.

        :Examples:

            >>> rft1d.DISTFLAG.p_set(2, 0.5, 3.0,DOFFLAG 101, 15.0)
        '''

_DOC_SF = '''
        RFT survival function.

        Probability that 1D Gaussian fields with a smoothness *FWHM* would produce a 1D statistic field whose maximum exceeds *u*.

        :Parameters:

            *u* -- threshold (int, float, or sequence of int or float)

            *df* -- degrees of freedom (int or float)

            SEARCHVOLFLAG

            *withBonf* -- use a Bonferroni correction if less severe than the RFT correction (bool)
NNODESFLAG

        :Returns:

            The probability of exceeding the specified heights.

        :Examples:

            >>> rft1d.DISTFLAG.sf([1,2,3,4,5],DOFFLAG 101, 10.0)
        '''

_DOC_ISF0D = '''
        Inverse survival function (0D);  equivalent to **scipy.stats.DISTFLAG.isf**

        :Examples:

            >>> rft1d.DISTFLAG.isf0d([0.01, 0.05, 0.1]DOFFLAG2)
            >>> scipy.stats.DISTFLAG.isf([0.01, 0.05, 0.1]DOFFLAG3)
        '''

_DOC_SF0D = '''
        Survival function (0D);  equivalent to **scipy.stats.DISTFLAG.sf**

        :Examples:

            >>> rft1d.DISTFLAG.sf0d([0,1,2]DOFFLAG2)
            >>> scipy.stats.DISTFLAG.sf([0,1,2]DOFFLAG3)
        '''

_DOC_TEMPLATES = {'isf': _DOC_ISF, 'p_cluster': _DOC_P_CLUSTER,
                  'p_set': _DOC_P_SET, 'sf': _DOC_SF}


def _expand(template, resels):
    '''Fill the search-volume placeholders in a docstring template.'''
    if resels:
        s = template.replace(_SEARCHVOLFLAG, _SEARCHVOL_RESELS)
        return s.replace(_NNODESFLAG, _NNODES)
    s = template.replace(_SEARCHVOLFLAG, _SEARCHVOL_NODES)
    return s.replace('\n' + _NNODESFLAG, '')


def add_docstrings(distname, ndf=0):
    '''
    Class decorator which gives a concrete distribution its own copies of the
    inherited docstrings, with the distribution name (DISTFLAG) and the
    degree-of-freedom examples (DOFFLAG, DOFFLAG2, DOFFLAG3) substituted in.
    '''
    def add_docstrings_decorator(cls):
        ### this decorator was adapted from:
        ### http://stackoverflow.com/questions/8100166/inheriting-methods-docstrings-in-python
        for name, func in vars(cls).items():
            if not isinstance(func, types.FunctionType):
                continue    # skip __doc__, class attributes, staticmethods, ...
            if not func.__doc__:
                for parent in cls.__bases__:
                    parfunc            = getattr(parent, name)
                    if parfunc and getattr(parfunc, '__doc__', None):
                        docstr0 = parfunc.__doc__
                        s       = docstr0.replace('DISTFLAG', distname)
                        if ndf==0:
                            s   = s.replace('DOFFLAG2', '')
                            s   = s.replace('DOFFLAG', '')
                            s   = s.replace('\n            *df* -- degrees of freedom (int or float)\n', '')
                        elif ndf==1:
                            s   = s.replace('DOFFLAG3', ', 8')
                            s   = s.replace('DOFFLAG2', ', 8')
                            s   = s.replace('DOFFLAG', ' 8,')
                        elif ndf==2:
                            s   = s.replace('DOFFLAG3', ', 4, 21')
                            s   = s.replace('DOFFLAG2', ', (3,15)')
                            s   = s.replace('DOFFLAG', ' (2,14),')
                            s   = s.replace('degrees of freedom (int or float)', 'degrees of freedom (two-tuple of int or float)')
                        func.__doc__ = s
                        break
        return cls
    return add_docstrings_decorator


###############################################################################
# Base class
###############################################################################

class _RFTDistribution:
    '''
    Base class for the 1D RFT distributions.

    Subclasses supply the SPM statistic code (*STAT*), the number of degrees
    of freedom the user must specify (*ndf*), and the name of the equivalent
    **scipy.stats** distribution used for the 0D methods.
    '''

    _scipyname = None

    def __init__(self, STAT, ndf):
        self._STAT  = STAT
        self._ndf   = int(ndf)

    def _get_df(self, df):
        if self._ndf==1:
            df = 1, df
        return df

    def _dfargs(self, df):
        '''Degrees of freedom as positional arguments for scipy.stats.'''
        if self._ndf==0:
            return ()
        elif self._ndf==1:
            return (df,)
        return tuple(df)

    def _calc(self, df, nodes, FWHM, withBonf):
        return RFTCalculator(STAT=self._STAT, df=self._get_df(df), nodes=nodes, FWHM=FWHM, withBonf=withBonf)

    def _calc_resels(self, df, resels, withBonf, nNodes):
        return RFTCalculatorResels(STAT=self._STAT, df=self._get_df(df), resels=resels, withBonf=withBonf, nNodes=nNodes)

    def _isf0d(self, alpha, df):
        from scipy import stats
        return getattr(stats, self._scipyname).isf(alpha, *self._dfargs(df))

    def _sf0d(self, u, df):
        from scipy import stats
        return getattr(stats, self._scipyname).sf(u, *self._dfargs(df))

    ### RFT methods (field defined by node count and FWHM):

    def isf(self, alpha, df, nodes, FWHM, withBonf=False):
        return self._calc(df, nodes, FWHM, withBonf).isf( alpha )

    def p_cluster(self, k, u, df, nodes, FWHM, withBonf=False):
        return self._calc(df, nodes, FWHM, withBonf).p.cluster(k, u)

    def p_set(self, c, k, u, df, nodes, FWHM, withBonf=False):
        return self._calc(df, nodes, FWHM, withBonf).p.set(c, k, u)

    def sf(self, u, df, nodes, FWHM, withBonf=False):
        return self._calc(df, nodes, FWHM, withBonf).sf( u )

    ### RFT methods (field defined by resel counts):

    def isf_resels(self, alpha, df, resels, withBonf=False, nNodes=None):
        return self._calc_resels(df, resels, withBonf, nNodes).isf( alpha )

    def p_cluster_resels(self, k, u, df, resels, withBonf=False, nNodes=None):
        return self._calc_resels(df, resels, withBonf, nNodes).p.cluster(k, u)

    def p_set_resels(self, c, k, u, df, resels, withBonf=False, nNodes=None):
        return self._calc_resels(df, resels, withBonf, nNodes).p.set(c, k, u)

    def sf_resels(self, u, df, resels, withBonf=False, nNodes=None):
        return self._calc_resels(df, resels, withBonf, nNodes).sf( u )

    ### 0D methods (overridden by every concrete distribution):

    def isf0d(self):
        pass

    def sf0d(self):
        pass


### expand the shared templates onto the base class methods:
for _name, _template in _DOC_TEMPLATES.items():
    getattr(_RFTDistribution, _name).__doc__            = _expand(_template, resels=False)
    getattr(_RFTDistribution, _name + '_resels').__doc__ = _expand(_template, resels=True)
_RFTDistribution.isf0d.__doc__ = _DOC_ISF0D
_RFTDistribution.sf0d.__doc__  = _DOC_SF0D
del _name, _template


###############################################################################
# Concrete distributions
#
# Each subclass exists to (a) fix the statistic and the scipy equivalent, and
# (b) present a distribution-specific call signature and docstring.  The
# Gaussian has no degrees of freedom, so its methods drop the "df" argument;
# the others simply forward to the base class so that "add_docstrings" has a
# per-class function object on which to hang the substituted docstring.
###############################################################################

@add_docstrings('norm', ndf=0)
class Gaussian(_RFTDistribution):
    '''
    Gaussian distributions are accessible via **rft1d.norm** and **rft1d.distributions.norm**
    '''
    _scipyname = 'norm'
    def __init__(self):
        super().__init__('Z', 0)
    def isf(self, alpha, nodes, FWHM, withBonf=False):
        return super().isf(alpha, None, nodes, FWHM, withBonf)
    def isf0d(self, alpha):
        return self._isf0d(alpha, None)
    def p_cluster(self, k, u, nodes, FWHM, withBonf=False):
        return super().p_cluster(k, u, None, nodes, FWHM, withBonf)
    def p_set(self, c, k, u, nodes, FWHM, withBonf=False):
        return super().p_set(c, k, u, None, nodes, FWHM, withBonf)
    def sf(self, u, nodes, FWHM, withBonf=False):
        return super().sf(u, None, nodes, FWHM, withBonf)
    def sf0d(self, heights):
        return self._sf0d(heights, None)


@add_docstrings('t', ndf=1)
class StudentsT(_RFTDistribution):
    _scipyname = 't'
    def __init__(self):
        super().__init__('T', 1)
    def isf(self, alpha, df, nodes, FWHM, withBonf=False):
        return super().isf(alpha, df, nodes, FWHM, withBonf)
    def isf0d(self, alpha, df):
        return self._isf0d(alpha, df)
    def p_cluster(self, k, u, df, nodes, FWHM, withBonf=False):
        return super().p_cluster(k, u, df, nodes, FWHM, withBonf)
    def p_set(self, c, k, u, df, nodes, FWHM, withBonf=False):
        return super().p_set(c, k, u, df, nodes, FWHM, withBonf)
    def sf(self, u, df, nodes, FWHM, withBonf=False):
        return super().sf(u, df, nodes, FWHM, withBonf)
    def sf0d(self, u, df):
        return self._sf0d(u, df)


@add_docstrings('chi2', ndf=1)
class Chi2(_RFTDistribution):
    _scipyname = 'chi2'
    def __init__(self):
        super().__init__('X2', 1)
    def isf(self, alpha, df, nodes, FWHM, withBonf=False):
        return super().isf(alpha, df, nodes, FWHM, withBonf)
    def isf0d(self, alpha, df):
        return self._isf0d(alpha, df)
    def p_cluster(self, k, u, df, nodes, FWHM, withBonf=False):
        return super().p_cluster(k, u, df, nodes, FWHM, withBonf)
    def p_set(self, c, k, u, df, nodes, FWHM, withBonf=False):
        return super().p_set(c, k, u, df, nodes, FWHM, withBonf)
    def sf(self, u, df, nodes, FWHM, withBonf=False):
        return super().sf(u, df, nodes, FWHM, withBonf)
    def sf0d(self, u, df):
        return self._sf0d(u, df)


@add_docstrings('f', ndf=2)
class FisherSnedecorF(_RFTDistribution):
    _scipyname = 'f'
    def __init__(self):
        super().__init__('F', 2)
    def isf(self, alpha, df, nodes, FWHM, withBonf=False):
        return super().isf(alpha, df, nodes, FWHM, withBonf)
    def isf0d(self, alpha, df):
        return self._isf0d(alpha, df)
    def p_cluster(self, k, u, df, nodes, FWHM, withBonf=False):
        return super().p_cluster(k, u, df, nodes, FWHM, withBonf)
    def p_set(self, c, k, u, df, nodes, FWHM, withBonf=False):
        return super().p_set(c, k, u, df, nodes, FWHM, withBonf)
    def sf(self, u, df, nodes, FWHM, withBonf=False):
        return super().sf(u, df, nodes, FWHM, withBonf)
    def sf0d(self, u, df):
        return self._sf0d(u, df)


@add_docstrings('T2', ndf=2)
class HotellingsT2(_RFTDistribution):
    ### Hotelling's T2 is not implemented in scipy.stats, so the 0D methods
    ### are computed via the equivalent F distribution:
    ###     F  =  T2 * (m - p + 1) / (p * m)      with df  =  (p, m - p + 1)
    _scipyname = 'f'
    def __init__(self):
        super().__init__('T2', 2)
    @staticmethod
    def _f_equivalent(df):
        '''F-distribution df and the T2-to-F scale factor.'''
        p,m    = map(float,df)
        return (p, m - p + 1), ( (m-p+1)/(p*m) )
    def isf(self, alpha, df, nodes, FWHM, withBonf=False):
        return super().isf(alpha, df, nodes, FWHM, withBonf)
    def isf0d(self, alpha, df):
        '''
        0D inverse survival function for the Hotelling's T2 distribution
        (not implemented in **scipy.stats**)

        :Examples:

            >>> rft1d.T2.isf0d(0.05, (2,14))
            >>> rft1d.T2.isf0d([0.01, 0.05, 0.10], (2,14))
        '''
        from scipy import stats
        df_F,scale = self._f_equivalent(df)
        fstar      = stats.f.isf(alpha, df_F[0], df_F[1])
        return fstar / scale
    def p_cluster(self, k, u, df, nodes, FWHM, withBonf=False):
        return super().p_cluster(k, u, df, nodes, FWHM, withBonf)
    def p_set(self, c, k, u, df, nodes, FWHM, withBonf=False):
        return super().p_set(c, k, u, df, nodes, FWHM, withBonf)
    def sf(self, u, df, nodes, FWHM, withBonf=False):
        return super().sf(u, df, nodes, FWHM, withBonf)
    def sf0d(self, u, df):
        '''
        0D survival function for the Hotelling's T2 distribution
        (not implemented in **scipy.stats**)

        :Examples:

            >>> rft1d.T2.sf0d([0,1,2], (2,14))
        '''
        from scipy import stats
        df_F,scale = self._f_equivalent(df)
        if isinstance(u, (int,float)):
            uF = u * scale
        else:
            uF = np.asarray(u) * scale
        return stats.f.sf(uF, df_F[0], df_F[1])


norm   = Gaussian()
t      = StudentsT()
chi2   = Chi2()
f      = FisherSnedecorF()
T2     = HotellingsT2()
