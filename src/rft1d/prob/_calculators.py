'''
Convenience classes offering high-level access to the RFT expectations and
probabilities computed in "_rft.py".
'''

# Copyright (C) 2026  Todd Pataky


import numpy as np

from .. import geom
from ._rft import isf, rft


def _float_if_possible(x):
    if isinstance(x, np.ndarray):
        if x.size==1:
            return float( x.ravel()[0] )
        else:
            return x
    else:
        return x


class _Expected:
    def __init__(self, calc):
        self._calc  = calc
    def _require_fwhm(self, methodname):
        if (self._calc is None) or (self._calc.FWHM is None):
            raise( ValueError('RFT1D Error:  "%s" is a node count, so it requires a FWHM. This calculator was constructed from resel counts only.'%methodname) )
    def nodes_per_upcrossing(self, u):
        '''
        Number of nodes expected for each uprcrossing at threshold *u*.

        :Example:

            >>> calc = rft1d.prob.RFTCalculator('T', (1,8), 101, 15.0)
            >>> calc.expected.nodes_per_upcrossing(2.7)

        .. warning:: This is a node count, so is equivalent to: (FWHM x **resels_per_upcrossing**)  + 1
        '''
        self._require_fwhm('nodes_per_upcrossing')
        x = self._calc.FWHM * self.resels_per_upcrossing(u) + 1
        return _float_if_possible(x)
    def number_of_upcrossings(self, u):
        '''
        Number of upcrossings expected for threshold *u*.

        :Example:

            >>> calc = rft1d.prob.RFTCalculator('T', (1,8), 101, 15.0)
            >>> calc.expected.number_of_upcrossings(2.7)
        '''
        x = self._calc._get_all(u, expectations_only=True)[:,0]
        return _float_if_possible(x)
    def number_of_suprathreshold_nodes(self, u):
        '''
        Number of nodes expected in the entire excursion set at threshold *u*.
        These nodes can come from multiple upcrossings.

        :Example:

            >>> calc = rft1d.prob.RFTCalculator('T', (1,8), 101, 15.0)
            >>> calc.expected.number_of_suprathreshold_nodes(2.8)

        .. warning:: This is a node count, so is equivalent to: (FWHM x **number_of_suprathreshold_resels**)  + **number_of_upcrossings**
        '''
        self._require_fwhm('number_of_suprathreshold_nodes')
        return self._calc.FWHM * self.number_of_suprathreshold_resels(u) + self.number_of_upcrossings(u)
    def number_of_suprathreshold_resels(self, u):
        '''
        Number of resels expected in the entire excursion set at threshold *u*.
        These resels can come from multiple upcrossings.
        One resel contains (1 x FWHM) nodes.
        Thus this is equivalent to:  (FWHM x number_of_suprathreshold_nodes)

        :Example:

            >>> calc = rft1d.prob.RFTCalculator('T', (1,8), 101, 15.0)
            >>> calc.expected.number_of_suprathreshold_resels(2.9)
        '''
        x = self._calc._get_all(u, expectations_only=True)[:,2]
        return _float_if_possible(x)
    def resels_per_upcrossing(self, u):
        '''
        Number of nodes expected for each uprcrossing at threshold *u*.
        One resel contains (1 x FWHM) nodes.
        Thus this is equivalent to:  (FWHM x nodes_per_upcrossing)

        :Example:

            >>> calc = rft1d.prob.RFTCalculator('T', (1,8), 101, 15.0)
            >>> calc.expected.resels_per_upcrossing(3.0)
        '''
        x = self._calc._get_all(u, expectations_only=True)[:,1]
        return _float_if_possible(x)




class _Probability:
    def __init__(self, calc):
        self._calc  = calc
    def cluster(self, k, u):
        '''
        Cluster-level inference.

        Probability that 1D Gaussian fields would produce an upcrossing of extent *k*
        when thresholded at *u*.

        .. warning:: The threshold *u* should generally be chosen objectively. One possibility is to calculate the *alpha*-based critical threshold using the inverse survival function: **RFTCalculator.isf**

        :Parameters:

            *k* -- cluster extent (resels)

            *u* -- threshold

        :Returns:

            Cluster-specific probability value.

        :Examples:

            >>> calc = rft1d.prob.RFTCalculator('T', (1,8), 101, 15.0)
            >>> calc.p.cluster(0.1, 3.0)
        '''
        return rft(1, k, self._calc.STAT, u, self._calc.df, self._calc.resels, self._calc.n, self._calc.Q, False, self._calc.version)[0]

    def set(self, c, k, u):
        '''
        Set-level inference.

        Probability that 1D Gaussian fields would produce at least *c* upcrossings
        with a minimum extent of *k* when thresholded at *u*.
        This probability pertains to the entire excursion set.

        .. warning:: The threshold *u* should generally be chosen objectively. One possibility is to calculate the *alpha*-based critical threshold using the inverse survival function: **RFTCalculator.isf**

        :Parameters:

            *c* -- number of upcrossings

            *k* -- minimum cluster extent (resels)

            *u* -- threshold

        :Returns:

            Set-specific probability value.

        :Examples:

            >>> calc = rft1d.prob.RFTCalculator('T', (1,8), 101, 15.0)
            >>> calc.p.set(2, 0.1, 2.7)
        '''
        return rft(c, k, self._calc.STAT, u, self._calc.df, self._calc.resels, self._calc.n, self._calc.Q, False, self._calc.version)[0]

    def upcrossing(self, u):
        '''
        Survival function (equivalent to **RFTCalculator.sf**)

        Probability that 1D Gaussian fields would produce a 1D statistic field whose maximum exceeds *u*.

        :Parameters:

            *u* -- threshold (int, float, or sequence of int or float)

        :Returns:

            The probability of exceeding the specified heights.

        :Examples:

            >>> calc = rft1d.prob.RFTCalculator('T', (1,8), 101, 15.0)
            >>> calc.sf(3.5)
        '''
        x = self._calc._get_all(u)[:,0]
        return _float_if_possible(x)





class RFTCalculator:
    '''
    A convenience class for high-level access to RFT probabilities.

    :Parameters:

        *STAT* --- test statistic (one of:  "Z", "T", "F", "X2", "T2")

        *df* --- degrees of freedom [df{interest} df{error}]

        *nodes* --- number of field nodes (int)  OR a binary field (boolean array)

        *FWHM* --- field smoothness (float)

        *n* --- number of test statistic fields in conjunction

        *withBonf* --- use a Bonferroni correction if less severe than the RFT correction

        *version* --- "spm8" or "spm12" (see below)

    :Returns:

        An instance of the RFTCalculator class.

    :Attributes:

        *expected* --- access to RFT expectations

        *p* --- access to RFT probabilities

    :Methods:

        *isf* --- inverse survival function

        *sf* --- survival function

    :Examples:

        >>> calc = rft1d.prob.RFTCalculator('T', (1,8), 101, 15.0)
        >>> calc.expected.number_of_upcrossings(1.0) #yields 1.343
        >>> calc.expected.number_of_upcrossings(4.5) #yields 0.0223
    '''

    def __init__(self, STAT='Z', df=None, nodes=101, FWHM=10.0, n=1, withBonf=False, version='spm12'):
        self.FWHM     = None
        self.Q        = None
        self.STAT     = STAT
        self.df       = df
        self.mask     = None
        self.nNodes   = None
        self.n        = n
        self.resels   = None
        self.version  = version
        self.withBonf = None
        self._parse_nodes_argument(nodes)
        self.set_fwhm(FWHM)
        self.set_bonf(withBonf)
        self.expected = _Expected(self)
        self.p        = _Probability(self)

    def __repr__(self):
        s    = ''
        s   += 'RFT1D RFTCalculator object:\n'
        s   += '   STAT     :  %s\n' %self.STAT
        s   += '   df       :  %s\n' %str(self.df)
        s   += '   nNodes   :  %d\n' %self.nNodes
        s   += '   FWHM     :  %.1f\n' %self.FWHM
        s   += '   withBonf :  %s\n' %self.withBonf
        return s

    def _get_all(self, u, expectations_only=False):
        if isinstance(u, (int,float)):
            u       = [u]
        return np.array([rft(1, 0, self.STAT, uu, self.df, self.resels, self.n, self.Q, expectations_only, self.version)   for uu in u])

    def _parse_nodes_argument(self, nodes):
        if isinstance(nodes, int):
            self.nNodes = nodes
        elif np.ma.is_mask(nodes):
            if nodes.ndim!=1:
                raise( ValueError('RFT1D Error:  the "nodes" argument must be a 1D boolean array. Received a %dD array'%nodes.ndim)  )
            self.nNodes = nodes.size
            self.mask   = np.logical_not(nodes)
        else:
            raise( ValueError('RFT1D Error:  the "nodes" argument must be an integer or a 1D boolean array')  )

    def isf(self, alpha):
        '''
        Inverse survival function.
        (see also the survival function: **RFTCalculator.sf**)

        :Parameters:

            *alpha* -- upper tail probability (float;  0 < alpha < 1)

        :Returns:

            Quantile corresponding to upper-tail probability alpha.
            Equivalently: critical threshold at a Type I error rate of alpha.

        :Examples:

            >>> calc = rft1d.prob.RFTCalculator('T', (1,8), 101, 15.0)
            >>> calc.isf(0.05)
        '''
        x = isf(self.STAT, alpha, self.df, self.resels, self.n, self.Q, self.version)
        return _float_if_possible(x)
    def set_bonf(self, wBonf):
        self.withBonf = bool(wBonf)
        self.Q        = float(self.nNodes) if self.withBonf else None
    def set_fwhm(self, w):
        self.FWHM   = float(w)
        if self.mask is None:
            self.resels = 1, (self.nNodes-1)/self.FWHM  #field length is (nNodes - 1)
        else:
            self.resels = geom.resel_counts(self.mask, fwhm=self.FWHM)
    def sf(self, u):
        '''
        Survival function.
        (Equivalent to **RFTCalculator.p.upcrossing**)

        Probability that 1D Gaussian fields with a smoothness *FWHM* would produce a 1D statistic field whose maximum exceeds *u*.

        :Parameters:

            *u* -- threshold (int, float, or sequence of int or float)

        :Returns:

            The probability of exceeding the specified heights.

        :Examples:

            >>> calc = rft1d.prob.RFTCalculator('T', (1,8), 101, 15.0)
            >>> calc.sf(3.5)
        '''
        return _float_if_possible(  self.p.upcrossing(u)  )



class RFTCalculatorResels(RFTCalculator):
    '''
    A convenience class for high-level access to RFT probabilities (based on resel counts).

    :Parameters:

        *STAT* --- test statistic (one of:  "Z", "T", "F", "X2", "T2")

        *df* --- degrees of freedom [df{interest} df{error}]

        *resels* --- resolution element counts

        *n* --- number of test statistic fields in conjunction

        *withBonf* --- use a Bonferroni correction if less severe than the RFT correction

        *nNodes* --- number of field nodes (int)  (must be specified if "withBonf" is True)

        *version* --- "spm8" or "spm12" (see below)

    :Returns:

        An instance of the RFTCalculator class.

    :Attributes:

        *expected* --- access to RFT expectations

        *p* --- access to RFT probabilities

    :Methods:

        *isf* --- inverse survival function

        *sf* --- survival function

    :Examples:

        >>> calc = rft1d.prob.RFTCalculatorResels('T', (1,8), [1, 6.667])
        >>> calc.expected.number_of_upcrossings(1.0) #yields 1.343
        >>> calc.expected.number_of_upcrossings(4.5) #yields 0.0223
    '''

    def __init__(self, STAT='Z', df=None, resels=[1,10], n=1, withBonf=False, nNodes=None, version='spm12'):
        self.FWHM     = None
        self.Q        = None
        self.STAT     = STAT
        self.df       = df
        self.mask     = None
        self.nNodes   = nNodes
        self.n        = n
        self.resels   = tuple(resels)
        self.version  = version
        self.withBonf = None
        self.set_bonf(withBonf)
        self.expected = _Expected(self)
        self.p        = _Probability(self)

    def __repr__(self):
        s    = ''
        s   += 'RFT1D RFTCalculatorResels object:\n'
        s   += '   STAT     :  %s\n' %self.STAT
        s   += '   df       :  %s\n' %str(self.df)
        s   += '   resels   :  (%d, %.3f)\n' %self.resels
        s   += '   nNodes   :  %s\n' %self.nNodes
        s   += '   withBonf :  %s\n' %self.withBonf
        return s

    def set_bonf(self, wBonf):
        self.withBonf = bool(wBonf)
        if self.withBonf and (self.nNodes is None):
            raise( ValueError('Must specify an integer value for "nNodes" when "withBonf" is True.') )
        self.Q        = float(self.nNodes) if self.withBonf else None
