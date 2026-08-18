'''
Geometric characteristics of the excursion set:  numbers of upcrossings,
upcrossing extents, upcrossing minima and centroids.
'''

# Copyright (C) 2026  Todd Pataky


import numpy as np

from ._labeling import bwlabel
from ._upcrossing import Upcrossing


class ClusterMetricCalculator:
    '''
    A class for computing various geometric characteristics of the excursion set
    including number of upcrossings, upcrossing (cluster) extents, etc.

    :Parameters:

        *None*

    :Returns:

        *calc* --- a ClusterMetricCalculator instance

    :Example:

        >>> y = rft1d.random.randn1d(1, 101, 15.0)
        >>> calc = rft1d.geom.ClusterMetricCalculator()
        >>> k = calc.cluster_extents(y, 0.5) #cluster extents when thresholded at 0.5

    '''
    def __repr__(self):
        s    = ''
        s   += 'RFT1D ClusterMetricCalculator:\n'
        s   += '   (no attributes)\n'
        return s

    def cluster_extents(self, y, u, interp=True, wrap=False):
        '''
        Upcrossing extents (units: nodes).

        :Parameters:

            *y* --- a 1D field

            *u* --- threshold height

            *interp* --- interpolate to threshold *u*

            *wrap* --- wrap upcrossings from the end to the start of the field

        :Returns:

            *k* --- list of upcrossing extents, or [0] if no upcrossings

        :Example:

            >>> k = calc.cluster_extents(y, 0.0) #cluster extents when thresholded at 0.0

        .. danger:: Setting *interp* to False is faster, but it will cause disagreements between node-based and element-based sampling. If the upcrossing is large this difference is negligible, but for small upcrossing there may be strange results (e.g. upcrossing with an extent of zero). Recommendation: **always interpolate**.
        '''
        L,n = bwlabel(np.array(y >= u), merge_wrapped=wrap)
        if n==0:
            m = [0]
        else:
            m   = []
            for i in range(n):
                b       = L==(i+1)
                if np.all(b):
                    mm  = y.size - 1
                elif interp:
                    up  = Upcrossing(y, b, interp, wrap)
                    mm  = up.extent(u)
                else:
                    mm  = b.sum() - 1
                m.append(mm)
        return m


    def cluster_extents_locations(self, y, u, interp=True, wrap=False):
        '''
        Compute both:
        -- Upcrossing extents (units: nodes)
        -- Upcrossing locations (units: continuum position units)


        :Parameters:

            *y* --- a 1D field

            *u* --- threshold height

            *interp* --- interpolate to threshold *u*

            *wrap* --- wrap upcrossings from the end to the start of the field

        :Returns:

            *k* --- list of upcrossing extents, or [np.nan] if no upcrossings
            *q* --- list of upcrossing locations, or [np.nan] if no upcrossings

        :Example:

            >>> k,q = calc.cluster_extents_locations_lo(y, 0.0) #cluster extents and locations when thresholded at 0.0

        .. danger:: Setting *interp* to False is faster, but it will cause disagreements between node-based and element-based sampling. If the upcrossing is large this difference is negligible, but for small upcrossing there may be strange results (e.g. upcrossing with an extent of zero). Recommendation: **always interpolate**.
        '''
        L,n = bwlabel(np.array(y >= u), merge_wrapped=wrap)
        if n==0:
            x = [np.nan]
            m = [np.nan]
        else:
            m,x           = [],[]
            for i in range(n):
                b         = L==(i+1)
                if np.all(b):
                    mm    = y.size - 1
                    xx    = 0.5 * y.size
                elif interp:
                    up    = Upcrossing(y, b, interp, wrap)
                    yi    = up.isolate()
                    x0,x1 = up.endpoints(yi, u)
                    mm    = x1 - x0
                    xx    = 0.5 * (x0 + x1)
                    xx   += np.argwhere(b)[0,0]

                else:
                    mm    = b.sum() - 1
                    xx    = np.argwhere(b).mean()
                m.append(mm)
                x.append(xx)
        return m,x


    def cluster_minima(self, y, u, interp=True):
        '''
        Minimum field height inside each upcrossing.

        :Parameters:

            *y* --- a 1D field

            *u* --- threshold height

            *interp* --- interpolate to threshold *u*

        :Returns:

            *zmin* --- list of upcrossing minima; [0] if no upcrossings

        :Example:

            >>> k = calc.cluster_minima(y, 0.0)

        .. warning :: If *interp* is *True*, the minima are all *u*.

        .. danger:: If *u* is zero and *interp* is *True* the user may be unable to distinguish between two cases: (i) no upcrossings and (ii) one upcrossing with a minimum of zero. Most thresholds we're interested in are much higher than zero, so this buggy behavior is not deemed serious. To check the number of upcrossings use the **nMaxima** method.

        '''
        b   = np.array(y > u)
        if np.all(b):
            m = [y.min()]
        elif np.any(b):
            L,n        = bwlabel(b)
            if interp:
                m      = [u]*n
            else:
                m      = [y[L==(i+1)].min()  for i in range(n)]
        else:
            m = [0]
        return m

    def max_cluster_extent(self, y, u, interp=True, wrap=False):
        '''
        Maximum upcrossing extent

        :Parameters:

            *y* --- a 1D field

            *u* --- threshold height

            *interp* --- interpolate to threshold *u*

            *wrap* --- wrap upcrossings from the end to the start of the field

        :Returns:

            *kmax* --- maximum upcrossing extent (unit: nodes)

        :Example:

            >>> k = calc.max_cluster_extent(y, 0.2)

        .. danger:: Setting *interp* to False is faster, but it will cause disagreements between node-based and element-based sampling. If the upcrossing is large this difference is negligible, but for small upcrossing there may be strange results (e.g. upcrossing with an extent of zero). Recommendation: **always interpolate**.
        '''
        return max(  self.cluster_extents(y, u, interp, wrap)  )

    def mean_cluster_extent(self, y, u, interp=True, wrap=False):
        '''
        Mean upcrossing extent

        :Parameters:

            *y* --- a 1D field

            *u* --- threshold height

            *interp* --- interpolate to threshold *u*

            *wrap* --- wrap upcrossings from the end to the start of the field

        :Returns:

            *kmean* --- mean upcrossing extent (unit: nodes)

        :Example:

            >>> k = calc.mean_cluster_extent(y, 0.5)

        .. danger:: Setting *interp* to False is faster, but it will cause disagreements between node-based and element-based sampling. If the upcrossing is large this difference is negligible, but for small upcrossing there may be strange results (e.g. upcrossing with an extent of zero). Recommendation: **always interpolate**.
        '''
        return np.mean(  self.cluster_extents(y, u, interp, wrap)  )

    def nMaxima(self, y, u):
        '''
        Number of maxima. Equivalent to **nUpcrossings**.
        '''
        return self.nUpcrossings(y, u)


    def nSuprathresholdNodes(self, y, u):
        '''
        Number of nodes in the excursion set.

        :Parameters:

            *y* --- a 1D field

            *u* --- threshold height

        :Returns:

            *nNodes* --- number of nodes which survive the threshold *u*

        :Example:

            >>> nNodes = calc.nSuprathresholdNodes(y, 0.5)

        .. warning:: This returns simply the number of nodes, which is not equivelent to extent. To compute the total extent you must subtract the number of upcrossings from this value. Otherwise use **rft1d.geom.nSuprathresholdResels** or **rft1d.geom.total_excursion_set_extent**.

        '''
        return (y > u).sum()

    def nSuprathresholdResels(self, y, u, fwhm=1.0, interp=True):
        '''
        Number of resels in the excursion set.

        :Parameters:

            *y* --- a 1D field

            *u* --- threshold height

            *fwhm* --- actual or estimated FWHM

            *interp* --- interpolate to threshold *u*

        :Returns:

            *nResels* --- number of resels which survive the threshold *u*

        :Example:

            >>> nResels = calc.nSuprathresholdResels(y, 0.5)

        .. warning:: This is a length measure, so is similar to (**nSuprathresholdNodes** minus **nUpcrossings**) divided by the FWHM, with the exception that extents can be interpolated to *u*.

        .. danger:: Setting *interp* to False is faster, but it will cause disagreements between node-based and element-based sampling. If the upcrossing is large this difference is negligible, but for small upcrossing there may be strange results (e.g. upcrossing with an extent of zero). Recommendation: **always interpolate**.
        '''
        return self.total_excursion_set_extent(y, u, interp=interp) / float(fwhm)


    def nUpcrossings(self, y, u):
        '''
        Number of upcrossings.

        :Parameters:

            *y* --- a 1D field

            *u* --- threshold height

        :Returns:

            *c* --- number of upcrossings

        :Example:

            >>> c = calc.nUpcrossings(y, 0.5)

        '''
        b   = np.array(y > u)
        if np.any(b):
            L,n        = bwlabel(b)
        else:
            n = 0
        return n

    def nUpcrossingsByExtent(self, y, u, k, interp=True, wrap=False):
        '''
        Number of upcrossings at threshold extent *k*.

        :Parameters:

            *y* --- a 1D field

            *u* --- threshold height

            *k* --- cluster extent threshold (unit: nodes)

            *interp* --- interpolate to threshold *u*

            *wrap* --- wrap upcrossings from the end to the start of the field

        :Returns:

            *c* --- number of upcrossings whose extents equal or exceed *k*

        :Example:

            >>> c = calc.nUpcrossingsByExtent(y, 3.5, 5.0)

        .. danger:: Setting *interp* to False is faster, but it will cause disagreements between node-based and element-based sampling. If the upcrossing is large this difference is negligible, but for small upcrossing there may be strange results (e.g. upcrossing with an extent of zero). Recommendation: **always interpolate**.
        '''
        kk  = self.cluster_extents(y, u, interp, wrap)
        return (np.array(kk) >= k).sum()



    def total_excursion_set_extent(self, y, u, interp=True):
        '''
        Total extent of the excursion set.

        :Parameters:

            *y* --- a 1D field

            *u* --- threshold height

            *fwhm* --- actual or estimated FWHM

            *interp* --- interpolate to threshold *u*

        :Returns:

            *k* --- total extent of the excursion set *u*

        :Example:

            >>> nResels = calc.total_excursion_set_extent(y, 0.5)

        :Note:

            This is a length measure, so is similar to **nSuprathresholdNodes** minus **nUpcrossings**, with the exception that extents can be interpolated to *u*.

        .. danger:: Setting *interp* to False is faster, but it will cause disagreements between node-based and element-based sampling. If the upcrossing is large this difference is negligible, but for small upcrossing there may be strange results (e.g. upcrossing with an extent of zero). Recommendation: **always interpolate**.
        '''
        return sum( self.cluster_extents(y, u, interp=interp) )









class ClusterMetricCalculatorInitialized:
    def __init__(self, y, u, interp=True, wrap=False):
        self.y      = y
        self.u      = u
        L,n         = bwlabel(np.array(y >= u), merge_wrapped=wrap)
        self.L      = L
        self.n      = n
        self.interp = interp
        self.wrap   = wrap


    def cluster_centroids(self):
        c = []
        if self.n > 0:
            for i in range(self.n):
                i     = self.L==(i+1)
                x     = np.arange(self.y.size)[i]
                z     = self.y[i]
                z0    = np.sign(z[0]) * self.u * np.ones(z.size)
                z     = np.hstack([z,z0])
                c.append( (x.mean(), z.mean()) )
        return c


    def cluster_extents(self):
        if self.n==0:
            m = []
        else:
            m   = []
            for i in range(self.n):
                b       = self.L==(i+1)
                if np.all(b):
                    mm  = self.y.size - 1
                elif self.interp:
                    up  = Upcrossing(self.y, b, self.interp, self.wrap)
                    mm  = up.extent(self.u)
                else:
                    mm  = b.sum() - 1
                m.append(mm)
        return m


    def cluster_minima(self):
        b   = self.y > self.u
        if np.all(b):
            m = [self.y.min()]
        elif np.any(b):
            if self.interp:
                m      = [self.u]*self.n
            else:
                m      = [self.y[self.L==(i+1)].min()  for i in range(self.n)]
        else:
            m = []
        return m

    def get_all(self):
        if self.n > 0:
            extents    = self.cluster_extents()
            minima     = self.cluster_minima()
            centroids  = self.cluster_centroids()
        else:
            extents,minima,centroids = [],[],[]
        return extents, minima, centroids, self.L
