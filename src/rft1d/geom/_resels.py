'''
Field smoothness (FWHM) estimation and resolution element (resel) counts.

The resel counts define the field on which the random process occurs;  all
RFT expectations stem directly from them.
'''

# Copyright (C) 2026  Todd Pataky


from math import log

import numpy as np

from ._labeling import bwlabel


eps = np.finfo(float).eps


def estimate_fwhm(R):
    '''
    Estimate field smoothness (FWHM) from a set of random fields or a set of residuals.

    :Parameters:

        *R* --- a set of random fields, or a set of residuals

    :Returns:

        *FWHM* --- the estimated FWHM

    :Example:

        >>> FWHM = 12.5
        >>> y = rft1d.random.randn1d(8, 101, FWHM)
        >>> w = rft1d.geom.estimate_fwhm(y) #should be close to 12.5

    .. note:: The estimated FWHM will differ from the specified FWHM (just like sample means differ from the population mean). This function implements an unbiased estimate of the FWHM, so the average of many FWHM estimates is expected to converge to the specified value.

    '''
    ssq    = (R**2).sum(axis=0)
    # ### gradient estimation (Method 1:  SPM5, SPM8)
    # dx     = np.diff(R, axis=1)
    # v      = (dx**2).sum(axis=0)
    # v      = np.append(v, 0)   #this is likely a mistake but is entered to match the code in SPM5 and SPM8;  including this yields a more conservative estimate of smoothness
    ### gradient estimation (Method 2)
    dy,dx  = np.gradient(R)
    v      = (dx**2).sum(axis=0)
    # normalize:
    v     /= (ssq + eps)
    # ### gradient estimation (Method 3)
    # dx     = np.diff(R, axis=1)
    # v      = (dx**2).sum(axis=0)
    # v     /= (ssq[:-1] + eps)
    # ignore zero-variance nodes:
    i      = np.isnan(v)
    v      = v[np.logical_not(i)]
    # global FWHM estimate:
    reselsPerNode = np.sqrt(v / (4*log(2)))
    FWHM   = 1 / reselsPerNode.mean()
    return FWHM


def resel_counts(R, fwhm=1, element_based=False):
    '''
    Resolution element (resel) counts.

    This function assembles resel counts, either from a set of residuals or
    from a binary mask. If using a binary mask, True represents regions which
    are masked out.

    :Parameters:

        *R* --- a set of random fields, a set of residuals, or a binary field mask

        *fwhm* --- the true or estimated FWHM

        *element_based* --- element-based sampling (default: node-based sampling)

    :Returns:

        *resels* --- (*r0*, *r1*):  0D and 1D resel counts, respectively

    :Note:

        The resel counts define the field on which the random process occurs.
        The first count (*r0*) is the Hadwiger characteristic, which specifies
        the number of unbroken field segments. An unbroken field has *r0* = 1.
        The second count (*r1*) is the field size divided by the FWHM.

    :Important:

        All RFT expectations stem directly from these resel counts.

    :Important cases:

        1. Node-based sampling (default), unbroken field with Q nodes --- the field size is (Q-1) and **resels = [1, (Q-1)/FWHM]**

        2. Element-based sampling, unbroken field, Q elements --- the field size is Q and **resels = [1, Q/FWHM]**

        3. Node-based sampling (default), broken field with S segements and Q nodes --- the field size is (Q-S) and **resels = [S, (Q-S)/FWHM]**

        4. Element-based sampling, broken field with S segements and Q elements --- the field size is Q and **resels = [S, Q/FWHM]**

    :Examples (unbroken field):

        >>> import numpy as np
        >>> b = np.zeros(101) #no masked regions
        >>> resels = rft1d.geom.resel_counts(b, fwhm=10.0) #yields(1,10)
        >>> resels = rft1d.geom.resel_counts(b, fwhm=10.0, element_based=True) #yields(1,10.1)

    :Examples (broken field):

        >>> b = np.zeros(101)
        >>> b[25:55] = 1
        >>> resels = rft1d.geom.resel_counts(b, fwhm=10.0) #yields(2,6.9)
        >>> resels = rft1d.geom.resel_counts(b, fwhm=10.0, element_based=True) #yields(2,7.1)

    '''
    ### Define binary search area (False = masked):
    if R.ndim==2:
        b     = np.any( np.logical_not(np.isnan(R)), axis=0)
    else:
        b     = np.asarray(np.logical_not(R), dtype=bool)
    ### Summarize search area geometry:
    nNodes    = b.sum()
    nClusters = bwlabel(b)[1]
    if element_based:
        resels    = nClusters,  float(nNodes)/fwhm
    else:
        resels    = nClusters,  float(nNodes-nClusters)/fwhm
    return resels



def resels2fwhm(resels, nNodes, element_based=False):
    '''
    Get the FWHM from resel counts based on the number of field nodes.

    :Parameters:

        *resels* --- resel counts

        *nNodes* --- number of field nodes (in broken fields: number of unbroken nodes)

        *element_based* --- element-based sampling (default: node-based sampling)

    :Returns:

        *fwhm* --- field FWHM

    :Example:

        >>> resels = (1, 10.0)
        >>> w = rft1d.geom.resels2fwhm(resels, 101) #yields 10.0
        >>> resels = (2, 6.9)
        >>> w = rft1d.geom.resels2fwhm(resels, 71) #yields 10.0

    :Note:

        See **rft1d.geom.resel_counts** for details regarding the keyword "element_based"** and node-based vs. element-based sampling.
    '''
    if element_based:
        return float(nNodes) / resels[1]
    else:
        return float(nNodes - resels[0]) / resels[1]
def resels2fwhm_masked(resels, mask, element_based=False):
    '''
    Get the FWHM from resel counts based on a binary field mask

    :Parameters:

        *resels* --- resel counts

        *mask* --- binary field mask

        *element_based* --- element-based sampling (default: node-based sampling)

    :Returns:

        *fwhm* --- field FWHM

    :Example:

        >>> resels = (2, 6.9)
        >>> b = np.zeros(101)
        >>> b[25:55] = 1
        >>> w = rft1d.geom.resels2fwhm_masked(resels, b) #yields 10.0

    :Note:

        See **rft1d.geom.resel_counts** for details regarding the keyword "element_based"** and node-based vs. element-based sampling.
    '''
    nNodes = np.logical_not(mask).sum()
    return resels2fwhm(resels, nNodes, element_based)
def resels2nelements(resels, fwhm):
    '''
    Get the field size from resel counts based on the FWHM (element-based sampling).

    :Example:

        >>> resels = (1, 10.0)
        >>> nNodes = rft1d.geom.resels2nelements(resels, 10.0) #yields 100
        >>> resels = (2, 6.9)
        >>> nNodes = rft1d.geom.resels2nelements(resels, 10.0) #yields 69

    :Note:

        See **rft1d.geom.resel_counts** for details regarding node-based vs. element-based sampling.
    '''
    return int(fwhm*resels[1])
def resels2nnodes(resels, fwhm):
    '''
    Get the number of field nodes from resel counts based on the FWHM

    :Parameters:

        *resels* --- resel counts

        *fwhm* --- actual or estimated FWHM*element_based* --- element-based sampling (default: node-based sampling)

    :Returns:

        *nNodes* --- number of field nodes

    :Example:

        >>> resels = (1, 10.0)
        >>> nNodes = rft1d.geom.resels2nnodes(resels, 10.0) #yields 101
        >>> resels = (2, 6.9)
        >>> nNodes = rft1d.geom.resels2nnodes(resels, 10.0) #yields 71

    :Note:

        See **rft1d.geom.resel_counts** for details regarding node-based vs. element-based sampling.
    '''
    return int(resels[0] + fwhm*resels[1])
def resels2fieldsize(resels, fwhm, element_based=False):
    '''
    Get the field size from resel counts based on the FWHM.
    Equivalent to **rft1d.geom.resels2nnodes** minus the number of unbroken
    field segments.

    :Example:

        >>> resels = (1, 10.0)
        >>> nNodes = rft1d.geom.resels2fieldsize(resels, 10.0) #yields 100
        >>> resels = (2, 6.9)
        >>> nNodes = rft1d.geom.resels2fieldsize(resels, 10.0) #yields 69

    :Note:

        See **rft1d.geom.resel_counts** for details regarding the keyword "element_based"** and node-based vs. element-based sampling.
    '''
    if element_based:
        return resels2nelements(resels, fwhm)
    else:
        return resels2nnodes(resels, fwhm) - resels[0]
