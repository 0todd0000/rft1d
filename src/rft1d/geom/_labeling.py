'''
Connected-component labeling for binary 1D fields.
'''

# Copyright (C) 2025  Todd Pataky


import numpy as np


def bwlabel(b, merge_wrapped=False):
    '''
    Label clusters in a binary field *b*.
    This function yields the same output as **scipy.ndimage.measurements.label**
    but is much faster for 1D fields.
    If *merge_wrapped*

    :Parameters:

        *b* --- a binary field

        *merge_wrapped* --- if True, boundary upcrossings will be merged into a single cluster.

    :Returns:

        *L* --- labeled upcrossings (array of integers)

        *n* --- number of upcrossings


    :Example:

        >>> y = rft1d.random.randn1d(1, 101, 10.0)
        >>> b = y > 2.0
        >>> L,n = rft1d.geom.bwlabel(b)
    '''
    Q      = b.size
    d      = np.diff(b+0)
    i0     = list(1 + np.argwhere(d==1).flatten())
    i1     = list(np.argwhere(d==-1).flatten())
    if b[0]:
        i0 = [0] + i0
    if b[-1]:
        i1.append(Q-1)
    n      = len(i0)
    L      = np.zeros(Q)
    for i,(ii0,ii1) in enumerate(zip(i0,i1)):
        L[ii0:ii1+1] = i+1
    if merge_wrapped and L[0] and L[-1]:
        L[L==L[-1]] = 1
        n -= 1
    return L,n
