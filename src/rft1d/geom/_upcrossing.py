'''
Upcrossing extent computation.

An "upcrossing" is a single connected suprathreshold region of a 1D field.
Measuring its extent is complicated by interpolation to the threshold,
by upcrossings which touch the field boundary, and by wrapping.
'''

# Copyright (C) 2025  Todd Pataky


import numpy as np

from ._labeling import bwlabel


class Upcrossing(object):
    '''
    A class for computing upcrossing extents.
    Computing upcrossing extents is simple if the upcrossing extent is to be
    calculated using only the number of suprathreshold nodes (i.e. an integer).
    However, extent computation can be complicated by three factors:
    * interpolation
    * boundary touching
    * wrapping
    Interpolating to a specified height is straightforward, if the
    upcrossing does not touch the boundary.
    Boundary touching and wrapping are easy to deal with, if there is no
    interpolation.
    However, a number of cases needed to be programmed when interpolating and
    also allowing for both boundary touches and wrapping.
    This class accounts for all possibilities via the keywords *interp* and
    *wrap*.
    '''
    def __init__(self, y, b, interp=True, wrap=True):
        self.y             = y
        self.b             = b
        self.ind           = np.argwhere(b).flatten()
        self.touch_start   = b[0]
        self.touch_end     = b[-1]
        self.touch_both    = b[0] and b[-1]
        self.touch_neither = not (b[0] or b[-1])
        self.interp        = interp
        self.wrap          = wrap
    
    def _interp(self, y, i0, h):
        i1    = i0+1
        y0,y1 = y[i0], y[i1]
        x0    = float(i0)
        m     = y1-y0
        x     = (h-y0)/m + x0
        return x

    def endpoints(self, yi, h):
        if yi.size==1:    #no interp
            x0,x1  = 0,0
        elif yi.size==2:  #interp, or no wrapping
            if yi[0]>yi[1]:
                x0  = 0
                if self.interp:
                    x1  = self._interp(yi, 0, h)
                else:
                    x1  = 1
            else:
                x1  = 1
                if self.interp:
                    x0  = self._interp(yi, 0, h)
                else:
                    x0 = 0
        else:
            if yi[0] > h:  #no interpolation or no wrap
                x0  = 0
            else:
                x0  = self._interp(yi, 0, h)
            if yi[-1] > h:
                x1  = yi.size-1
            else:
                x1  = self._interp(yi, yi.size-2, h)
        return x0,x1

    def isolate(self):
        if self.interp:
            if self.touch_neither:
                i0      = self.ind[0]-1
                i1      = self.ind[-1]+2
                yi      = self.y[i0:i1]
            elif self.touch_both:
                if np.all(self.b):
                    yi  = self.y
                else:  #wrapped
                    L,n     = bwlabel(self.b)
                    b0,b1   = L==1, L==2
                    i0end   = np.argwhere(b0).flatten()[-1]
                    i1start = np.argwhere(b1).flatten()[0]
                    y0      = self.y[:i0end+2]
                    y1      = self.y[i1start-1:]
                    yi      = np.hstack([y1,y0])
            elif self.touch_start:
                if self.wrap:
                    i1      = self.ind[-1]+2
                    yi0     = self.y[-1]
                    yi1     = self.y[:i1]
                    yi      = np.hstack([yi0,yi1])
                else:
                    i1      = self.ind[-1]+2
                    yi      = self.y[:i1]
            elif self.touch_end:
                if self.wrap:
                    i0      = self.ind[0]-1
                    yi0     = self.y[i0:]
                    yi1     = self.y[0]
                    yi      = np.hstack([yi0,yi1])
                else:
                    i0      = self.ind[0]-1
                    yi      = self.y[i0:]

        else:  #no interpolation
            if self.touch_neither:
                i0      = self.ind[0]
                i1      = self.ind[-1]+1
                yi      = self.y[i0:i1]
            elif self.touch_both:
                if np.all(self.b):
                    yi  = self.y
                else:  #wrapped
                    L,n     = bwlabel(self.b)
                    b0,b1   = L==1, L==2
                    i0end   = np.argwhere(b0).flatten()[-1]
                    i1start = np.argwhere(b1).flatten()[0]
                    y0      = self.y[:i0end+1]
                    y1      = self.y[i1start:]
                    yi      = np.hstack([y1,y0])
            elif self.touch_start:
                i1      = self.ind[-1]+1
                yi      = self.y[:i1]
            elif self.touch_end:
                i0      = self.ind[0]
                yi      = self.y[i0:]
        return yi

    def extent(self, h, endpoints=False):
        if np.all(self.b):
            m      = self.y.size - 1
        elif self.interp:
            yi     = self.isolate()
            x0,x1  = self.endpoints(yi, h)
            m      = x1 - x0
        else:
            m      = self.b.sum() - 1
        return m
    
    def extent_nodes(self, h):
        return self.extent(h) + 1
