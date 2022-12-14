
import numpy as np
import matplotlib.pyplot as plt
import rft1d

'''
WARNING!
Calls to rft1d.random.randn1d must set pad=True
when FWHM is greater than 50
'''

#(0) Set parameters:
np.random.seed(0)
nResponses   = 1000
nNodes       = 101
### generate a field mask:
nodes        = np.array([True]*nNodes) #nothing masked out
nodes[10:30] = False  #this region will be masked out
nodes[60:85] = False


#(1) Cycle through smoothing kernels:
FWHM       = np.linspace(1, 50, 21) #actual FWHM
FWHMe      = [] #estimated FWHM
for w in FWHM:
	y      = rft1d.random.randn1d(nResponses, nodes, w, pad=False)  #broken fields
	FWHMe.append(   rft1d.geom.estimate_fwhm(y)   )
	print( 'Actual FWHM: %06.3f, estimated FWHM: %06.3f' %(w, FWHMe[-1])  )


#(2) Plot results:
plt.close('all')
plt.plot(FWHM, FWHM,  'k:', label='Actual')
plt.plot(FWHM, FWHMe, 'go', label='Estimated')
plt.legend(loc='upper left')
plt.xlabel('Actual FWHM', size=16)
plt.ylabel('Estimated FWHM', size=16)
plt.title('FWHM estimation validation (broken fields)', size=20)
plt.show()