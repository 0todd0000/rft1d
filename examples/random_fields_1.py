
'''
Random field generation using rft1d.randn1d

Note:
When FWHM gets large (2FWHM>nNodes), the data should be padded
using the *pad* keyword.
'''


import numpy as np
import matplotlib.pyplot as plt
import rft1d


#(0) Set parameters:
np.random.seed(12345)
nResponses = 5
nNodes     = 101
FWHM       = 20.0


#(1) Generate Gaussian 1D fields:
y          = rft1d.randn1d(nResponses, nNodes, FWHM, pad=False)


#(2) Plot:
plt.close('all')
plt.plot(y.T)
plt.plot([0,100], [0,0], 'k:')
plt.xlabel('Field position', size=16)
plt.ylabel('z', size=20)
plt.title('Random (Gaussian) fields', size=20)
plt.show()
