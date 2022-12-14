
'''
Random field generation using rft1d.random.Generator1D

Notes:
-- Using Generator1D is faster than rft1d.randn1d for iterative
generation.
-- When FWHM gets large (2FWHM>nNodes), the data should be
padded using the *pad* keyword.
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
generator  = rft1d.random.Generator1D(nResponses, nNodes, FWHM, pad=False)
y          = generator.generate_sample()
y          = generator.generate_sample()
y          = generator.generate_sample()
y          = generator.generate_sample()


#(2) Plot fields:
plt.close('all')
plt.plot(y.T)
plt.plot([0,100], [0,0], 'k:')
plt.xlabel('Field position', size=16)
plt.ylabel('z', size=20)
plt.title('Random (Gaussian) fields', size=20)
plt.show()
