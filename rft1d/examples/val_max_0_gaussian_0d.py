
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt



#(0) Set parameters:
np.random.seed(0)
nResponses    = 10000


#(1) Generate Gaussian data:
y             = np.random.randn(nResponses)


#(2) Survival functions:
heights       = np.linspace(0, 5, 21)
sf            = np.array(  [ (y>h).mean()  for h in heights]  )
sfE           = stats.norm.sf(heights)  #theoretical


#(3) Plot results:
plt.close('all')
ax            = plt.axes()
ax.plot(heights, sf, 'o', label='Simulated')
ax.plot(heights, sfE, '-', label='Theoretical')
ax.set_xlabel('$u$', size=20)
ax.set_ylabel('$P (z > u)$', size=20)
ax.legend()
ax.set_title('Gaussian univariate validation (0D)', size=20)
plt.show()
