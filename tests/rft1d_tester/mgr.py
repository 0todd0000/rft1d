
import numpy as np
import rft1d
from . import P_RF_Parameters, P_RF_Results


class TestCase(object):
	
	def __init__(self, p, r0, atol=1e-7, id=None):
		self.id       = id
		self.params   = p
		self.results0 = r0
		self.results  = self._get_rft1d_results()
		self.atol     = atol
		
	def __repr__(self):
		s  = f'TestCase (ID={self.id})\n'
		s += '----- PARAMS -----\n'
		s += f'{self.params}\n'
		s += '----- EXPECTED -----\n'
		s += f'{self.results0}\n'
		s += '----- ACTUAL -----\n'
		s += f'{self.results}\n'
		s += '----- DIFFERENCE -----\n'
		s += f'{self.results - self.results0}\n'
		return s
	
	def _get_rft1d_results(self):
		p            = self.params
		P,p,Ec,Ek    = rft1d.prob.rft(p.c, p.k, p.stat, p.z, p.df, p.resels, n=p.n, Q=None, version='spm12', _0d_check=False)[:4]
		return P_RF_Results( [P,p,Ec,Ek] )
	
	def get_rft1d_results(self):
		return self.results
	
	def test(self):
		if self.params.stat=='X2' and self.params.z<=2:
			# ignore this case
			pass
		elif self.params.stat=='T' and self.params.z<=1:
			pass
		else:
			r0   = self.results0
			r    = self.get_rft1d_results()
			d    = r - r0
			np.testing.assert_allclose(d, 0, atol=self.atol)
		



class TestCaseManager(object):
	def __init__(self, params, expected, atol=1e-7):
		self.atol     = atol
		self.params   = params
		self.expected = expected
	
	@property
	def ncases(self):
		return self.params.shape[0]
	
	def get_single_case_by_index(self, ind):
		p  = P_RF_Parameters( self.params[ind] )
		e  = P_RF_Results( self.expected[ind] )
		return TestCase( p , e, atol=self.atol, id=ind )



