
import numpy as np
import rft1d
import pytest


class _TestCase(object):
	
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
		from . import P_RF_Results
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
			np.testing.assert_allclose(d, 0, rtol=1e-5, atol=self.atol)
			
			

class P_RF_TestCase( _TestCase ):
	pass

class UC_RF_TestCase( _TestCase ):
	def _get_rft1d_results(self):
		from . import UC_RF_Results
		p     = self.params
		# calc  = rft1d.prob.RFTCalculatorResels(STAT=p.stat, df=p.df, resels=p.resels, withBonf=False)
		# u     = calc.isf( p.alpha )
		u     = rft1d.prob.isf(p.stat, p.alpha, p.df, p.resels, p.n, Q=None, version='spm12')
		return UC_RF_Results( float(u) )
	
	def test(self):
		r0   = self.results0
		r    = self.get_rft1d_results()
		d    = r - r0
		assert abs(d) < self.atol



