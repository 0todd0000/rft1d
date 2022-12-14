

from . import P_RF_Parameters, P_RF_Results
from . import UC_RF_Parameters, UC_RF_Results
from . import P_RF_TestCase
from . import UC_RF_TestCase






class _TestCaseManager(object):
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
		return P_RF_TestCase( p , e, atol=self.atol, id=ind )



class P_RF_TestCaseManager(_TestCaseManager):
	pass



class UC_RF_TestCaseManager(_TestCaseManager):
	def get_single_case_by_index(self, ind):
		p  = UC_RF_Parameters( self.params[ind] )
		e  = UC_RF_Results( self.expected[ind] )
		return UC_RF_TestCase( p , e, atol=self.atol, id=ind )

