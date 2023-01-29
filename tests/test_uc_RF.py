
import os
import numpy as np
import rft1d
import rft1d_tester as rt
import pytest


# load SPM12b test cases
dir0            = os.path.dirname( __file__ )
fpathNPZ        = os.path.join(dir0,  'data-spm12b', 'testcases_uc_RF.npz')
with np.load(fpathNPZ) as z:
	params      = z['params']   # PARAMETERS:  'stat', 'alpha', 'df1', 'df2', 'r1', 'r2', 'n'
	expected    = z['expected']

# # reduce dataset
# i0  = params[:,0] == 0
# i1  = params[:,4] > 1
# i   = np.logical_not( i0 & i1 )
# params,expected = params[i], expected[i]


mgr             = rt.UC_RF_TestCaseManager( params, expected, atol=1e-7 )


# # example single case (for debugging)
# ind   = 0
# p     = rt.UC_RF_Parameters( params[ind] )
# r0    = rt.UC_RF_Results( expected[ind] )
#
# print( p )
# print( r0 )
#
# case  = rt.UC_RF_TestCase( p, r0 )
# print( case )
# case.test()

# u     = rft1d.prob.isf(p.stat, p.alpha, p.df, p.resels, p.n, Q=None, version='spm12')
# print(u)

# calc  = rft1d.prob.RFTCalculatorResels(STAT=p.stat, df=p.df, resels=p.resels, withBonf=False)
# u     = calc.isf( p.alpha )
# print(u)
#
# mgr   = rt.UC_RF_TestCaseManager( params, expected )
# case  = mgr.get_single_case_by_index( 0 )
# case.test()
# print( case )




# ind    = 17268
# case  = mgr.get_single_case_by_index( ind )
# print( case )



def test_single_case():
	case = mgr.get_single_case_by_index( 0 )
	case.test()


def test_random_100():
	np.random.seed(0)
	ind  = np.random.permutation( mgr.ncases )[:100]
	for i in ind:
		case = mgr.get_single_case_by_index( i )
		case.test()



# # about 6-7 min to run
# @pytest.mark.slow
# def test_all():
# 	for i in range( mgr.ncases ):
# 		case = mgr.get_single_case_by_index( i )
# 		case.test()


