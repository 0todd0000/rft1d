
import os
import numpy as np
import rft1d
import rft1d_tester as rt
import pytest


# load SPM12b test cases
dir0            = os.path.dirname( __file__ )
fpathNPZ        = os.path.join(dir0,  'data-spm12b', 'testcases_p_RF.npz')
with np.load(fpathNPZ) as z:
	params      = z['params']
	expected    = z['expected']
mgr             = rt.P_RF_TestCaseManager( params, expected, atol=1e-6 )


# # example single case (for debugging)
# ind   = 0
# ind   = 1321603
# ind   = 18147
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
		# if case.params.stat=='Z':
		# 	continue
		# print( case )
		case.test()


# about 20 s to run
@pytest.mark.slow
def test_random_100000():
	np.random.seed(0)
	ind  = np.random.permutation( mgr.ncases )[:100000]
	for i in ind:
		case = mgr.get_single_case_by_index( i )
		# if case.params.stat=='Z':
		# 	continue
		# print(case)
		case.test()


# # about 6-7 min to run
# @pytest.mark.full
# def test_all():
# 	for i in range( mgr.ncases ):
# 		case = mgr.get_single_case_by_index( i )
# 		case.test()


