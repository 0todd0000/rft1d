
import os
import pytest
from scipy.io import loadmat
import rft1d



@pytest.mark.slow
def test_myslow():
	import time
	time.sleep(5)


def test_myfast():
	assert 1==1


def test_load_mat():
	dir0            = os.path.dirname( __file__ )
	fname           = os.path.join(dir0,  'data-spm12b', 'testcases_p_RF.mat')
	M               = loadmat(fname)
	params,expected = M['PARAMS'], M['RESULTS']
	assert params.shape[0]   == 2268000
	assert params.shape[1]   == 9
	assert expected.shape[0] == 2268000
	assert expected.shape[1] == 5


