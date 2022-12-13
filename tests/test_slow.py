
import pytest



@pytest.mark.slow
def test_myslow():
	import time
	time.sleep(5)


def test_myfast():
	assert 1==1



