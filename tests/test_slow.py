
import pytest
# import rft1d
# print(rft1d)


@pytest.mark.slow
def test_myslow():
	import time
	time.sleep(5)


def test_myfast():
	assert 1==1



