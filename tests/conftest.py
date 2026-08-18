'''
Make the test suite locate its own source tree.

Without this, "import rft1d" resolves to whatever happens to be importable
first -- an installed wheel, a stale build/ or egg-info copy, or another
checkout earlier on PYTHONPATH -- and the suite silently tests the wrong
package.  (test_characterization.py detects that; this file prevents it.)

Set RFT1D_ALLOW_FOREIGN_SOURCE=1 to skip the path insertion and test an
installed copy on purpose;  the same variable relaxes the corresponding
check in test_characterization.py.
'''

import os
import sys


HERE = os.path.dirname(os.path.abspath(__file__))
SRC  = os.path.abspath(os.path.join(HERE, os.pardir, 'src'))


def _prepend(path):
    '''Put *path* first on sys.path, without duplicating it.'''
    while path in sys.path:
        sys.path.remove(path)
    sys.path.insert(0, path)


if not os.environ.get('RFT1D_ALLOW_FOREIGN_SOURCE'):
    # this repository's src/ wins over anything already installed
    _prepend(SRC)

# the rft1d_tester helper package lives beside this file;  pytest's default
# "prepend" import mode adds this directory anyway, but doing it explicitly
# keeps the suite working under --import-mode=importlib as well
_prepend(HERE)
