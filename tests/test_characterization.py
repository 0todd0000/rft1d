'''
Characterization tests.

These tests replay several thousand probes of the public rft1d API and compare
the results against reference values recorded in
``data-characterization/golden.npz``.

Their purpose is to make refactoring safe:  they say nothing about whether the
package is *correct* (``test_p_RF.py`` and ``test_uc_RF.py`` validate that
against SPM12b), only that behaviour has not changed.

Regenerate the reference file with::

    python tests/gen_characterization_golden.py

and review the resulting diff -- a change there is a change to the public
behaviour of the package.
'''

import os

import numpy as np
import pytest

import rft1d
from rft1d_tester import characterization as ch


FPATH = os.path.join(os.path.dirname(__file__), 'data-characterization',
                     'golden.npz')

### The golden values describe the source tree next to this test suite.  If
### "import rft1d" resolves somewhere else -- a stale build/ or egg-info copy,
### an installed wheel, or another checkout earlier on PYTHONPATH -- then every
### case that the refactor deliberately changed will fail, which looks alarming
### and says nothing useful.  Check it once, up front, and say so plainly.
REPO_SRC   = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir,
                                          'src', 'rft1d'))
IMPORTED   = os.path.dirname(os.path.abspath(rft1d.__file__))

with np.load(FPATH, allow_pickle=False) as z:
    GOLDEN = {k: z[k] for k in z.files}

GOLDEN_NAMES = [str(s) for s in GOLDEN['__names__']]

RTOL = 1e-10
ATOL = 1e-12


def test_imported_package_is_the_one_under_test():
    '''
    Guard against testing one copy of rft1d against another copy's golden file.

    Set RFT1D_ALLOW_FOREIGN_SOURCE=1 to test an installed copy on purpose.
    '''
    if os.environ.get('RFT1D_ALLOW_FOREIGN_SOURCE'):
        pytest.skip('RFT1D_ALLOW_FOREIGN_SOURCE is set')
    assert IMPORTED == REPO_SRC, (
        '"import rft1d" resolved to a different copy of the package than the '
        'one these golden values were recorded from, so the cases that this '
        'branch deliberately changed will all fail.\n'
        f'  imported : {IMPORTED}\n'
        f'  expected : {REPO_SRC}\n'
        'Put this repository\'s src/ first on PYTHONPATH (or uninstall the '
        'other copy), then re-run.  Quick check:\n'
        '  python -c "import rft1d.geom; print(rft1d.geom.__file__)"\n'
        'a path ending in "geom/__init__.py" is this branch; one ending in '
        '"geom.py" is the pre-refactor package.')


def test_case_inventory():
    '''Every recorded case still exists, and no case has been silently added.'''
    assert ch.case_names() == GOLDEN_NAMES


@pytest.mark.parametrize('name', GOLDEN_NAMES)
def test_case(name):
    kind, value = ch.evaluate(name)
    numkey, strkey = f'num::{name}', f'str::{name}'
    if kind == 'str':
        assert strkey in GOLDEN, f'{name}: was numeric, now a string/error'
        assert value == str(GOLDEN[strkey])
    else:
        assert numkey in GOLDEN, f'{name}: was a string/error, now numeric'
        expected = GOLDEN[numkey]
        assert value.shape == expected.shape, f'{name}: shape changed'
        assert np.allclose(value, expected, rtol=RTOL, atol=ATOL,
                           equal_nan=True), f'{name}: values changed'
