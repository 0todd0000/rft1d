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

from rft1d_tester import characterization as ch


FPATH = os.path.join(os.path.dirname(__file__), 'data-characterization',
                     'golden.npz')

with np.load(FPATH, allow_pickle=False) as z:
    GOLDEN = {k: z[k] for k in z.files}

GOLDEN_NAMES = [str(s) for s in GOLDEN['__names__']]

RTOL = 1e-10
ATOL = 1e-12


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
