'''
Record golden values for the characterization test suite.

Run this ONLY to (re)create the reference file:

    python tests/gen_characterization_golden.py

Regenerating overwrites the reference values, so it should be done
deliberately -- never as a way of making ``test_characterization.py`` pass.
Any intended behaviour change must be reviewed in the golden-file diff.
'''

import argparse
import os
import sys

import numpy as np


HERE = os.path.dirname(os.path.abspath(__file__))
SRC  = os.path.abspath(os.path.join(HERE, os.pardir, 'src'))

### the same two entries tests/conftest.py adds for pytest, so that this script
### can be run directly with no environment setup:
sys.path.insert(0, HERE)
if not os.environ.get('RFT1D_ALLOW_FOREIGN_SOURCE'):
    sys.path.insert(0, SRC)

from rft1d_tester import characterization as ch    # noqa: E402


FPATH = os.path.join(HERE, 'data-characterization', 'golden.npz')


def build():
    numeric, strings = {}, {}
    for name in ch.case_names():
        kind, value = ch.evaluate(name)
        if kind == 'num':
            numeric[name] = value
        else:
            strings[name] = value
    return numeric, strings


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('-o', '--output', default=FPATH,
                        help='output NPZ path')
    args = parser.parse_args()

    numeric, strings = build()
    payload = {f'num::{k}': v for k, v in numeric.items()}
    payload.update({f'str::{k}': np.array(v) for k, v in strings.items()})
    payload['__names__'] = np.array(ch.case_names())

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    np.savez_compressed(args.output, **payload)

    print(f'wrote {args.output}')
    print(f'  cases   : {len(ch.case_names())}')
    print(f'  numeric : {len(numeric)}')
    print(f'  strings : {len(strings)}')


if __name__ == '__main__':
    main()
