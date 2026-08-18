[![Python](https://img.shields.io/badge/Python-FFD43B?style=for-the-badge&logo=python&logoColor=blue)](https://www.python.org)
[![Numpy](https://img.shields.io/badge/Numpy-777BB4?style=for-the-badge&logo=numpy&logoColor=white)](https://numpy.org)
[![Scipy](https://img.shields.io/badge/SciPy-654FF0?style=for-the-badge&logo=SciPy&logoColor=white)](https://scipy.org)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
![version](https://img.shields.io/badge/version-0.2.6-blue)
![downloads](https://static.pepy.tech/badge/rft1d/month)

rft1d
=====

One-Dimensional <b>Random Field Theory</b> in Python.

<b>rft1d</b> is a Python package for exploring and validating Random Field Theory (RFT)
expectations regarding upcrossings in univariate and multivariate 1D continua.
These expectations can be used to make statistical inferences regarding signals
observed in experimentally measured 1D continua including scalar and vector time series.

<b>Please cite:</b>

Pataky TC (2016) RFT1D: Smooth One-Dimensional Random Field Upcrossing Probabilities in Python.
<b>Journal of Statistical Software</b> 71(7): 1-22. https://doi.org/10.18637/jss.v071.i07

Documentation is available at:
[www.spm1d.org/rft1d](http://spm1d.org/rft1d)



___



Installation
------------

```
pip install rft1d
```

**rft1d** requires Python 3.9 or later, along with [numpy](https://numpy.org), [scipy](https://scipy.org) and [matplotlib](https://matplotlib.org).

**rft1d** has been tested on Python 3.9 through 3.13.



___

Running examples:
-----------

The scripts in `./examples` call `import rft1d` so they need the package importable.  Either install **rft1d**, or add the `./src` directory to your PYTHONPATH.

The scripts in `./examples/paper` reproduce the figures and code examples from the <i>Journal of Statistical Software</i> paper.  Run `./examples/paper/all_results.py` to run generate everything in one pass.







___



Development
-----------

The package uses a `src` layout, and the test suite adds `./src` to `sys.path` itself (see `tests/conftest.py`), so no installation step is needed to work on the source:

```
git clone https://github.com/0todd0000/rft1d.git
cd rft1d
pip install pytest numpy scipy matplotlib
```

<b>Running tests:</b>

```
pytest                      # the whole suite
pytest -m "not slow"        # skip the 100,000-case validation (about 20 s)
pytest tests/test_p_RF.py   # a single file
```

There are two test types:

* `test_p_RF.py` and `test_uc_RF.py` check rft1d's probabilities and critical
  thresholds against reference values computed with [SPM12b](https://github.com/spm/spm12), which are stored in
  `tests/data-spm12b`.  These establish that **rft1d**'s probability calculations are <i>correct</i>.
* `test_characterization.py` replays about 1,400 probes of the public API --
  including an inventory of every public name and signature -- against values
  recorded in `tests/data-characterization/golden.npz`.  These establish that
  behaviour has not <i>changed</i>.

If you deliberately change public behavior, re-record the golden values and
review the resulting diff:

```
python tests/gen_characterization_golden.py
```



___



Issues
------

Please report software bugs or other problems by searching existing issues or creating a new issue [here](https://github.com/0todd0000/rft1d/issues).
