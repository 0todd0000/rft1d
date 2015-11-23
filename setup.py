
from setuptools import setup



long_description = '''
**rft1d** is a Python package for exploring and validating Random Field Theory (RFT)
expectations regarding upcrossings in univariate and multivariate 1D continua.
These expectations can be used to make statistical inferences regarding signals
observed in experimentally measured 1D continua including scalar and vector time series.
'''

setup(
	name             = 'rft1d',
	version          = '0.1.2',
	description      = 'One-Dimensional Random Field Theory',
	author           = 'Todd Pataky',
	author_email     = 'spm1d.mail@gmail.com',
	url              = 'https://github.com/0todd0000/rft1d',
	download_url     = 'https://github.com/0todd0000/rft1d/archive/master.zip',
	packages         = ['rft1d'],
	package_data     = {'rft1d' : ['examples/*.*', 'examples/paper/*.*', 'data/weather/*.*'] },
	long_description = long_description,
	keywords         = ['future', 'statistics', 'time series analysis'],
	classifiers      = [],
) 