# xsec

[![Tag](https://img.shields.io/github/release-pre/jeriek/xstest.svg)]()
[![Pre-release downloads](https://img.shields.io/github/downloads-pre/jeriek/xstest/latest/total.svg)](./LICENSE)
[![Python version](https://img.shields.io/badge/python-2.7-blue.svg)](https://www.python.org/downloads/release/python-2715/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/ambv/black)
[![License](https://img.shields.io/github/license/jeriek/xstest.svg)](./LICENSE)




Requirements:
- python 2.7.x
- numpy v1.14 or later
- scipy v1.0.0 or later
- joblib v0.12.2 or later
- pySLHA v3.2.2 or later
- setuptools v20.7.0 or later

Currently: for `pip` installation, clone the repo and `cd` into it, then run
```
pip install --user .
```
To uninstall: 
```
pip uninstall xsec
```
To install only the requirements:
```
pip install -r requirements.txt
```

The `pip` installation **doesn't automatically include the data** required to run. To download data after `pip` installation, ensure that the pip install directory for scripts, *e.g.* `~/.local/bin`, is in `$PATH`, and execute the following shell command:
```
xsec-download-gprocs [<directory-for-xsec-data>]
```
If no argument is given, the data will be extracted in a new `gprocs` directory created in the current working directory, otherwise the user-specified path is used.

To check whether the xsec installation finds the data correctly, try a test cross-section evaluation:
```
xsec-test [<directory-for-xsec-data>]
```
If no argument is given, the data is assumed to be in a `gprocs` directory in the current working directory, otherwise the user-specified path is used.

NOTE: Right now, the `init` function does not yet look at the current working directory if the `data_dir` keyword is not specified.

An example main programme showing how to use the module can be found in the [`example_xsec_eval.py`](examples/example_xsec_eval.py) file. This shows evaluation both by specifying the model parameters by hand and by importing a SLHA file.
