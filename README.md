## xsec: the cross section evaluation code

[![Tag](https://img.shields.io/github/release-pre/jeriek/xstest.svg)]()
[![Pre-release downloads](https://img.shields.io/github/downloads-pre/jeriek/xstest/latest/total.svg)](./LICENSE)
[![Python version](https://img.shields.io/badge/python-2.7-blue.svg)](https://www.python.org/downloads/release/python-2715/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/ambv/black)
[![License](https://img.shields.io/github/license/jeriek/xstest.svg)](./LICENSE)


## Installation

xsec is compatible with both Python 2 and 3. The following Python libraries are required to run xsec:
- setuptools v20.7.0 or later
- numpy v1.14 or later
- scipy v1.0.0 or later
- joblib v0.12.2 or later
- pySLHA v3.2.0 or later

xsec can be installed from PyPI using regular pip commands, *e.g.*
```
pip install xsec
```
optionally with the `--user` flag if your IT-department is mean.

Alternatively, you can clone the repo from GitHub, `cd` into it, and then run
```
pip install --user .
```
It is also possible to use the content of the cloned repo as a module without installing.

To uninstall: 
```
pip uninstall xsec
```
To install only the requirements:
```
pip install -r requirements.txt
```

The `pip` installation **doesn't automatically include the data** required to run. To download data after pip installation, ensure that the pip install directory for scripts, *e.g.* `~/.local/bin`, is in `$PATH`, and execute the following shell command:
```
xsec-download-gprocs [<directory-for-xsec-data>]
```
If no argument is given, the data will be extracted in a new `gprocs` directory created in the current working directory, otherwise the user-specified path is used. If you have not pip installed xsec, the same script can be executed from the `scripts` folder.

To check whether the xsec installation finds the data correctly, try a test cross-section evaluation:
```
xsec-test [<directory-for-xsec-data>]
```
If no argument is given, the data is assumed to be in a `gprocs` directory in the current working directory, otherwise the user-specified path is used.

An example main programme showing how to use the module can be found in the [`example_xsec_eval.py`](examples/example_xsec_eval.py) file. This shows evaluation both by specifying the model parameters by hand and by importing an SLHA file. An example showing a simple loop over parameters can be found as [`example_xsec_loop.py`](examples/example_xsec_loop.py)

## Licence

xsec: the cross section evaluation code

Copyright (C) 2020  Andy Buckley, Ingrid Angélica Vazquez Holm, Anders Kvellestad, Are Raklev, Pat Scott, Jon Vegard Sparre, Jeriek Van den Abeele
  
This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with this program. If not, see <https://www.gnu.org/licenses/>.
