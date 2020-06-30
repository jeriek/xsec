# xsec: the cross-section evaluation code

[![Release](https://img.shields.io/github/v/release/jeriek/xsec)](https://github.com/jeriek/xsec/releases)
<!-- [![Downloads](https://img.shields.io/github/downloads-pre/jeriek/xsec/latest/total.svg)]() -->
[![Python version](https://img.shields.io/pypi/pyversions/xsec.svg)](https://www.python.org/downloads/)
[![PyPI](https://img.shields.io/pypi/v/xsec.svg)](https://pypi.org/project/xsec/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/ambv/black)
[![License](https://img.shields.io/github/license/jeriek/xstest.svg)](./LICENSE)

## Installation

### Requirements
`xsec` is compatible with both Python 2.7 and 3.
The following external Python libraries are required to run `xsec`:
- Setuptools 20.7.0 or later
- NumPy 1.14 or later
- SciPy 1.0.0 or later
- Joblib 0.12.2 or later
- PySLHA 3.2.3 or later

### `pip` installation
`xsec` can be installed from [PyPI](https://pypi.org/project/xsec/) using
```
pip install xsec
```
Optionally with the `--user` flag if your IT department is mean.

Alternatively, you can clone the repo from GitHub and install from there:
```
git clone https://github.com/jeriek/xsec.git
pip install ./xsec
```

It is also possible, though not recommended, to use the code without `pip` installation.
In that case, to install only the requirements, run
```
pip install -r requirements.txt
```

### Downloading required data
The `pip` installation **does NOT automatically include the data** required to run `xsec`. To download data after the `pip` installation, ensure that the `pip` install directory for scripts, *e.g.* `~/.local/bin`, is in `$PATH`, and execute the following shell command:
```
xsec-download-gprocs [-g GP_DIR] [-t PROCESS_TYPE]
```
The first optional argument `GP_DIR` specifies the name of the (preferably new) directory where the data files will be downloaded and extracted.
If this argument is not specified, a new directory `gprocs` is created in the current working directory. The second optional argument `PROCESS_TYPE` allows for selecting which data to download:

- `gg` (gluino pair production, 220 MB)
- `sg` (1st/2nd gen. squark--gluino pair production, 148 MB)
- `ss` (1st/2nd gen. squark pair production, 1.6 GB)
- `sb` (1st/2nd gen. squark--anti-squark pair production, 767 MB)
- `tb` (3rd gen. squark--anti-squark pair production, 210 MB)
- `all` (everything, 3 GB)

The default option is `all`.

(If you have not `pip`-installed `xsec`, the download script can be executed from the `scripts` folder.)

### Testing the installation
To check whether the `xsec` installation works properly, download the gluino pair production data (`gg`) and try a test cross-section evaluation:
```
xsec-test [GP_DIR]
```
If no argument is given, the data is assumed to be in a directory called `gprocs` in the current working directory, otherwise the user-specified path `GP_DIR` is used.

### Removing the package
Uninstalling `xsec` is as simple as running
```
pip uninstall xsec
```

## Code examples
An example main program showing how to use `xsec` can be found in the [`example_xsec_eval.py`](examples/example_xsec_eval.py) file. This shows evaluation both by specifying the model parameters by hand and by importing an SLHA file. An example showing a simple loop over parameters is available in [`example_xsec_loop.py`](examples/example_xsec_loop.py)

## Licence

xsec: the cross-section evaluation code

Copyright (C) 2020  Andy Buckley, Anders Kvellestad, Are Raklev, Pat Scott, Jon Vegard Sparre, Jeriek Van den Abeele, Ingrid A. Vazquez-Holm

This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with this program. If not, see <https://www.gnu.org/licenses/>.
