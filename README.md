# xsec

Requirements:
- Python 2.7.x
- numpy v1.14 or later
- joblib v0.12.2 or later

[NOTE: `pip` installation is still in the development phase.]

Currently: for `pip` installation, clone the repo and `cd` into it, then run
```
pip install --user .
```
To uninstall: 
```
pip uninstall xsec
```
The `pip` installation doesn't automatically include the data required to run `run_evaluation.py`.

To download data after `pip` installation, ensure `~/.local/bin` is in `$PATH`, and execute the following shell command:
```
download_gprocs.py [<directory-for-xsec-data>]
```
If no argument is given, the data will be extracted in the current working directory, otherwise the user-specified path is used.

To check whether the xsec installation finds the data correctly, check whether the following shell command runs without returning an error:
```
python2 -c "import xsec.evaluation as evl; evl.init(data_dir='<directory-for-xsec-data>')"
```

Right now, the `init` function does not yet look at the current working directory if the `data_dir` keyword is not specified. 
