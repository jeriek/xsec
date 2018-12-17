#! /usr/bin/env python

"""
Example of an evaluation which loops over parameter values.

This script can be used as a basis for a more complete routine to
evaluate cross sections.
"""

import xsec

# *** Set directory and cache choices ***
xsec.init(data_dir="gprocs")  # run with default settings (no caching)

# *** Load GP models for the specified process(es) ***
processes = [(1000021, 1000021)]
xsec.load_processes(processes)
