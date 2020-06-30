#! /usr/bin/env python

"""
Run a simple test instance of the xsec evaluation module.

These scripts can be used as a basis for a more complete routine to
evaluate cross-sections.
"""

from __future__ import print_function
import xsec

# Set directory and cache choices
xsec.init(data_dir="gprocs")  # run with default settings (no caching)

# Set center-of-mass energy (in GeV)
xsec.set_energy(13000)

# Load GP models for the specified process(es)
processes = [(1000021, 1000021)]
xsec.load_processes(processes)

# ----------------------------------------------------------------------
# * EXAMPLE 1: Setting parameter values with convenience functions
print("Running Example 1 ...")

# Set parameter values
xsec.set_all_squark_masses(500)
xsec.set_gluino_mass(1000)

# Evaluate the cross section with the given input parameters
xsec.eval_xsection()

# Finalise the evaluation procedure
xsec.finalise()

# ----------------------------------------------------------------------
# * EXAMPLE 2: Setting parameter values with a dictionary *
print("Running Example 2 ...")

# Clear all parameter values from the previous example
xsec.clear_parameters()

# Enter dictionary with parameter values
xsec.set_parameters(
    {
        "m1000021": 1000,
        "m1000001": 500,
        "m1000002": 500,
        "m1000003": 500,
        "m1000004": 500,
        "m1000005": 500,
        "m1000006": 500,
        "m2000001": 500,
        "m2000002": 500,
        "m2000003": 500,
        "m2000004": 500,
        "m2000005": 500,
        "m2000006": 500,
        "sbotmix11": 0,
        "stopmix11": 0,
        "mean": 500,
    }
)

# Evaluate the cross section with the given input parameters
xsec.eval_xsection()

# Finalise the evaluation procedure
xsec.finalise()

# ----------------------------------------------------------------------
# * EXAMPLE 3: Setting parameter values with a SLHA file *
print("Running Example 3 ...")

# Clear all parameter values from the previous example
xsec.clear_parameters()

# Import input parameters from a SLHA file (e.g. examples/sps1a.slha)
import os.path

slha_path = os.path.join(os.path.dirname(__file__), "sps1a.slha")
xsec.import_slha(slha_path)

# Evaluate the cross section with the given input parameters
result = xsec.eval_xsection()

# Write result back to SLHA file in XSECTION block
xsec.write_slha(slha_path, result)

# Finalise the evaluation procedure
xsec.finalise()
