#! /usr/bin/env python

"""
Run a simple test instance of the evaluation module.

This script can be used as a basis for a more complete routine to
evaluate cross sections.
"""

import xsec

# *** Set directory and cache choices ***
xsec.init(data_dir="gprocs")  # run with default settings (no caching)

# *** Set center-of-mass energy (in GeV) ***
xsec.set_energy(13000)

# *** Load GP models for the specified process(es) ***
processes = [(1000021, 1000021)]
xsec.load_processes(processes)

# *** Enter parameter values ***
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

# *** Evaluate the cross section with the given input parameters ***
xsec.eval_xsection()

# *** Clear all parameter values ***
xsec.clear_parameters()

# *** Evaluate the cross section with input parameters from a SLHA file ***
xsec.import_slha("sps1a.slha")
xsec.set_energy(13000)
result = xsec.eval_xsection()

# *** Write result back to SLHA file in XSECTION block ***
xsec.write_slha('sps1a.slha', result)

# *** Finalise the evaluation procedure ***
xsec.finalise()
