#! /usr/bin/env python

"""
Run a simple test instance of the evaluation module.

This script can be used as a basis for a more complete routine to evalute cross
sections.

@author: Ingrid A V Holm and Jeriek VdA
"""

import xsec

# *** Set directory and cache choices ***
xsec.init(data_dir="gprocs")  # run with default settings (no caching)

# *** Load GP models for the specified process(es) ***
processes = [(1000021, 1000021)]
xsec.load_processes(processes)

# *** Evaluate a cross-section with given input parameters ***
xsec.set_parameters(
    {
        "m1000021": 1000.0,
        "m1000001": 500.0,
        "m1000002": 500.0,
        "m1000003": 500.0,
        "m1000004": 500.0,
        "m1000005": 500.0,
        "m1000006": 500.0,
        "m2000001": 500.0,
        "m2000002": 500.0,
        "m2000003": 500.0,
        "m2000004": 500.0,
        "m2000005": 500.0,
        "m2000006": 500.0,
        "thetab": 0.0,
        "thetat": 0.0,
        "mean": 500.0,
        "energy": 13000,
    }
)

xsec.eval_xsection()

# *** Evaluate a cross-section with input from a SLHA file ***
xsec.import_slha("sps1a.slha")
xsec.set_parameter("energy", 13000)
xsec.eval_xsection()

# *** Finalize (clears cache if necessary, inactive otherwise) ***
xsec.finalize()
