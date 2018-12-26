#! /usr/bin/env python

"""
Example of an xsec evaluation which loops over parameter values.
"""

import xsec

# *** Set directory and cache choices ***
xsec.init(data_dir="gprocs")  # run with default settings (no caching)

# *** Load GP models for the specified process(es) ***
processes = [(1000021, 1000021)]
xsec.load_processes(processes)

# *** Set parameter values ***
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
        "energy": 13000,
    }
)

# *** Evaluate in loop over gluino mass ***
for mgluino in range(50, 3000, 50):

    # *** Set gluino mass ***
    xsec.set_gluino_mass(mgluino)

    # *** Evaluate the cross section, printing only one line per point ***
    # The output format:
    #   PID1 PID2 xsection_central regdown_rel regup_rel scaledown_rel
    #   scaleup_rel pdfdown_rel pdfup_rel alphasdown_rel alphasup_rel
    xsec.eval_xsection(verbose=1)


# *** Finalise the evaluation procedure ***
xsec.finalise()
