#! /usr/bin/env python

"""
Run a simple test instance of the evaluation module.

This script can be used as a basis for a more complete routine to evalute cross
sections.

@author: Ingrid A V Holm and Jeriek VdA
"""

import xsec.evaluation as evl


# *** Set processes to load ***
evl.PROCESSES = [(1000021, 1000021)]

# *** Reset directory with trained GP models (default: '') ***
evl.DATA_DIR = ''

# *** Set cache choices ***
evl.init()  # run with default settings (no caching)

# *** Load GP models for the specified process(es) ***
evl.load_processes(evl.PROCESSES)

# *** Evaluate a cross-section with given input parameters ***
evl.set_parameters({
    'm1000021': 1000.,
    'm1000001': 500.,
    'm1000002': 500.,
    'm1000003': 500.,
    'm1000004': 500.,
    'm1000005': 500.,
    'm1000006': 500.,
    'm2000001': 500.,
    'm2000002': 500.,
    'm2000003': 500.,
    'm2000004': 500.,
    'm2000005': 500.,
    'm2000006': 500.,
    'thetab': 0.,
    'thetat': 0.,
    'mean': 500.
})
evl.eval_xsection()

# *** Evaluate a cross-section with input from a SLHA file ***
evl.import_slha('sps1a.slha')
evl.eval_xsection()

# *** Clear cache if necessary (inactive otherwise) ***
evl.clear_cache()