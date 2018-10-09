"""
Run an instance of the evaluation.py program

@author: Ingrid A V Holm and Jeriek VdA
"""

import evaluation as evl

# *** Set processes to load ***
#evl.XSECTIONS = [(1000021, 1000021)]
evl.XSECTIONS = [(1000006, -1000006)]
# evl.XSECTIONS = [(1000001, -1000001), (1000001, 1000002),
                #  (1000021, 1000001), (1000021, 1000003)]
# evl.XSECTIONS = [(1000001, -1000001), (1000001, 1000002)]

# *** Set directory with trained GP models (checks ./data if not specified here) ***
evl.DATA_DIR = '../nimbus/NIMBUS/gps'
# evl.DATA_DIR = '../../Nimbus/NIMBUS/gps'
# evl.DATA_DIR = './data/1000pts'
# evl.DATA_DIR = './data'

# *** Set cache choices ***
evl.init()  # run with default settings (no caching)
# evl.init(use_cache=False, cache_dir="$SCRATCH/xsec_cache", flush_cache=True,\
# 	 use_memmap=False)

# *** Load GP models for the specified process(es) ***
# print 'Loading process(es):', evl.get_process_type(eval.xsections)
evl.load_processes(evl.XSECTIONS)

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
    'thetat': 0.
})
evl.eval_xsection()

# *** Clear cache if necessary (inactive otherwise) ***
evl.clear_cache()
