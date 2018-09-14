"""
Run an instance of the evaluation.py program

@author: Ingrid A V Holm and Jeriek VdA
"""

import evaluation as eval

# *** Set processes to load ***
# eval.xsections = [(1000021, 1000001), (1000021, 1000003)]
# eval.xsections = [(1000001, -1000001), (1000001, 1000002)]
eval.xsections = [(1000021, 1000021)]

# *** Set directory with trained GP models (checks ./data if not specified here) ***
# eval.DATA_DIR = './data/1000pts/'
eval.DATA_DIR = '../../Nimbus/NIMBUS/gps/newdata_lin/'

# *** Set cache choices ***
eval.init() # run with default settings (no caching)
# eval.init(use_cache=False, cache_dir="$SCRATCH/xsec_cache", flush_cache=True,\
# 	 use_memmap=False)

# *** Load GP models for the specified process(es) ***
print 'Loading process(es):', eval.get_type(eval.xsections)
eval.load_processes(eval.xsections)

# *** Evaluate a cross-section with given input parameters *** 
eval.eval_xsection(1000, 500, 500, 500, 500, 500, 600, 600, 500)

# *** Clear cache if necessary (inactive otherwise) ***
eval.clear_cache() 