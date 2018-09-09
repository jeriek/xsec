"""
Run an instance of the evaluation.py program

@author: Ingrid A V Holm and Jeriek VdA
"""

import evaluation as eval

eval.init() # default: no caching
# eval.init(use_cache=True, cache_dir="$HOME/xsec_cache", flush_cache=True, use_memmap=True)

# eval.xsections = [(1000021, 1000001), (1000021, 1000003)]
# eval.xsections = [(1000001, -1000001), (1000001, 1000002)]
eval.xsections = [(1000021, 1000021)]

print 'Loading processes:', eval.get_type(eval.xsections)
eval.load_processes(eval.xsections)

eval.eval_xsection(1000, 500, 600, 500, 600, 500, 500, 500, 500)

eval.clear_cache() # idle if not using cache