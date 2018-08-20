"""
Run an instance of the evaluation.py program
with slha-file as input

@author: Ingrid A V Holm
"""

import evaluation as eval

eval.xsections = [(1000021, 1000001), (1000021, 1000003)]
eval.xsections = [(1000001, -1000001), (1000001, 1000002)]
eval.xsections = [(1000021, 1000021)]
a = eval.get_type(eval.xsections)
print 'See here', a

#print a[(1000001, 1000002)] # Getting out the production channel type
eval.load_processes(eval.xsections)
eval.eval_xsection(1000, 500, 600, 500, 600, 500, 500, 500, 500)
