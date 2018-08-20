"""
Run an instance of the evaluation.py program
with slha-file as input

@author: Ingrid A V Holm
"""
import evaluation as eam
#import evaluation_as_module as eam
#import evaluation as eam

eam.xsections = [(1000021, 1000001), (1000021, 1000003)]
eam.xsections = [(1000001, -1000001), (1000001, 1000002)]
eam.xsections = [(1000021, 1000021)]
a = eam.get_type(eam.xsections)
print 'See here', a

eam.load_single_process(eam.xsections)
#print a[(1000001, 1000002)] # Getting out the production channel type
eam.eval_xsection(1000, 500, 600, 500, 600, 500, 500, 500, 500)
