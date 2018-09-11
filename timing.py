#!/usr/bin/python

import time, random
import sys, os
import numpy as np
import evaluation as eval



def main(): 
	ntimes = 1000

	# eval.xsections = [(1000021, 1000001), (1000021, 1000003)]
	# eval.xsections = [(1000001, -1000001), (1000001, 1000002)]
	eval.xsections = [(1000021, 1000021)]
	start_parameters = (1000, 500, 600, 500, 600, 500, 500, 500, 500)
	eval.init(use_cache=False, cache_dir="$HOME/xsec_cache", flush_cache=False,\
		 use_memmap=False)

	# Start occupying all requested cores 
	# ---------------------------
	size = 5000
	A, B = np.random.random((size, size)), np.random.random((size, size))
	# Matrix multiplication
	N = 10
	t_wall = time.time()
	t_CPU = time.clock()
	for i in range(N):
	    np.dot(A, B)
	delta_wall = time.time() - t_wall
	delta_CPU = time.clock() - t_CPU
	print('Dotted two %dx%d matrices in %0.2f s. (wall time)' % (size, size, delta_wall / N))
	print('Dotted two %dx%d matrices in %0.2f s. (CPU time)' % (size, size, delta_CPU / N))
	del A, B
	# ---------------------------
	
	# Run actual cross-section evaluation code
	# ---------------------------
	blockPrint()
	TOTAL_UNPICKLE_TIME, TOTAL_K_LOAD_TIME, TOTAL_LOAD_TIME = \
		eval.load_processes(eval.xsections)

	avg_TOTAL_EVAL_SETUP_TIME, avg_TOTAL_EVAL_COMP_TIME, \
		avg_TOTAL_GP_COMP_TIME, avg_TOTAL_EVAL_TIME = \
		timing(eval.eval_xsection, ntimes, start_parameters)
	# ---------------------------	

	ndec = 5
	enablePrint()
	print "TOTAL_UNPICKLE_TIME = ", round(TOTAL_UNPICKLE_TIME, ndec) 
	print "TOTAL_K_LOAD_TIME = ", round(TOTAL_K_LOAD_TIME, ndec) 
	print "TOTAL_LOAD_TIME = ", round(TOTAL_LOAD_TIME, ndec) 
	print "avg_TOTAL_EVAL_SETUP_TIME = ", round(avg_TOTAL_EVAL_SETUP_TIME, ndec) 
	print "avg_TOTAL_GP_COMP_TIME = ", round(avg_TOTAL_GP_COMP_TIME, ndec) 
	print "avg_TOTAL_EVAL_COMP_TIME = ", round(avg_TOTAL_EVAL_COMP_TIME, ndec) 
	print "avg_TOTAL_EVAL_TIME = ", round(avg_TOTAL_EVAL_TIME, ndec) 

	return 0

def timing(f, n, arg):
	# print f.__name__,

	avg_TOTAL_EVAL_SETUP_TIME = 0.
	avg_TOTAL_EVAL_COMP_TIME = 0.
	avg_TOTAL_GP_COMP_TIME = 0.
	avg_TOTAL_EVAL_TIME = 0.

	random.seed(42) 

	for i in range(n):
		# Vary argument within +- 100
		new_arg = []
		for j in range(len(arg)):
			new_arg.append(arg[j] + ((-1)**(i+j))*100*random.random())
			# new_arg.append(arg[j]+((-1)**(i+j))*(i/100.))
		new_arg = tuple(new_arg)
		# print new_arg

		t1 = time.time()
		TOTAL_EVAL_SETUP_TIME, TOTAL_EVAL_COMP_TIME, TOTAL_GP_COMP_TIME = \
			f(*new_arg);
		t2 = time.time()
		# print TOTAL_EVAL_SETUP_TIME, TOTAL_EVAL_COMP_TIME, TOTAL_GP_COMP_TIME
		
		avg_TOTAL_EVAL_SETUP_TIME += TOTAL_EVAL_SETUP_TIME
		avg_TOTAL_EVAL_COMP_TIME += TOTAL_EVAL_COMP_TIME
		avg_TOTAL_GP_COMP_TIME += TOTAL_GP_COMP_TIME
		avg_TOTAL_EVAL_TIME += t2-t1

	avg_TOTAL_EVAL_SETUP_TIME /= n
	avg_TOTAL_EVAL_COMP_TIME /= n
	avg_TOTAL_GP_COMP_TIME /= n
	avg_TOTAL_EVAL_TIME /= n

	return avg_TOTAL_EVAL_SETUP_TIME, avg_TOTAL_EVAL_COMP_TIME, \
		avg_TOTAL_GP_COMP_TIME, avg_TOTAL_EVAL_TIME

# Disable printing
def blockPrint():
    sys.stdout = open(os.devnull, 'w')

# Restore printing
def enablePrint():
    sys.stdout = sys.__stdout__

if __name__ == '__main__':
	main()

