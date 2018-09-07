
import time
import evaluation as eval

def main():
	ntimes = 1

	# eval.xsections = [(1000021, 1000001), (1000021, 1000003)]
	# eval.xsections = [(1000001, -1000001), (1000001, 1000002)]
	eval.xsections = [(1000021, 1000021)]
	a = eval.get_type(eval.xsections)
	print 'Type of process: ', a

	#print a[(1000001, 1000002)] # Getting out the production channel type
	# eval.eval_xsection(1000, 500, 600, 500, 600, 500, 500, 500, 500)
	timing(eval.eval_xsection, ntimes, (1000, 500, 600, 500, 600, 500, 500, 500, 500))


def timing(f, n, a):
    print f.__name__,
    r = range(n)
    t1 = time.clock()
    for i in r:
        f(*a);# f(*a); f(*a); f(*a); f(*a); f(*a); f(*a); f(*a); f(*a); f(*a)
    t2 = time.clock()
    print "Total time: ", t2-t1
    print "Time for one eval_xsection: ", round((t2-t1)/(1.*n),8), "seconds"
    # print "Time for one eval_xsection: ", round((t2-t1)/(10.*n),8), "seconds"

if __name__ == '__main__':
	main()