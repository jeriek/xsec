# Module containing a dictionary of features for processes keyed by pid tuple pairs

features = { (1000021,1000021) : ['m1000021', 'm2000004', 'm2000003',
                                  'm2000002', 'm2000001', 'm1000004',
                                  'm1000003', 'm1000002', 'm1000001', 'mean'],
             (1000021,1000001) :  ['m1000021', 'm1000001', 'mean'],
             (1000021,1000002) :  ['m1000021', 'm1000002', 'mean'],
             (1000021,1000003) :  ['m1000021', 'm1000003', 'mean'],
             (1000021,1000004) :  ['m1000021', 'm1000004', 'mean']
           }

# Function that provides features for a tuple pid pair. Order of pids irrelevant.
# The function will raise errors when keys are not found.
def get_features(pid1, pid2):
    try:
        return features[(pid1,pid2)]
    except KeyError:
        try:
            return features[(pid2,pid1)]
        except KeyError as e:
            print 'Feature requested for process not in dictonary!', pid1, pid2
            raise e

#print get_features(1000021,1000001)
#print get_features(1000001,1000021)
