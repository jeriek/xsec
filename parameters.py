# Module containing dictionary of parameters

PARAMS = {
    'm1000001': None,
    'm1000002': None,
    'm1000003': None,
    'm1000004': None,
    'm1000005': None,
    'm1000006': None,
    'm2000001': None,
    'm2000002': None,
    'm2000003': None,
    'm2000004': None,
    'm2000005': None,
    'm2000006': None,
    'm1000021': None,
    'mean': None,
    'thetab': None,
    'thetat': None
}

MEAN_INDEX = ['m1000004', 'm1000003', 'm1000001', 'm1000002',
              'm2000002', 'm2000001', 'm2000003', 'm2000004']


# Function to import parameters from SLHA-file
def import_slha(filename):
    print 'Not implemented!'

# Set single parameter with key name to value
# TODO try/except
def set_parameter(name, value):
    if name not in PARAMS.keys():
        print 'Parameter name %s not known!' % name
        raise KeyError
    PARAMS[name] = value


def set_parameters(params_in):
    for name, value in params_in.items():
        set_parameter(name, value)

# Calculate mean (light) squark mass
def set_mean_mass():
    m = sum([PARAMS[key] for key in MEAN_INDEX])/float(len(MEAN_INDEX))
    PARAMS['mean'] = m
    return m

# set_parameter('rar',700)
# set_parameters({'m1000001':700,'m1000001':600})
#print PARAMS['m1000001'], PARAMS['m1000002']
