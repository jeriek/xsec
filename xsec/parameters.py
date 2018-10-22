# Module containing dictionary of parameters and input output methods

import pyslha   # Needs v3.2 or later

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


################################
# SLHA1 interface using pySLHA #
################################

# Import parameters from SLHA-file. This also calculates a mean squark mass
def import_slha(filename):
    # Try to open file
    try:
        slha = pyslha.read(filename, ignoreblocks=['DCINFO']) # TODO: More checking of reasonable file?
    except IOError as e:
        print 'Unable to find SLHA file %s. Parameters not set.' % filename
        raise e

    # Find masses
    PARAMS['m1000001'] = slha.blocks['MASS'][1000001]
    PARAMS['m1000002'] = slha.blocks['MASS'][1000002]
    PARAMS['m1000003'] = slha.blocks['MASS'][1000003]
    PARAMS['m1000004'] = slha.blocks['MASS'][1000004]
    PARAMS['m1000005'] = slha.blocks['MASS'][1000005]
    PARAMS['m1000006'] = slha.blocks['MASS'][1000006]
    PARAMS['m2000001'] = slha.blocks['MASS'][2000001]
    PARAMS['m2000002'] = slha.blocks['MASS'][2000002]
    PARAMS['m2000003'] = slha.blocks['MASS'][2000003]
    PARAMS['m2000004'] = slha.blocks['MASS'][2000004]
    PARAMS['m2000005'] = slha.blocks['MASS'][2000005]
    PARAMS['m2000006'] = slha.blocks['MASS'][2000006]
    PARAMS['m1000021'] = slha.blocks['MASS'][1000021]
    # Also calculate mean squark mass
    set_mean_mass()
    # Find mixing angles
    PARAMS['thetab'] = slha.blocks['SBOTMIX'][1,1]
    PARAMS['thetat'] = slha.blocks['STOPMIX'][1,1]

# Write calculated cross sections to SLHA file in XSECTION block
def write_slha(filename):
    print 'Not implemented!'


#import_slha('sps2a.slha')
#for key in PARAMS:
#    print key, PARAMS[key]
