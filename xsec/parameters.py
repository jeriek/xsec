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

MIXING_INDEX = ['thetab', 'thetat']

# Set single parameter with key name to value
def set_parameter(name, value):
    if name not in PARAMS.keys():
        print 'Parameter name %s not known!' % name
        raise KeyError
    PARAMS[name] = value

# Set multiple parameters from a dictionary
def set_parameters(params_in):
    for name, value in params_in.items():
        set_parameter(name, value)

# Calculate mean (first and second generation) squark mass
def set_mean_mass():
    m = sum([PARAMS[key] for key in MEAN_INDEX])/float(len(MEAN_INDEX))
    PARAMS['mean'] = m
    return m

# Checks the consistencey of a parameter
def check_parameter(key):
    # Check that the value has been supplied
    if PARAMS[key] is None:
        raise ValueError(
                         'The feature {feature} used in this cross section '
                         'evaluation has not been set!'.format(feature=key))
    # Check that the value is sensible
    # First check if we have a mixing parameter
    elif key in MIXING_INDEX:
        if abs(PARAMS[key]) > 1. :
            raise ValueError('The absolute value of the mixing angle {feature} '
                             'is greater than one!'
                             .format(feature=key))
    # If we get here we have a set mass parameter
    else:
        if PARAMS[key] > 4000:
            raise ValueError('The mass feature {feature} has been set to a '
                             'value ({value}) where the evaluation is an '
                             'extrapolation outside of training data.'
                             .format(feature=key, value=PARAMS[key]))
        elif PARAMS[key] < 0:
            raise ValueError('The mass feature {feature} has been set to a '
                             'negative value!'.format(feature=key))

# Checks the consistencey of a list of parameters
def check_parameters(parameters):
    # Check each individual parameter
    for par in parameters:
        check_parameter(par)
    # Check internal consistency of parameters.
    # For now just that the mean mass is set correctly
    mean = 0
    nsquark = 0
    for key in MEAN_INDEX:
        if PARAMS[key] is not None:
            mean += PARAMS[key]
            nsquark += 1
    mean = mean/8.
    if nsquark == 8 and abs(PARAMS['mean'] - mean) > 0.1:
        raise ValueError(
                         'The squark masses mean {mean1} is not equal to the '
                         'mean mass feature used {mean2}!'
                         .format(mean1=mean, mean2=PARAMS['mean']))


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

# Write calculated cross sections to already existing SLHA file in XSECTION block
def write_xsec(filename):
    # Try to open file
    try:
        slha = pyslha.read(filename, ignoreblocks=['DCINFO'])
    except IOError as e:
        print 'Unable to find SLHA file %s. Cross sections not recorded.' % filename
        raise e

    sqrts = 13000.
    qcd_order = 1
    ew_order = 0
    kappa_f = 1.
    kappa_r = 1.
    pdf_id = 10000 # fake
    value = 5.
    print 'Not implemented!'

