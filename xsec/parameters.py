# Module containing dictionary of parameters and input output methods

import os

import pyslha   # Needs v3.2 or later

# TODO: Make compatible with Python 3

# Dictionary of all parameters and their values
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

# List of all parameter names
PARAM_NAMES = PARAMS.keys()

# List of mass parameters considered when taking the mean squark mass
MEAN_INDEX = ['m1000004', 'm1000003', 'm1000001', 'm1000002',
              'm2000002', 'm2000001', 'm2000003', 'm2000004']

# List of mixing angle parameters
MIXING_INDEX = ['thetab', 'thetat']


###############################################
# Set functions                               #
###############################################

def set_parameter(name, value):
    """
    Set single parameter with key name to value.
    """
    try:
        PARAMS[name] = float(value)
    except KeyError:
        print 'Parameter name \'%s\' not known!' % name
        raise
    except TypeError:
        print 'Parameter name \'%s\' should be set to a number!' % name
        raise



def set_parameters(params_in):
    """
    Set multiple parameters from a dictionary.
    """
    for name, value in params_in.items():
        set_parameter(name, float(value))


def calc_mean_squark_mass():
    """
    Calculate mean (first and second generation) squark mass and set the
    'mean' parameter.
    """
    m = sum([PARAMS[key] for key in MEAN_INDEX])/float(len(MEAN_INDEX))
    PARAMS['mean'] = m
    # return m


def set_common_squark_mass(mass):
    """
    Set all squark masses to the given common value.
    """
    set_parameters({
        'm1000001': mass,
        'm1000002': mass,
        'm1000003': mass,
        'm1000004': mass,
        'm1000005': mass,
        'm1000006': mass,
        'm2000001': mass,
        'm2000002': mass,
        'm2000003': mass,
        'm2000004': mass,
        'm2000005': mass,
        'm2000006': mass,
        'mean': mass
    })


def set_gluino_mass(mass):
    """
    Set the gluino mass to the given value.
    """
    set_parameter('m1000021', mass)


def clear_parameter(name):
    """
    Clear the value of a parameter.
    """
    try:
        PARAMS[name] = None
    except KeyError:
        print 'Parameter name \'%s\' not known!' % name
        raise


def clear_parameters(name_list=PARAM_NAMES):
    """
    Clear the values of a list of parameters. If no argument is
    given, all parameter values are erased.
    """
    for name in name_list:
        clear_parameter(name)


###############################################
# Get functions                               #
###############################################

def get_parameter(name):
    """
    Get the value of a parameter.
    """
    try:
        return PARAMS[name]
    except KeyError:
        print 'Parameter name %s not known!' % name
        raise


def get_parameters(name_list=None):
    """
    Get the values of a list of parameters. If no argument is
    given, a dictionary of all parameter values is returned.
    """
    if name_list is None:
        return PARAMS
    else:
        return [PARAMS[name] for name in name_list]


###############################################
# Check functions                             #
###############################################

def check_parameter(key):
    """
    Checks the consistency of a parameter.
    """
    # Check that the value has been supplied
    if PARAMS[key] is None:
        raise ValueError('The feature \'{feature}\' used in this cross-section'
                         ' evaluation has not been set!'.format(feature=key))
    # Check that the value is sensible
    # First check if we have a mixing parameter
    elif key in MIXING_INDEX:
        if abs(PARAMS[key]) > 1.:
            raise ValueError('The absolute value of the mixing angle '
                             '\'{feature}\' is greater than one!'
                             .format(feature=key))
    # If we get here we have a set mass parameter
    else:
        if PARAMS[key] > 4000:
            raise ValueError('The mass feature \'{feature}\' has been set to '
                             'a value ({value}) where the evaluation is an '
                             'extrapolation outside of training data.'
                             .format(feature=key, value=PARAMS[key]))
        elif PARAMS[key] < 0:
            raise ValueError('The mass feature \'{feature}\' has been set to '
                             'a negative value!'.format(feature=key))


def check_parameters(parameters):
    """
    Checks the consistency of a list of parameters.
    """
    # 1/ Check each individual parameter
    for par in parameters:
        check_parameter(par)

    # 2/ Check internal consistency of parameters
    # (For now just that the mean mass is set correctly. Only check when
    # all of the masses in MEAN_INDEX are specified, otherwise the
    # undefined mass(es) could be such that the user-specified mean mass
    # is correct.)

    # Collect the MEAN_INDEX masses in a list
    mean_index_masses = [PARAMS[key] for key in MEAN_INDEX]
    # Check only when all MEAN_INDEX masses are neither None nor zero
    if all(mean_index_masses):
        mean = sum(mean_index_masses)/float(len(mean_index_masses))
        # Compare correct mean computed now to user-specified mean
        if abs(PARAMS['mean'] - mean) > 0.1:
            raise ValueError(
                'The mean of the user-specified 1st and 2nd generation '
                'squark masses ({mean1}) is not equal to the '
                'specified \'mean\' mass feature ({mean2})!'
                .format(mean1=mean, mean2=PARAMS['mean']))


###############################################
# SLHA1 interface using pySLHA                #
###############################################

# TODO: Isn't this compatible with SLHA2 as well? It works for me ...

def import_slha(filename):
    """
    Import parameters from SLHA-file. This also calculates a mean squark
    mass.
    """
    # Try to open file (expand any environment variables and ~)
    filename = os.path.expandvars(os.path.expanduser(filename))
    try:
        slha = pyslha.read(filename, ignoreblocks=['DCINFO'])
        # TODO: More checking of reasonable file?
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
    calc_mean_squark_mass()

    # Find mixing angles
    PARAMS['thetab'] = slha.blocks['SBOTMIX'][1, 1]
    PARAMS['thetat'] = slha.blocks['STOPMIX'][1, 1]


def write_xsec(filename):
    """
    Write calculated cross sections to already existing SLHA file in
    XSECTION block.
    """
    # Try to open file (expand any environment variables and ~)
    filename = os.path.expandvars(os.path.expanduser(filename))
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
