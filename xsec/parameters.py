# Module containing dictionary of parameters and input output methods

from __future__ import print_function
import os

import utils
import gploader

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
    'mean'    : None,
    'thetab'  : None,
    'thetat'  : None,
    'energy'  : None       # CoM energy sqrt(s) in GeV
}

# List of all parameter names
PARAM_NAMES = PARAMS.keys()

# List of mass parameters considered when taking the mean squark mass
MEAN_INDEX = ['m1000004', 'm1000003', 'm1000001', 'm1000002',
              'm2000002', 'm2000001', 'm2000003', 'm2000004']

# List of mixing angle parameters
MIXING_INDEX = ['thetab', 'thetat']

# List of sparticle PDG ids
SQUARK_IDS = [1000001, 1000002, 1000003, 1000004, 1000005, 1000006,
              2000001, 2000002, 2000003, 2000004, 2000005, 2000006]
ANTISQUARK_IDS = [-id for id in SQUARK_IDS]
GLUINO_ID = 1000021
SPARTICLE_IDS = SQUARK_IDS + ANTISQUARK_IDS + [GLUINO_ID]


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
        print('Parameter name {name} not known!'.format(name=name))
        raise
    except TypeError:
        print('Parameter name {name} should be set to a number!'.format(name=name))
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


def set_energy(energy):
    """
    Set the CoM energy sqrt(s), in GeV.
    """
    set_parameter('energy', energy)


def clear_parameter(name):
    """
    Clear the value of a parameter.
    """
    try:
        PARAMS[name] = None
    except KeyError:
        print('Parameter name {name} not known!'.format(name=name))
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
        print('Parameter name {name} not known!'.format(name=name))
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
    # (For now just that the mean mass and energy is set correctly. Only check
    # when all of the masses in MEAN_INDEX are specified, otherwise the
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

    # Check energy
    if PARAMS['energy'] != 13000 :
        raise ValueError('Currently the only available energy is 13000 GeV')


###############################################
# SLHA1 interface using pySLHA                #
###############################################

def import_slha(filename):
    """
    Import parameters from SLHA-file.
    This also calculates a mean squark mass for the first two generations.
    """
    # Try to open file (expand any environment variables and ~)
    filename = os.path.expandvars(os.path.expanduser(filename))
    try:
        slha = pyslha.read(filename, ignoreblocks=['DCINFO'])
        # TODO: More checking of reasonable file?
    except IOError as e:
        print('Unable to find SLHA file {file}. Parameters not set.'.format(file=filename))
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

    # References to SLHA and pySLHA
    slharef = ['Skands:2003cj','Buckley:2013jua']
    utils.REF = list(set(utils.REF+slharef))

def write_slha(filename, results):
    """
    Write calculated cross sections to already existing SLHA file in
    XSECTION block.
    """
    # Try to open file for appending (expand any environment variables and ~)
    filename = os.path.expandvars(os.path.expanduser(filename))
    try:
        slha = open(filename,'a')
    except IOError as e:
        print('Unable to find SLHA file {file} for output. Cross sections not'
              'recorded.'.format(name=filename))
        raise e

    # Set fixed entries
    # Find CoM energy
    sqrts = PARAMS['energy']
    # For the time being we are only doing pp cross sections
    istate = [2212, 2212]
    # Calculation currently based on QCD NLO, Prospino uses average mass as scale
    scale_scheme, qcd_order, ew_order  = 0, 1, 0
    # We currently use PDF4LHC15 PDFs
    pdf_id = 90400
    # Advertise our smoking code
    # TODO: Get the version number automagically
    code = ["xsec", "v0.1.0"]

    # Get the processes we have calculated
    processes = gploader.PROCESSES
    # Loop over processes
    for i, process in enumerate(processes):
        fstate = [process[0], process[1]]
        # Make process object
        proc = pyslha.Process(istate, fstate)
        # Add cross sections to process object
        # TODO: Handle indexing for more than one process
        # WARNING: PDF variations break XSECTION standard by adding 1 and 2 to the central PDF set
        #          for lower and upper 1\sigma variation
        central_xs = results[0]/1000.   # Convert to pb
        xs = central_xs
        # proc.add_xsec(sqrts, scale_scheme, qcd_order, ew_order, kappa_f, kappa_r, pdf_id, xs, code)
        proc.add_xsec(sqrts, scale_scheme, qcd_order, ew_order, 1.0, 1.0, pdf_id, xs, code)    # Central scale
        xs = central_xs*results[3]
        proc.add_xsec(sqrts, scale_scheme, qcd_order, ew_order, 2.0, 2.0, pdf_id, xs, code)    # Double scale
        xs = central_xs*results[4]
        proc.add_xsec(sqrts, scale_scheme, qcd_order, ew_order, 0.5, 0.5, pdf_id, xs, code)    # Half scale
        xs = central_xs*results[5]
        proc.add_xsec(sqrts, scale_scheme, qcd_order, ew_order, 1.0, 1.0, pdf_id+1, xs, code)  # PDF down
        xs = central_xs*results[6]
        proc.add_xsec(sqrts, scale_scheme, qcd_order, ew_order, 1.0, 1.0, pdf_id+2, xs, code)  # PDF up
        xs = central_xs*results[7]
        proc.add_xsec(sqrts, scale_scheme, qcd_order, ew_order, 1.0, 1.0, pdf_id+31, xs, code) # \alpha_s down
        xs = central_xs*results[8]
        proc.add_xsec(sqrts, scale_scheme, qcd_order, ew_order, 1.0, 1.0, pdf_id+32, xs, code) # \alpha_s up

        # Construct dictionary for writing
        xsection = { tuple(istate + fstate) : proc }

        # Write cross section for particular process to file
        #print(pyslha.writeSLHAXSections(xsection))
        slha.write(pyslha.writeSLHAXSections(xsection,precision=5)+'\n')

    print('XSECTION block writing routine not yet complete! Use at own risk!')
