# Module containing dictionary of parameters and input output methods

from __future__ import print_function

import os

import pyslha

import xsec.utils as utils
import xsec.gploader as gploader


# Dictionary of all parameters and their values
PARAMS = {
    "m1000001": None,  # squark mass [GeV]
    "m1000002": None,  # squark mass [GeV]
    "m1000003": None,  # squark mass [GeV]
    "m1000004": None,  # squark mass [GeV]
    "m1000005": None,  # squark mass [GeV]
    "m1000006": None,  # squark mass [GeV]
    "m2000001": None,  # squark mass [GeV]
    "m2000002": None,  # squark mass [GeV]
    "m2000003": None,  # squark mass [GeV]
    "m2000004": None,  # squark mass [GeV]
    "m2000005": None,  # squark mass [GeV]
    "m2000006": None,  # squark mass [GeV]
    "m1000021": None,  # gluino mass [GeV]
    "mean": None,  # mean mass of 1st and 2nd gen. squarks [GeV]
    "sbotmix11": None,
    "stopmix11": None,
    "energy": None,  # CoM energy sqrt(s) [GeV]
}

# List of all parameter names
PARAM_NAMES = PARAMS.keys()

# List of mass parameters considered when taking the mean squark mass
MEAN_INDEX = [
    "m1000004",
    "m1000003",
    "m1000001",
    "m1000002",
    "m2000002",
    "m2000001",
    "m2000003",
    "m2000004",
]

# List of mixing angle parameters
MIXING_INDEX = ["sbotmix11", "stopmix11"]

# List of sparticle PDG ids
SQUARK_IDS = [
    1000001,
    1000002,
    1000003,
    1000004,
    1000005,
    1000006,
    2000001,
    2000002,
    2000003,
    2000004,
    2000005,
    2000006,
]
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
        print("Parameter name {name} not known!".format(name=name))
        raise
    except TypeError:
        print(
            "Parameter name {name} should be set to a number!".format(
                name=name
            )
        )
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
    m = sum([PARAMS[key] for key in MEAN_INDEX]) / float(len(MEAN_INDEX))
    PARAMS["mean"] = m


def set_all_squark_masses(mass):
    """
    Set all squark masses to the given common value, and set the mean
    squark mass to that value too.
    """
    set_parameters(
        {
            "m1000001": mass,
            "m1000002": mass,
            "m1000003": mass,
            "m1000004": mass,
            "m1000005": mass,
            "m1000006": mass,
            "m2000001": mass,
            "m2000002": mass,
            "m2000003": mass,
            "m2000004": mass,
            "m2000005": mass,
            "m2000006": mass,
            "mean": mass,
        }
    )


def set_12gen_squark_masses(mass):
    """
    Set 1st and 2nd generation squark masses to the given common value,
    and set the mean squark mass to that value too.
    """
    set_parameters(
        {
            "m1000001": mass,
            "m1000002": mass,
            "m1000003": mass,
            "m1000004": mass,
            "m2000001": mass,
            "m2000002": mass,
            "m2000003": mass,
            "m2000004": mass,
            "mean": mass,
        }
    )


def set_gluino_mass(mass):
    """
    Set the gluino mass to the given value.
    """
    set_parameter("m1000021", mass)


def set_energy(energy):
    """
    Set the CoM energy sqrt(s), in GeV.
    """
    set_parameter("energy", energy)


def clear_parameter(name):
    """
    Clear the value of a parameter.
    """
    try:
        PARAMS[name] = None
    except KeyError:
        print("Parameter name {name} not known!".format(name=name))
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
        print("Parameter name {name} not known!".format(name=name))
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
        raise ValueError(
            "The feature '{feature}' used in this cross-section"
            " evaluation has not been set!".format(feature=key)
        )
    # Check that the value is sensible
    # First check if we have a mixing parameter
    elif key in MIXING_INDEX:
        if abs(PARAMS[key]) > 1.0:
            raise ValueError(
                "The absolute value of the mixing angle "
                "'{feature}' is greater than one!".format(feature=key)
            )
    # If we get here we have a set mass parameter
    else:
        if PARAMS[key] > 3000:
            raise ValueError(
                "The mass feature '{feature}' has been set to "
                "a value ({value}) where the evaluation is an "
                "extrapolation outside of training data.".format(
                    feature=key, value=PARAMS[key]
                )
            )
        elif PARAMS[key] < 50:
            raise ValueError(
                "The mass feature '{feature}' has been set close to "
                "half the Z-mass. Cross section predictions here "
                "may be unreliable!".format(feature=key, value=PARAMS[key]
                )
            )


def check_parameters(parameters):
    """
    Checks the consistency of a list of parameters.
    """
    # 1/ Check each individual parameter
    for par in parameters:
        check_parameter(par)

    # 2/ Check internal consistency of parameters
    # (For now just that the mean mass and energy is set correctly. Only
    # check when all of the masses in MEAN_INDEX are specified,
    # otherwise the undefined mass(es) could be such that the
    # user-specified mean mass is correct.)

    # Collect the MEAN_INDEX masses in a list
    mean_index_masses = [PARAMS[key] for key in MEAN_INDEX]
    # Check only when all MEAN_INDEX masses are neither None nor zero
    if all(mean_index_masses):
        mean = sum(mean_index_masses) / float(len(mean_index_masses))
        # Compare correct mean computed now to user-specified mean
        if abs(PARAMS["mean"] - mean) > 0.1:
            raise ValueError(
                "\n The mean of the user-specified 1st and 2nd generation "
                "squark masses ({mean1}) is not equal to the "
                "specified 'mean' mass feature ({mean2})!\n"
                "Perhaps you may want to clear some parameters from a "
                "previous evaluation? You can do this by calling "
                "clear_parameters(<list-of-parameter-names>), or "
                "clear_parameters() to clear all parameters.\n"
                "Currently, the parameters have been set to:\n{params}\n"
                "[This consistency check was performed as the evaluation "
                "was called with the relevant (default) option:\n\t"
                "eval_xsection(..., check_consistency=True).]".format(
                    mean1=mean, mean2=PARAMS["mean"]
                )
            )

    # Check energy
    if PARAMS["energy"] != 13000:
        raise ValueError(
            "Currently the only available CoM energy is 13000 GeV. (The "
            "requested CoM energy was {energy} GeV.)".format(
                energy=PARAMS["energy"]
                )
            )


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
        slha = pyslha.read(filename, ignoreblocks=["DCINFO"])
        # TODO: More checking of reasonable file?
    except IOError as e:
        print(
            "Unable to find SLHA file {file}. Parameters not set.".format(
                file=filename
            )
        )
        raise e

    # Find masses
    PARAMS["m1000001"] = slha.blocks["MASS"][1000001]
    PARAMS["m1000002"] = slha.blocks["MASS"][1000002]
    PARAMS["m1000003"] = slha.blocks["MASS"][1000003]
    PARAMS["m1000004"] = slha.blocks["MASS"][1000004]
    PARAMS["m1000005"] = slha.blocks["MASS"][1000005]
    PARAMS["m1000006"] = slha.blocks["MASS"][1000006]
    PARAMS["m2000001"] = slha.blocks["MASS"][2000001]
    PARAMS["m2000002"] = slha.blocks["MASS"][2000002]
    PARAMS["m2000003"] = slha.blocks["MASS"][2000003]
    PARAMS["m2000004"] = slha.blocks["MASS"][2000004]
    PARAMS["m2000005"] = slha.blocks["MASS"][2000005]
    PARAMS["m2000006"] = slha.blocks["MASS"][2000006]
    PARAMS["m1000021"] = slha.blocks["MASS"][1000021]

    # Also calculate mean squark mass
    calc_mean_squark_mass()

    # Find mixing angles
    PARAMS["sbotmix11"] = slha.blocks["SBOTMIX"][1, 1]
    PARAMS["stopmix11"] = slha.blocks["STOPMIX"][1, 1]

    # References to SLHA and pySLHA
    slharef = ["Skands:2003cj", "Buckley:2013jua"]
    utils.REF = list(set(utils.REF + slharef))


def write_slha(filename, results):
    """
    Write calculated cross sections to already existing SLHA file in
    XSECTION blocks.

    WARNING: Our treatment of PDF errors breaks the XSECTION standard 
             by adding 1 and 2 to the central PDF set index to give the 
             lower and upper 1\sigma uncertainty in cross section from the 
             PDF variation sets following PDF4LHC guidelines.
    """
    
    # Try to open file for appending (expand any environment variables and ~)
    filename = os.path.expandvars(os.path.expanduser(filename))
    try:
        slha = open(filename, "a")
    except IOError as e:
        print(
            "Unable to find SLHA file {file} for output. Cross sections not"
            "recorded.".format(name=filename)
        )
        raise e

    # Set fixed entries
    # Find CoM energy
    sqrts = PARAMS["energy"]
    # For the time being we are only doing pp cross sections
    istate = [2212, 2212]
    # Calculation currently based on QCD NLO, Prospino uses average mass
    # as scale
    scale_scheme, qcd_order, ew_order = 0, 1, 0
    # We currently use PDF4LHC15 PDFs
    pdf_id = 90400
    # Advertise our smoking code
    code = ["xsec", utils.__version__]

    # Get the processes we have calculated
    processes = gploader.PROCESSES
    # Loop over processes
    for i, process in enumerate(processes):
        fstate = [process[0], process[1]]
        # Indexing of result for more than one process
        result = results[:, i]
        # Make process object
        proc = pyslha.Process(istate, fstate)
        # Add cross sections to process object
        central_xs = result[0] / 1000.0  # Convert to pb
        xs = central_xs
        # proc.add_xsec(sqrts, scale_scheme, qcd_order, ew_order, kappa_f, kappa_r, pdf_id, xs, code)
        proc.add_xsec(
            sqrts,
            scale_scheme,
            qcd_order,
            ew_order,
            1.0,
            1.0,
            pdf_id,
            xs,
            code,
        )  # Central scale
        xs = central_xs * result[3]
        proc.add_xsec(
            sqrts,
            scale_scheme,
            qcd_order,
            ew_order,
            2.0,
            2.0,
            pdf_id,
            xs,
            code,
        )  # Double scale
        xs = central_xs * result[4]
        proc.add_xsec(
            sqrts,
            scale_scheme,
            qcd_order,
            ew_order,
            0.5,
            0.5,
            pdf_id,
            xs,
            code,
        )  # Half scale
        xs = central_xs * result[5]
        proc.add_xsec(
            sqrts,
            scale_scheme,
            qcd_order,
            ew_order,
            1.0,
            1.0,
            pdf_id + 1,
            xs,
            code,
        )  # PDF down
        xs = central_xs * result[6]
        proc.add_xsec(
            sqrts,
            scale_scheme,
            qcd_order,
            ew_order,
            1.0,
            1.0,
            pdf_id + 2,
            xs,
            code,
        )  # PDF up
        xs = central_xs * result[7]
        proc.add_xsec(
            sqrts,
            scale_scheme,
            qcd_order,
            ew_order,
            1.0,
            1.0,
            pdf_id + 31,
            xs,
            code,
        )  # \alpha_s down
        xs = central_xs * result[8]
        proc.add_xsec(
            sqrts,
            scale_scheme,
            qcd_order,
            ew_order,
            1.0,
            1.0,
            pdf_id + 32,
            xs,
            code,
        )  # \alpha_s up

        # Construct dictionary for writing
        xsection = {tuple(istate + fstate): proc}

        # Write cross section for particular process to file
        slha.write(pyslha.writeSLHAXSections(xsection, precision=5) + "\n")

    # Close file when finished writing
    slha.close()

