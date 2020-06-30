"""
Module containing a dictionary of parameters and input/output methods.
"""

from __future__ import print_function

import sys
import os

import pyslha

import xsec.utils as utils


###############################################
# Global variables                            #
###############################################

# Dictionary of all parameters and their values (only float or int!)
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
    "m1000022": None,  # chi_1^0 mass [GeV]
    "m1000023": None,  # chi_2^0 mass [GeV]
    "m1000025": None,  # chi_3^0 mass [GeV]
    "m1000035": None,  # chi_4^0 mass [GeV]
    "m1000024": None,  # chi_1^+ mass [GeV]
    "m1000037": None,  # chi_2^+ mass [GeV]
    "mean": None,  # mean mass of 1st and 2nd gen. squarks [GeV]
    "sbotmix11": None,  # sbottom mixing matrix element SBOTMIX [1, 1]
    "stopmix11": None,  # stop mixing matrix element STOPMIX [1, 1]
    "nmix11": None,  # neutralino mixing matrix element NMIX [1, 1]
    "nmix12": None,  # neutralino mixing matrix element NMIX [1, 2]
    "nmix13": None,  # neutralino mixing matrix element NMIX [1, 3]
    "nmix14": None,  # neutralino mixing matrix element NMIX [1, 4]
    "nmix21": None,  # neutralino mixing matrix element NMIX [2, 1]
    "nmix22": None,  # neutralino mixing matrix element NMIX [2, 2]
    "nmix23": None,  # neutralino mixing matrix element NMIX [2, 3]
    "nmix24": None,  # neutralino mixing matrix element NMIX [2, 4]
    "nmix31": None,  # neutralino mixing matrix element NMIX [3, 1]
    "nmix32": None,  # neutralino mixing matrix element NMIX [3, 2]
    "nmix33": None,  # neutralino mixing matrix element NMIX [3, 3]
    "nmix34": None,  # neutralino mixing matrix element NMIX [3, 4]
    "nmix41": None,  # neutralino mixing matrix element NMIX [4, 1]
    "nmix42": None,  # neutralino mixing matrix element NMIX [4, 2]
    "nmix43": None,  # neutralino mixing matrix element NMIX [4, 3]
    "nmix44": None,  # neutralino mixing matrix element NMIX [4, 4]
    "umix11": None,  # chargino mixing matrix element UMIX [1, 1]
    "umix12": None,  # chargino mixing matrix element UMIX [1, 2]
    "umix21": None,  # chargino mixing matrix element UMIX [2, 1]
    "umix22": None,  # chargino mixing matrix element UMIX [2, 2]
    "vmix11": None,  # chargino mixing matrix element VMIX [1, 1]
    "vmix12": None,  # chargino mixing matrix element VMIX [1, 2]
    "vmix21": None,  # chargino mixing matrix element VMIX [2, 1]
    "vmix22": None,  # chargino mixing matrix element VMIX [2, 2]
}

# CoM energy sqrt(s) [GeV]
COM_ENERGY = None

# List of all parameter names
PARAM_NAMES = list(PARAMS.keys())

# Dictionary of all parameter domains
PARAMS_DOM = {
    "m1000001": [200, 3000],  # squark mass [GeV]
    "m1000002": [200, 3000],  # squark mass [GeV]
    "m1000003": [200, 3000],  # squark mass [GeV]
    "m1000004": [200, 3000],  # squark mass [GeV]
    "m1000005": [100, 3000],  # squark mass [GeV]
    "m1000006": [100, 3000],  # squark mass [GeV]
    "m2000001": [200, 3000],  # squark mass [GeV]
    "m2000002": [200, 3000],  # squark mass [GeV]
    "m2000003": [200, 3000],  # squark mass [GeV]
    "m2000004": [200, 3000],  # squark mass [GeV]
    "m2000005": [100, 3000],  # squark mass [GeV]
    "m2000006": [100, 3000],  # squark mass [GeV]
    "m1000021": [200, 3000],  # gluino mass [GeV]
    "m1000022": [100, 1500],  # chi_1^0  mass [GeV]
    "m1000023": [100, 1500],  # chi_2^0  mass [GeV]
    "m1000025": [100, 1500],  # chi_3^0 mass [GeV]
    "m1000035": [100, 1500],  # chi_4^0 mass [GeV]
    "m1000024": [100, 1500],  # chi_1^+ mass [GeV]
    "m1000037": [100, 1500],  # chi_2^+ mass [GeV]
    "mean": [200, 3000],  # mean mass of 1st and 2nd gen. squarks [GeV]
    "sbotmix11": [-1, 1],  # sbottom mixing matrix element SBOTMIX [1, 1]
    "stopmix11": [-1, 1],  # stop mixing matrix element STOPMIX [1, 1]
    "nmix11": [-1, 1],  # neutralino mixing matrix element NMIX [1, 1]
    "nmix12": [-1, 1],  # neutralino mixing matrix element NMIX [1, 2]
    "nmix13": [-1, 1],  # neutralino mixing matrix element NMIX [1, 3]
    "nmix14": [-1, 1],  # neutralino mixing matrix element NMIX [1, 4]
    "nmix21": [-1, 1],  # neutralino mixing matrix element NMIX [2, 1]
    "nmix22": [-1, 1],  # neutralino mixing matrix element NMIX [2, 2]
    "nmix23": [-1, 1],  # neutralino mixing matrix element NMIX [2, 3]
    "nmix24": [-1, 1],  # neutralino mixing matrix element NMIX [2, 4]
    "nmix31": [-1, 1],  # neutralino mixing matrix element NMIX [3, 1]
    "nmix32": [-1, 1],  # neutralino mixing matrix element NMIX [3, 2]
    "nmix33": [-1, 1],  # neutralino mixing matrix element NMIX [3, 3]
    "nmix34": [-1, 1],  # neutralino mixing matrix element NMIX [3, 4]
    "nmix41": [-1, 1],  # neutralino mixing matrix element NMIX [4, 1]
    "nmix42": [-1, 1],  # neutralino mixing matrix element NMIX [4, 2]
    "nmix43": [-1, 1],  # neutralino mixing matrix element NMIX [4, 3]
    "nmix44": [-1, 1],  # neutralino mixing matrix element NMIX [4, 4]
    "umix11": [-1, 1],  # chargino mixing matrix element UMIX [1, 1]
    "umix12": [-1, 1],  # chargino mixing matrix element UMIX [1, 2]
    "umix21": [-1, 1],  # chargino mixing matrix element UMIX [2, 1]
    "umix22": [-1, 1],  # chargino mixing matrix element UMIX [2, 2]
    "vmix11": [-1, 1],  # chargino mixing matrix element VMIX [1, 1]
    "vmix12": [-1, 1],  # chargino mixing matrix element VMIX [1, 2]
    "vmix21": [-1, 1],  # chargino mixing matrix element VMIX [2, 1]
    "vmix22": [-1, 1],  # chargino mixing matrix element VMIX [2, 2]
}

# All allowed CoM energies
ALLOWED_ENERGIES = [7000, 8000, 13000, 14000]  # [GeV]

# List of mass parameters considered for the mean squark mass
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

# List of sparticle PDG IDs
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
ANTISQUARK_IDS = [-squark_id for squark_id in SQUARK_IDS]
ALL_SQUARK_IDS = SQUARK_IDS + ANTISQUARK_IDS
GEN3_SQUARK_IDS = [1000005, 1000006, 2000005, 2000006]
GEN3_ANTISQUARK_IDS = [-squark_id for squark_id in GEN3_SQUARK_IDS]
ALL_GEN3_IDS = GEN3_SQUARK_IDS + GEN3_ANTISQUARK_IDS
GLUINO_ID = 1000021
NEUTRALINO_IDS = [1000022, 1000023, 1000025, 1000035]
CHARGINO_IDS = [1000024, 1000037, -1000024, -1000037]
EWINO_IDS = NEUTRALINO_IDS + CHARGINO_IDS
SPARTICLE_IDS = SQUARK_IDS + ANTISQUARK_IDS + [GLUINO_ID] + EWINO_IDS


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
        set_parameter(name, value)


def calc_mean_squark_mass():
    """
    Calculate mean (first and second generation) squark mass and set the
    'mean' parameter.
    """
    mean_mass = sum([PARAMS[key] for key in MEAN_INDEX]) / float(
        len(MEAN_INDEX)
    )
    PARAMS["mean"] = mean_mass


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
    energy = int(energy)
    # Check energy
    if energy not in ALLOWED_ENERGIES:
        raise ValueError(
            "Currently the only available CoM energy is 13000 "
            "GeV. (The requested CoM energy was {energy} GeV.)".format(
                energy=energy
            )
        )
    else:
        if energy != 13000:
            print("Warning: your selected CoM energy is not 13000 GeV.")
        global COM_ENERGY
        COM_ENERGY = energy


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


def get_normalised_parameter(name):
    """
    Get the normalised value of a parameter.
    """
    try:
        return (PARAMS[name] - PARAMS_DOM[name][0]) / (
            PARAMS_DOM[name][1] - PARAMS_DOM[name][0]
        )
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
    # Else, if a specific parameter list is requested:
    return [PARAMS[name] for name in name_list]


def get_energy():
    """
    Get the value set for the CoM energy.
    """
    return COM_ENERGY


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
            "The feature '{feature}' used in this cross-section "
            "evaluation has not been set!".format(feature=key)
        )

    # Check that the value is in the domain of validity of the GPs
    elif not (PARAMS_DOM[key][0] <= PARAMS[key] <= PARAMS_DOM[key][1]):
        raise ValueError(
            "The mass feature '{feature}' has been set to "
            "a value ({value}) where the evaluation is an "
            "extrapolation outside of the validity of the "
            "training.".format(feature=key, value=PARAMS[key])
        )


def check_parameters(parameters):
    """
    Checks the consistency of a list of parameters.
    """
    # 1/ Check each individual parameter
    for par in parameters:
        check_parameter(par)

    # 2/ Check internal consistency of parameters
    # (For now just that the mean mass is set correctly. Only
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
                    mean1=mean, mean2=PARAMS["mean"], params=PARAMS
                )
            )


###############################################
# SLHA1 interface using pySLHA                #
###############################################


def import_slha(filepath):
    """
    Import parameters from SLHA file.
    This also calculates a mean squark mass for the first two generations.

    Parameters
    ----------
    filepath : string
        Path of SLHA file to import
    """

    # Try to open file (expand any environment variables and ~)
    filepath = os.path.abspath(
        os.path.expandvars(os.path.expanduser(filepath))
    )
    try:
        slha = pyslha.read(filepath, ignoreblocks=["DCINFO"])
        # TODO: More checking of reasonable file?
    except IOError:
        raise IOError(
            "Unable to find SLHA file {file}. Parameters not set.".format(
                file=filepath
            )
        )

    # Send the slha.Doc object to the import_slha_doc function
    import_slha_doc(slha)


def import_slha_string(slha_string):
    """
    Import parameters from a string in SLHA format.
    This also calculates a mean squark mass for the first two generations.

    Parameters
    ----------
    slha_string : string
        SLHA content to import
    """

    # Fix to ensure that the pyslha parsing of slha_string
    # works with both Python 2 and 3
    if sys.version_info < (3, 0):
        slha_string = slha_string.encode("ascii", "xmlcharrefreplace")

    # Parse the SLHA content with pyslha
    slha = pyslha.readSLHA(slha_string, ignoreblocks=["DCINFO"])

    # Send the slha.Doc object to the import_slha_doc function
    import_slha_doc(slha)


def import_slha_doc(slha):
    """
    Import parameters from a pyslha.Doc object.
    This also calculates a mean squark mass for the first two generations.

    Parameters
    ----------
    slha : pyslha.Doc
        SLHA content to import
    """

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

    PARAMS["m1000022"] = slha.blocks["MASS"][1000022]
    PARAMS["m1000023"] = slha.blocks["MASS"][1000023]
    PARAMS["m1000025"] = slha.blocks["MASS"][1000025]
    PARAMS["m1000035"] = slha.blocks["MASS"][1000035]
    PARAMS["m1000024"] = slha.blocks["MASS"][1000024]
    PARAMS["m1000037"] = slha.blocks["MASS"][1000037]

    # Also calculate mean squark mass
    calc_mean_squark_mass()

    # Find mixing angles
    PARAMS["sbotmix11"] = slha.blocks["SBOTMIX"][1, 1]
    PARAMS["stopmix11"] = slha.blocks["STOPMIX"][1, 1]

    PARAMS["nmix11"] = slha.blocks["NMIX"][1, 1]
    PARAMS["nmix12"] = slha.blocks["NMIX"][1, 2]
    PARAMS["nmix13"] = slha.blocks["NMIX"][1, 3]
    PARAMS["nmix14"] = slha.blocks["NMIX"][1, 4]
    PARAMS["nmix21"] = slha.blocks["NMIX"][2, 1]
    PARAMS["nmix22"] = slha.blocks["NMIX"][2, 2]
    PARAMS["nmix23"] = slha.blocks["NMIX"][2, 3]
    PARAMS["nmix24"] = slha.blocks["NMIX"][2, 4]
    PARAMS["nmix31"] = slha.blocks["NMIX"][3, 1]
    PARAMS["nmix32"] = slha.blocks["NMIX"][3, 2]
    PARAMS["nmix33"] = slha.blocks["NMIX"][3, 3]
    PARAMS["nmix34"] = slha.blocks["NMIX"][3, 4]
    PARAMS["nmix41"] = slha.blocks["NMIX"][4, 1]
    PARAMS["nmix42"] = slha.blocks["NMIX"][4, 2]
    PARAMS["nmix43"] = slha.blocks["NMIX"][4, 3]
    PARAMS["nmix44"] = slha.blocks["NMIX"][4, 4]

    PARAMS["umix11"] = slha.blocks["UMIX"][1, 1]
    PARAMS["umix12"] = slha.blocks["UMIX"][1, 2]
    PARAMS["umix21"] = slha.blocks["UMIX"][2, 1]
    PARAMS["umix22"] = slha.blocks["UMIX"][2, 2]
    PARAMS["vmix11"] = slha.blocks["VMIX"][1, 1]
    PARAMS["vmix12"] = slha.blocks["VMIX"][1, 2]
    PARAMS["vmix21"] = slha.blocks["VMIX"][2, 1]
    PARAMS["vmix22"] = slha.blocks["VMIX"][2, 2]

    # References to SLHA and pySLHA
    slharef = ["Skands:2003cj", "Buckley:2013jua"]
    utils.REF = list(set(utils.REF + slharef))


def write_slha(filename, results):
    """
    Write calculated cross sections to already existing SLHA file in
    XSECTION blocks.

    WARNING: Our treatment of PDF errors breaks the XSECTION standard
             by adding 1 and 2 to the central PDF set index to give the
             lower and upper 1-sigma uncertainty in cross section from
             the PDF variation sets, following PDF4LHC guidelines.
    """

    import xsec.gploader as gploader

    # Try to open file for appending (expand any environment variables and ~)
    filename = os.path.expandvars(os.path.expanduser(filename))
    try:
        slha = open(filename, "a")
    except IOError:
        raise IOError(
            "Unable to find SLHA file {file} for output. Cross sections not"
            "recorded.".format(file=filename)
        )

    # Set fixed entries
    # Find CoM energy
    sqrts = COM_ENERGY
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
        # Unpacking of result for more than one process
        (
            central,
            regdown_rel,
            regup_rel,
            scaledown_rel,
            scaleup_rel,
            pdfdown_rel,
            pdfup_rel,
            alphasdown_rel,
            alphasup_rel,
        ) = results[:, i]
        # Make process object
        proc = pyslha.Process(istate, fstate)

        # Add cross sections to process object.
        # NOTE: All the *_rel relative uncertainties are signed.
        #
        # proc.add_xsec(sqrts, scale_scheme, qcd_order, ew_order, kappa_f,
        # kappa_r, pdf_id, xs, code)
        def_args = (sqrts, scale_scheme, qcd_order, ew_order)
        # Central scale
        central_xs = central / 1000.0  # Convert to pb
        proc.add_xsec(*(def_args + (1.0, 1.0, pdf_id, central_xs, code)))
        # Half scale
        scaledown_xs = central_xs * (1.0 + scaledown_rel)
        proc.add_xsec(*(def_args + (0.5, 0.5, pdf_id, scaledown_xs, code)))
        # Double scale
        scaleup_xs = central_xs * (1.0 + scaleup_rel)
        proc.add_xsec(*(def_args + (2.0, 2.0, pdf_id, scaleup_xs, code)))
        # PDF down
        pdfdown_xs = central_xs * (1.0 + pdfdown_rel)
        proc.add_xsec(*(def_args + (1.0, 1.0, pdf_id + 1, pdfdown_xs, code)))
        # PDF up
        pdfup_xs = central_xs * (1.0 + pdfup_rel)
        proc.add_xsec(*(def_args + (1.0, 1.0, pdf_id + 2, pdfup_xs, code)))
        # \alpha_s
        asdown_xs = central_xs * (1.0 + alphasdown_rel)
        proc.add_xsec(*(def_args + (1.0, 1.0, pdf_id + 31, asdown_xs, code)))
        # \alpha_s up
        asup_xs = central_xs * (1.0 + alphasup_rel)
        proc.add_xsec(*(def_args + (1.0, 1.0, pdf_id + 32, asup_xs, code)))

        # Construct dictionary for writing
        xsection = {tuple(istate + fstate): proc}

        # Write cross section for particular process to file
        slha.write(pyslha.writeSLHAXSections(xsection, precision=5) + "\n")

    # Close file when finished writing
    slha.close()
