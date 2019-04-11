"""
Internal helper functions.
"""

from __future__ import print_function

import sys
import os.path
import pkg_resources

import numpy as np

import xsec.parameters as parameters
import xsec.features as features
import xsec.gploader as gploader

###############################################
# Global variables                            #
###############################################

# Link internal cross-section type (xstype) identifiers to the
# corresponding training file suffixes for each pre-trained xstype
XSTYPE_FILESUFFIX = {
    "centr": "",  # xsection @ central scale
    "sclup": "_2",  # xsection @ higher scale (2 x central scale)
    "scldn": "_05",  # xsection @ lower scale (0.5 x central scale)
    "pdf": "_pdf",  # xsection error due to pdf variation
    "aup": "_aup",  # xsection @ higher alpha_s
    "adn": "_adn",  # xsection @ lower alpha_s
}

# List of the internal xstype identifiers
XSTYPES = XSTYPE_FILESUFFIX.keys()

# List keeping track of original physics references
REF = []

# Set the module's __version__ by getting the version set in setup.py
# (which in turn is set by the VERSION file). After a pip installation,
# the VERSION file is no longer accessible.  If xsec was not installed
# by pip, instead we try to find the VERSION file directly.
try:
    _dist = pkg_resources.get_distribution("xsec")
    dist_loc = _dist.location
    here = __file__
    if not here.startswith(os.path.join(dist_loc, "xsec")):
        # If this xsec is not installed, but there is another version
        # that is, i.e. the current __file__ parent dir is not in the
        # xsec distribution location that was auto-detected.
        raise pkg_resources.DistributionNotFound
except pkg_resources.DistributionNotFound:
    try:
        # If not pip-installed, VERSION file is in top xsec directory.
        # Get current xsec/xsec directory
        xsec_package_dir = os.path.dirname(__file__)
        # Get parent xsec directory in platform-independent way
        top_xsec_dir = os.path.abspath(
            os.path.join(xsec_package_dir, os.pardir)
        )
        # Read the version file
        with open(os.path.join(top_xsec_dir, "VERSION")) as version_file:
            __version__ = version_file.read().strip()
    except IOError:
        __version__ = "Unknown version!"
else:
    # If DistributionNotFound was not raised
    __version__ = _dist.version


###############################################
# Helper functions                            #
###############################################


def get_processdir_name(process_xstype):
    # Get the partons of the process
    parton1, parton2, xstype = get_process_id_split(process_xstype)

    # Decide process name
    # Check if one of the partons is a gluino
    if parton1 == parameters.GLUINO_ID:
        processdir_name = str(parton1) + "_" + str(parton2) + "_NLO"
    elif parton2 == parameters.GLUINO_ID:
        processdir_name = str(parton2) + "_" + str(parton1) + "_NLO"

    # Otherwise name starts with the largest parton PID
    elif abs(parton1) >= abs(parton2):
        processdir_name = str(parton1) + "_" + str(parton2) + "_NLO"
    elif abs(parton1) < abs(parton2):
        processdir_name = str(parton2) + "_" + str(parton1) + "_NLO"

    # Add training file suffixes depending on the xstype
    try:
        processdir_name += XSTYPE_FILESUFFIX[xstype]
    except KeyError:
        raise KeyError(
            "Error: {xstype} is not a valid variation parameter!".format(
                xstype=xstype
            )
        )

    return processdir_name


def get_process_id(process, xstype):
    assert len(process) == 2
    process_xstype = (process[0], process[1], xstype)

    return process_xstype


def get_str_from_process_id(process_xstype):
    assert len(process_xstype) == 3
    process_xstype_str = (
        str(process_xstype[0])
        + "_"
        + str(process_xstype[1])
        + "_"
        + process_xstype[2]
    )

    return process_xstype_str


def get_process_id_split(process_xstype):
    assert len(process_xstype) == 3
    parton1 = process_xstype[0]
    parton2 = process_xstype[1]
    xstype = process_xstype[2]

    return parton1, parton2, xstype


def get_process_from_process_id(process_xstype):
    parton1 = process_xstype[0]
    parton2 = process_xstype[1]

    return (parton1, parton2)


def get_xstype_from_process_id(process_xstype):
    xstype = process_xstype[-1]

    return xstype


###############################################
# Printing                                    #
###############################################

# Print result of run
def print_result(return_array, verbose=2):
    """
    Output an evaluation result to screen. The given amount of detail
    can be adjusted.

    Parameters
    ----------
    return_array : array
        Evaluation result as returned by evaluation.eval_xsection().

    verbose : int, optional
        Degree of verbosity when printing results to screen:
        - verbose=0 : prints nothing
        - verbose=1 : prints a single line per process (no header),
            in the following format:
              PID1 PID2 xsection_central regdown_rel regup_rel scaledown_rel
              scaleup_rel pdfdown_rel pdfup_rel alphasdown_rel alphasup_rel
        - verbose=2 : prints full description of the results (default)

    """

    # Verbose level 0: print no description at all
    if verbose is 0:
        pass
    # Verbose level 1: print single-line description of each process
    elif verbose is 1:
        nr_dec = 4
        process_list = gploader.PROCESSES
        for i, process in enumerate(process_list):
            pid1 = process[0]
            pid2 = process[1]

            xsection_central = np.round(return_array[0][i], decimals=nr_dec)
            regdown_rel = np.round(return_array[1][i], decimals=nr_dec)
            regup_rel = np.round(return_array[2][i], decimals=nr_dec)
            scaledown_rel = np.round(return_array[3][i], decimals=nr_dec)
            scaleup_rel = np.round(return_array[4][i], decimals=nr_dec)
            pdfdown_rel = np.round(return_array[5][i], decimals=nr_dec)
            pdfup_rel = np.round(return_array[6][i], decimals=nr_dec)
            alphasdown_rel = np.round(return_array[7][i], decimals=nr_dec)
            alphasup_rel = np.round(return_array[8][i], decimals=nr_dec)

            print(pid1, pid2, xsection_central,
                  regdown_rel, regup_rel, scaledown_rel, scaleup_rel,
                  pdfdown_rel, pdfup_rel, alphasdown_rel, alphasup_rel)
            sys.stdout.flush()

    # Verbose level 2: print full description of the result
    elif verbose is 2:
        print(
            "\n"
            "\t    _/      _/    _/_/_/  _/_/_/_/    _/_/_/   \n"
            "\t     _/  _/    _/        _/        _/          \n"
            "\t      _/        _/_/    _/_/_/    _/           \n"
            "\t   _/  _/          _/  _/        _/            \n"
            "\t_/      _/  _/_/_/    _/_/_/_/    _/_/_/       \n"
        )
        sys.stdout.flush()
        nr_dec = 4
        np.set_printoptions(precision=nr_dec)
        process_list = gploader.PROCESSES
        print("* Requested processes, in order:\n   ", end="")
        for process in process_list:
            print(process, " ", end="")
        print()
        print("* Input features:")
        for process in process_list:
            feature_names = features.get_features(*process)
            feature_values = features.get_features_dict(process_list)[process]
            print("  ", process, ": \n      [", end="")
            for i, feature in enumerate(feature_names):
                print(
                    feature, "=",
                    str(np.round(feature_values[i], decimals=nr_dec)),
                    "\b, ", end=""
                    )
            print("\b\b]")

        print("* xsection_central (fb):", return_array[0])
        print("* regdown_rel:   ", return_array[1])
        print("* regup_rel:     ", return_array[2])
        print("* scaledown_rel: ", return_array[3])
        print("* scaleup_rel:   ", return_array[4])
        print("* pdfdown_rel:   ", return_array[5])
        print("* pdfup_rel:     ", return_array[6])
        print("* alphasdown_rel:", return_array[7])
        print("* alphasup_rel:  ", return_array[8])
        print("**************************************************************")
        sys.stdout.flush()
    else:
        raise ValueError(
            "The verbosity level can only be set to one of these options:\n"
            "\t - verbose=0 (print nothing to screen)\n"
            "\t - verbose=1 (print single line per process)\n"
            "\t - verbose=2 (default, print full description of results)\n"
            )


def print_references(file_handle=None):
    """"
    Print references to screen, or to a file if a file handle is
    supplied.
    """
    for reference in REF:
        bibtex = fetch_bibtex(reference)
        if file_handle:
            file_handle.write(bibtex)
            file_handle.write("\n\n")
        else:
            print(bibtex)


def get_references(pid1, pid2):
    """
    Collect references for a particular process (stored in global
    REFERENCES).
    """
    # Prospino and PDF4LHC references currently common to all processes
    newref = ["Beenakker:1996ed", "Butterworth:2015oua"]

    # Add process specific papers to reference list
    if (pid1 == 1000006 and pid2 == -1000006) or (
        pid1 == 1000005 and pid2 == -1000005
    ):
        newref.append("Beenakker:1997ut")
    else:
        newref.append("Beenakker:1996ch")

    # Add to global (unique content)
    global REF
    REF = list(set(REF + newref))


def fetch_bibtex(ref):
    """
    Fetch bibtex content for references.
    """
    if ref == "Beenakker:1996ch":
        bibtex = '@article{Beenakker:1996ch,\n      author         = "Beenakker, W. and Hopker, R. and Spira, M. and Zerwas, P.M.",\n      title          = "{Squark and gluino production at hadron colliders}",\n      journal        = "Nucl. Phys.",\n      volume         = "B492",\n      year           = "1997",\n      pages          = "51-103",\n      doi            = "10.1016/S0550-3213(97)80027-2",\n      eprint         = "hep-ph/9610490",\n      archivePrefix  = "arXiv",\n      primaryClass   = "hep-ph",\n      reportNumber   = "DESY-96-150, CERN-TH-96-215",\n      SLACcitation   = "%%CITATION = HEP-PH/9610490;%%"\n}'
    elif ref == "Beenakker:1996ed":
        bibtex = '@article{Beenakker:1996ed,\n        author         = "Beenakker, W. and Hopker, R. and Spira, M.",\n        title          = "{PROSPINO: A Program for the production of supersymmetric\n                          particles in next-to-leading order QCD}",\n        year           = "1996",\n        eprint         = "hep-ph/9611232",\n        archivePrefix  = "arXiv",\n        primaryClass   = "hep-ph",\n        SLACcitation   = "%%CITATION = HEP-PH/9611232;%%"\n}'
    elif ref == "Beenakker:1997ut":
        bibtex = '@article{Beenakker:1997ut,\n        author         = "Beenakker, W. and Kramer, M. and Plehn, T. and Spira, M.\n                          and Zerwas, P. M.",\n        title          = "{Stop production at hadron colliders}",\n        journal        = "Nucl. Phys.",\n        volume         = "B515",\n        year           = "1998",\n        pages          = "3-14",\n        doi            = "10.1016/S0550-3213(98)00014-5",\n        eprint         = "hep-ph/9710451",\n        archivePrefix  = "arXiv",\n        primaryClass   = "hep-ph",\n        reportNumber   = "DESY-97-214, CERN-TH-97-177, RAL-TR-97-056",\n        SLACcitation   = "%%CITATION = HEP-PH/9710451;%%"\n}'
    elif ref == "Butterworth:2015oua":
        bibtex = '@article{Butterworth:2015oua,\n        author         = "Butterworth, Jon and others",\n        title          = "{PDF4LHC recommendations for LHC Run II}",\n        journal        = "J. Phys.",\n        volume         = "G43",\n        year           = "2016",\n        pages          = "023001",\n        doi            = "10.1088/0954-3899/43/2/023001",\n        eprint         = "1510.03865",\n        archivePrefix  = "arXiv",\n        primaryClass   = "hep-ph",\n        reportNumber   = "OUTP-15-17P, SMU-HEP-15-12, TIF-UNIMI-2015-14,\n                          LCTS-2015-27, CERN-PH-TH-2015-249",\n        SLACcitation   = "%%CITATION = ARXIV:1510.03865;%%"\n}'
    elif ref == "Buckley:2013jua":
        bibtex = '@article{Buckley:2013jua,\n    author         = "Buckley, Andy",\n    title          = "{PySLHA: a Pythonic interface to SUSY Les Houches Accord\n                      data}",\n    journal        = "Eur. Phys. J.",\n    volume         = "C75",\n    year           = "2015",\n    umber         = "10",\n    pages          = "467",\n    doi            = "10.1140/epjc/s10052-015-3638-8",\n    eprint         = "1305.4194",\n    archivePrefix  = "arXiv",\n    primaryClass   = "hep-ph",\n    reportNumber   = "MCNET-13-06, GLAS-PPE-2015-02",\n    SLACcitation   = "%%CITATION = ARXIV:1305.4194;%%"\n}'
    elif ref == "Skands:2003cj":
        bibtex = '@article{Skands:2003cj,\n    author         = "Skands, Peter Z. and others",\n    title          = "{SUSY Les Houches accord: Interfacing SUSY spectrum\n                      calculators, decay packages, and event generators}",\n    journal        = "JHEP",\n    volume         = "07",\n    year           = "2004",\n    pages          = "036",\n    doi            = "10.1088/1126-6708/2004/07/036",\n    eprint         = "hep-ph/0311123",\n    archivePrefix  = "arXiv",\n    primaryClass   = "hep-ph",\n    reportNumber   = "LU-TP-03-39, SHEP-03-24, CERN-TH-2003-204, ZU-TH-15-03,\n                      LMU-19-03, DCPT-03-108, IPPP-03-54, CTS-IISC-2003-07,\n                      DESY-03-166, MPP-2003-111",\n    SLACcitation   = "%%CITATION = HEP-PH/0311123;%%"}'
    else:
        bibtex = ""

    return bibtex
