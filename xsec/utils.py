# coding=utf-8
"""
Internal helper functions.
"""

from __future__ import print_function

import sys
import os.path
import pkg_resources

import numpy as np

###############################################
# Global variables                            #
###############################################

# Link internal cross-section type (xstype) identifiers to the
# corresponding training file suffixes for each pre-trained xstype
XSTYPE_FILESUFFIX = {
    "centr": "_1",  # xsection @ central scale
    "sclup": "_2",  # xsection @ higher scale (2 x central scale)
    "scldn": "_05",  # xsection @ lower scale (0.5 x central scale)
    "pdf": "_pdf",  # xsection error due to pdf variation
    "aup": "_aup",  # xsection @ higher alpha_s
    "adn": "_adn",  # xsection @ lower alpha_s
}

# List of the internal xstype identifiers
XSTYPES = list(XSTYPE_FILESUFFIX.keys())

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


def get_processdir_name(process_xstype, energy):
    """
    Get the name of the directory where the GPs for a specific process
    and cross-section type are stored. The PIDs in the process tuple
    should already match those of the trained process, in the right
    order, as provided by features.get_trained_process().

    Parameters
    ----------
    process_xstype : str
        String containing parton1, parton2, xstype.

    energy : int or float
        Numerical value specifying the COM energy in GeV.

    Returns
    -------
    processdir_name : str
        The name of the relevant GP directory.

    """

    # Get the partons of the process
    parton1, parton2, xstype = get_process_id_split(process_xstype)

    # Decide process name
    processdir_name = (
        str(parton1) + "_" + str(parton2) + "_" + str(int(energy)) + "_NLO"
    )

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
    """
    Get a 3-tuple specifying the particles in a process and a
    cross-section type.
    """

    assert len(process) == 2
    process_xstype = (process[0], process[1], xstype)

    return process_xstype


def get_str_from_process_id(process_xstype):
    """
    Get a single string specifying the particles in a process and a
    cross-section type.
    """

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
    """
    Get the three individual elements of a 3-tuple specifying a process
    and a cross-section type.
    """

    assert len(process_xstype) == 3
    parton1 = process_xstype[0]
    parton2 = process_xstype[1]
    xstype = process_xstype[2]

    return parton1, parton2, xstype


def get_process_from_process_id(process_xstype):
    """
    Get a 2-tuple with the PIDs in the process from a 3-tuple specifying
    a process and a cross-section type.
    """

    parton1 = process_xstype[0]
    parton2 = process_xstype[1]

    return (parton1, parton2)


def get_xstype_from_process_id(process_xstype):
    """
    Get the cross-section type from a 3-tuple specifying a process and a
    cross-section type.
    """
    xstype = process_xstype[-1]

    return xstype


def unknown_process_error(pid1, pid2):
    """
    Raise an error noting that there is no training data for the
    process requested.
    """
    return KeyError(
        "Unknown process requested: ({pid1}, {pid2}) "
        "is not in the list of allowed processes!".format(pid1=pid1, pid2=pid2)
    )


###############################################
# Printing                                    #
###############################################

# Print result of run
def print_result(return_array, process_list, verbose=2):
    """
    Output an evaluation result to screen. The given amount of detail
    can be adjusted.

    Parameters
    ----------
    return_array : array
        Evaluation result as returned by evaluation.eval_xsection().

    process_list : list
        List of processes as stored in gploader.PROCESSES.

    verbose : int, optional
        Degree of verbosity when printing results to screen:
        - verbose=0 : prints nothing
        - verbose=1 : prints a single line per process (no header),
            in the following format:
              PID1 PID2 xsection_central regdown_rel regup_rel scaledown_rel
              scaleup_rel pdfdown_rel pdfup_rel alphasdown_rel alphasup_rel
        - verbose=2 : prints full description of the results (default)

    """
    import xsec.features as features

    # Verbose level 0: print no description at all
    if verbose == 0:
        pass
    # Verbose level 1: print single-line description of each process
    elif verbose == 1:
        for i, process in enumerate(process_list):
            pid1 = process[0]
            pid2 = process[1]

            xsection_central = return_array[0][i]
            regdown_rel = return_array[1][i]
            regup_rel = return_array[2][i]
            scaledown_rel = return_array[3][i]
            scaleup_rel = return_array[4][i]
            pdfdown_rel = return_array[5][i]
            pdfup_rel = return_array[6][i]
            alphasdown_rel = return_array[7][i]
            alphasup_rel = return_array[8][i]

            # Print one line, using scientific notation with 4 decimals
            result_str = "  {: d} {: d}  {: .4e}".format(
                pid1, pid2, xsection_central
            )
            result_str += "  {: .4e}  {: .4e}".format(regdown_rel, regup_rel)
            result_str += "  {: .4e}  {: .4e}".format(
                scaledown_rel, scaleup_rel
            )
            result_str += "  {: .4e}  {: .4e}".format(pdfdown_rel, pdfup_rel)
            result_str += "  {: .4e}  {: .4e}".format(
                alphasdown_rel, alphasup_rel
            )

            print(result_str)
            sys.stdout.flush()

    # Verbose level 2: print full description of the result
    elif verbose == 2:
        # 'Static' variable to ensure the xsec banner is only printed
        # the first time this function is run (with verbose=2).
        if "print_banner" not in print_result.__dict__:
            print_result.print_banner = True
        # Print banner
        if print_result.print_banner:
            print(
                "\n"
                "\t    _/      _/    _/_/_/  _/_/_/_/    _/_/_/   \n"
                "\t     _/  _/    _/        _/        _/          \n"
                "\t      _/        _/_/    _/_/_/    _/           \n"
                "\t   _/  _/          _/  _/        _/            \n"
                "\t_/      _/  _/_/_/    _/_/_/_/    _/_/_/       \n"
            )
            sys.stdout.flush()
            print_result.print_banner = False

        np.set_printoptions(precision=4, formatter={"float": "{: .4e}".format})
        print("* Requested processes, in order:\n   ", end="")
        for process in process_list:
            print(process, " ", end="")
        print()
        print("* Input features:")
        for process in process_list:
            feature_names = features.get_features(*process)
            feature_values = features.get_features_dict(
                process_list, auto_normf=False
            )[process]
            print("  ", process, ": \n      [", end="")
            for i, feature in enumerate(feature_names):
                print(
                    feature,
                    "=",
                    str(np.round(feature_values[i], decimals=4)),
                    "\b, ",
                    end="",
                )
                if (i + 1) % 3 == 0:
                    print()
                    print("       ", end="")
            print("\b\b]")

        # Switch from np.object array to make np.set_printoptions work
        return_array = return_array.astype(np.double)

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
    """
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

    from xsec.parameters import EWINO_IDS, ALL_GEN3_IDS

    # xsec, Prospino, PDF4LHC and LHAPDF references currently common to
    # all processes
    newref = [
        "Buckley:2020bxg",
        "Beenakker:1996ed",
        "Buckley:2014ana",
        "Butterworth:2015oua",
    ]

    # Add process specific papers to reference list
    if pid1 in ALL_GEN3_IDS and pid2 == -pid1:
        newref.append("Beenakker:1997ut")
    elif pid1 in EWINO_IDS and pid2 in EWINO_IDS:
        newref.append("Beenakker:1999xh")
    else:
        newref.append("Beenakker:1996ch")

    # Add to global (unique content)
    global REF
    REF = list(set(REF + newref))


def fetch_bibtex(ref):
    """
    Fetch bibtex content for references.
    """
    if ref == "Buckley:2020bxg":
        bibtex = (
            "@article{Buckley:2020bxg,\n"
            + "\t author         = {Buckley, Andy and Kvellestad, Anders and Raklev, Are and\n"
            + "\t                   Scott, Pat and Sparre, Jon Vegard and \n"
            + "\t                   Abeele, Jeriek Van den and Vazquez-Holm, Ingrid A.},\n"
            + "\t title          = {$\\textsf{Xsec}$: the cross-section evaluation code}},\n"
            + "\t eprint         = {2006.16273},\n"
            + "\t archivePrefix  = {arXiv},\n"
            + "\t primaryClass   = {hep-ph},\n"
            + "\t reportNumber   = {SAGEX-20-17-E},\n"
            + "\t month          = {6},\n"
            + "\t year           = {2020}\n"
            + "}"
        )

    elif ref == "Beenakker:1996ch":
        bibtex = (
            "@article{Beenakker:1996ch,\n"
            + "\t author         = {Beenakker, W. and Höpker, R. and Spira, M. and Zerwas, P.M.},\n"
            + "\t title          = {{Squark and gluino production at hadron colliders}},\n"
            + "\t journal        = {Nucl. Phys.},\n"
            + "\t volume         = {B492},\n"
            + "\t year           = {1997},\n"
            + "\t pages          = {51-103},\n"
            + "\t doi            = {10.1016/S0550-3213(97)80027-2},\n"
            + "\t eprint         = {hep-ph/9610490},\n"
            + "\t archivePrefix  = {arXiv},\n"
            + "\t primaryClass   = {hep-ph},\n"
            + "\t reportNumber   = {DESY-96-150, CERN-TH-96-215},\n"
            + "\t SLACcitation   = {%%CITATION = HEP-PH/9610490;%%}\n"
            + "}"
        )
    elif ref == "Beenakker:1996ed":
        bibtex = (
            "@article{Beenakker:1996ed,\n"
            + "\t author         = {Beenakker, W. and Höpker, R. and Spira, M.},\n"
            + "\t title          = {{PROSPINO: A Program for the production of supersymmetric\n"
            + "\t                   particles in next-to-leading order QCD}},\n"
            + "\t year           = {1996},\n"
            + "\t eprint         = {hep-ph/9611232},\n"
            + "\t archivePrefix  = {arXiv},\n"
            + "\t primaryClass   = {hep-ph},\n"
            + "\t SLACcitation   = {%%CITATION = HEP-PH/9611232;%%}\n"
            + "}"
        )
    elif ref == "Beenakker:1997ut":
        bibtex = (
            "@article{Beenakker:1997ut,\n"
            + "\t author         = {Beenakker, W. and Krämer, M. and Plehn, T. and Spira, M.\n"
            + "\t                   and Zerwas, P. M.},\n"
            + "\t title          = {{Stop production at hadron colliders}},\n"
            + "\t journal        = {Nucl. Phys.},\n"
            + "\t volume         = {B515},\n"
            + "\t year           = {1998},\n"
            + "\t pages          = {3-14},\n"
            + "\t doi            = {10.1016/S0550-3213(98)00014-5},\n"
            + "\t eprint         = {hep-ph/9710451},\n"
            + "\t archivePrefix  = {arXiv},\n"
            + "\t primaryClass   = {hep-ph},\n"
            + "\t reportNumber   = {DESY-97-214, CERN-TH-97-177, RAL-TR-97-056},\n"
            + "\t SLACcitation   = {%%CITATION = HEP-PH/9710451;%%}\n"
            + "}"
        )
    elif ref == "Beenakker:1999xh":
        bibtex = (
            "@article{Beenakker:1999xh,\n"
            + "\t author         = {Beenakker, W. and Klasen, M. and Krämer, M. and Plehn, T.\n"
            + "\t                   and Spira, M. and Zerwas, P. M.},\n"
            + "\t title          = {{The Production of charginos / neutralinos and sleptons at hadron colliders}},\n"
            + "\t journal        = {Phys. Rev. Lett.},\n"
            + "\t volume         = {83},\n"
            + "\t year           = {1999},\n"
            + "\t pages          = {3780-3783},\n"
            + "\t doi            = {10.1103/PhysRevLett.100.029901, 10.1103/PhysRevLett.83.3780},\n"
            + "\t note           = {[Erratum: Phys. Rev. Lett. 100, 029901 (2008)]},\n"
            + "\t eprint         = {hep-ph/9906298},\n"
            + "\t archivePrefix  = {arXiv},\n"
            + "\t primaryClass   = {hep-ph},\n"
            + "\t reportNumber   = {CERN-TH-99-159, DESY-99-055, DTP-99-44, MADPH-99-1114, ANL-HEP-PR-99-71},\n"
            + "\t SLACcitation   = {%%CITATION = HEP-PH/9906298;%%}\n"
            + "}"
        )
    elif ref == "Butterworth:2015oua":
        bibtex = (
            "@article{Butterworth:2015oua,\n"
            + "\t author         = {Butterworth, Jon and others},\n"
            + "\t title          = {{PDF4LHC recommendations for LHC Run II}},\n"
            + "\t journal        = {J. Phys.},\n"
            + "\t volume         = {G43},\n"
            + "\t year           = {2016},\n"
            + "\t pages          = {023001},\n"
            + "\t doi            = {10.1088/0954-3899/43/2/023001},\n"
            + "\t eprint         = {1510.03865},\n"
            + "\t archivePrefix  = {arXiv},\n"
            + "\t primaryClass   = {hep-ph},\n"
            + "\t reportNumber   = {OUTP-15-17P, SMU-HEP-15-12, TIF-UNIMI-2015-14,\n"
            + "\t                   LCTS-2015-27, CERN-PH-TH-2015-249},\n"
            + "\t SLACcitation   = {%%CITATION = ARXIV:1510.03865;%%}\n"
            + "}"
        )
    elif ref == "Buckley:2013jua":
        bibtex = (
            "@article{Buckley:2013jua,\n"
            + "\t author         = {Buckley, Andy},\n"
            + "\t title          = {{PySLHA: a Pythonic interface to SUSY Les Houches Accord data}},\n"
            + "\t journal        = {Eur. Phys. J.},\n"
            + "\t volume         = {C75},\n"
            + "\t year           = {2015},\n"
            + "\t number         = {10},\n"
            + "\t pages          = {467},\n"
            + "\t doi            = {10.1140/epjc/s10052-015-3638-8},\n"
            + "\t eprint         = {1305.4194},\n"
            + "\t archivePrefix  = {arXiv},\n"
            + "\t primaryClass   = {hep-ph},\n"
            + "\t reportNumber   = {MCNET-13-06, GLAS-PPE-2015-02},\n"
            + "\t SLACcitation   = {%%CITATION = ARXIV:1305.4194;%%}\n"
            + "}"
        )
    elif ref == "Skands:2003cj":
        bibtex = (
            "@article{Skands:2003cj,\n"
            + "\t author         = {Skands, Peter Z. and others},\n"
            + "\t title          = {{SUSY Les Houches accord: Interfacing SUSY spectrum\n"
            + "\t                   calculators, decay packages, and event generators}},\n"
            + "\t journal        = {JHEP},\n"
            + "\t volume         = {07},\n"
            + "\t year           = {2004},\n"
            + "\t pages          = {036},\n"
            + "\t doi            = {10.1088/1126-6708/2004/07/036},\n"
            + "\t eprint         = {hep-ph/0311123},\n"
            + "\t archivePrefix  = {arXiv},\n"
            + "\t primaryClass   = {hep-ph},\n"
            + "\t reportNumber   = {LU-TP-03-39, SHEP-03-24, CERN-TH-2003-204, ZU-TH-15-03,\n"
            + "\t                   LMU-19-03, DCPT-03-108, IPPP-03-54, CTS-IISC-2003-07,\n"
            + "\t                   DESY-03-166, MPP-2003-111},\n"
            + "\t SLACcitation   = {%%CITATION = HEP-PH/0311123;%%}\n"
            + "}"
        )
    elif ref == "Buckley:2014ana":
        bibtex = (
            "@article{Buckley:2014ana,\n"
            + "\t author         = {Buckley, Andy and Ferrando, James and Lloyd, Stephen and Nordström, Karl\n"
            + "\t                   and Page, Ben and Rüfenacht, Martin and Schönherr, Marek and Watt, Graeme},\n"
            + "\t title          = {{LHAPDF6: parton density access in the LHC precision era}},\n"
            + "\t journal        = {Eur. Phys. J.},\n"
            + "\t volume         = {C75},\n"
            + "\t year           = {2015},\n"
            + "\t pages          = {132},\n"
            + "\t doi            = {10.1140/epjc/s10052-015-3318-8},\n"
            + "\t eprint         = {1412.7420},\n"
            + "\t archivePrefix  = {arXiv},\n"
            + "\t primaryClass   = {hep-ph},\n"
            + "\t reportNumber   = {GLAS-PPE-2014-05, MCNET-14-29, IPPP-14-111, DCPT-14-222},\n"
            + "\t SLACcitation   = {%%CITATION = ARXIV:1412.7420;%%}\n"
            + "}"
        )
    else:
        bibtex = ""

    return bibtex


def list_all_xsec_processes():
    """
    Print a list of PID pairs of all the processes currently available in xsec.
    """
    from xsec.features import FEATURES_LIST, UNVALIDATED_PROCESSES

    # Retrieve and sort the list of PID pairs
    pid_pairs = list(FEATURES_LIST.keys())
    pid_pairs.sort()
    # Print the list with appropriate spacing
    print("List of PID pairs (pid1, pid2) of all trained processes in xsec:")
    for i, pid_pair in enumerate(pid_pairs):
        # Skip unvalidated processes
        if pid_pair in UNVALIDATED_PROCESSES:
            continue
        if i % 3 == 0:
            print()
        print("\t", pid_pair, sep="", end="")
    print("\n")
