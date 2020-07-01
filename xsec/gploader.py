"""
Initialisation, GP loading functions and cache management.
"""

from __future__ import print_function

import os
import gc
import imp
import warnings
from collections import OrderedDict

import joblib  # Needs v0.12.2 or later

import xsec.utils as utils
import xsec.kernels as kernels
import xsec.parameters as parameters


###############################################
# Global variables                            #
###############################################

# The GP model data directory. By default, DATA_DIR gets set to "gprocs"
# when init() is executed, such that xsec looks for a directory called
# "gprocs" in the current working directory. This can be overwritten by
# directly accessing DATA_DIR in the run script (before loading any
# processes), or, preferably, by specifying the data_dir keyword inside
# init().
DATA_DIR = ""

# List of selected processes (2-tuples of sparticle ids), to be set by
# the user
PROCESSES = []  # e.g. [(1000021, 1000021), (1000021, 1000002)]

# For each selected process, store trained GP model dictionaries here
# (or a list of their cache locations), with process_xstype as key
PROCESS_DICT = OrderedDict()

# For each selected process and GP expert, store the reconstructed
# kernel functions
KERNEL_DICT = OrderedDict()

# Dictionary of transform.py modules for the selected processes, with
# (process, xstype) as key
TRANSFORM_MODULES = OrderedDict()

# Default settings for using Joblib memory caching, can be
# modified by user with init()
USE_CACHE = False
CACHE_DIR = ""
FLUSH_CACHE = True
USE_MEMMAP = True
CACHE_MEMORY = None


###############################################
# Initialisation                              #
###############################################


def init(
    data_dir="gprocs",
    use_cache=False,
    cache_dir="",
    flush_cache=True,
    use_memmap=True,
):
    """
    Initialise run settings for the program. In particular, whether to
    use a cache, i.e. a temporary disk directory to store loaded GP
    models for use in predictive functions. This could be useful if the
    memory load needs to be decreased, when loading many large models.

    Parameters
    ----------
    data_dir : str, optional
        Specify the path of the top directory containing all of the GP
        model data directories. By default, init() tries to find a
        directory called "gprocs" in the current working directory.
    use_cache : bool, optional
        Specify whether to cache data on disk (default False).
    cache_dir: str, optional
        Specify a disk directory for the cache. Inactive if use_cache is
        False. If "", a random directory is created. (default "")
    flush_cache: bool, optional
        Specify whether to clear disk cache after completing the
        program. Inactive if use_cache is False. Warning: if False,
        non-empty temporary directories will persist on disk, if using
        cache, and may have to be deleted manually by the user.
        (default True)
    use_memmap : bool, optional
        Specify whether to use memory mapping when loading Numpy arrays
        into cache, which may speed up GP model readout during
        cross-section evaluation. Inactive if use_cache is False.
        (default True)
    """

    # --- Setting the data directory
    global DATA_DIR
    # If the global variable was directly set by the user, search for that path
    if DATA_DIR:
        DATA_DIR = os.path.abspath(
            os.path.expandvars(os.path.expanduser(DATA_DIR))
        )
    # If set, use the given data_dir
    # (by default "gprocs", such that ./gprocs in the current working
    # directory is set as GP directory)
    # This will override the direct setting of DATA_DIR.
    if data_dir:
        DATA_DIR = os.path.abspath(
            os.path.expandvars(os.path.expanduser(data_dir))
        )
    # Else, if the user set both DATA_DIR and data_dir to ""
    else:
        raise IOError(
            "The path to the Gaussian process data directories cannot "
            "be empty. Please ensure that the right path is entered with the "
            "data_dir keyword in init()."
        )
    # Check whether the given path is an existing directory
    if not os.path.isdir(DATA_DIR):
        raise IOError(
            "The Gaussian process data directories could not be found "
            "at {dir}. Please ensure that the right path is entered with the "
            "data_dir keyword in init().".format(dir=DATA_DIR)
        )

    # --- Fixing global variables coordinating the use of caching
    global USE_CACHE, CACHE_DIR, FLUSH_CACHE, USE_MEMMAP
    USE_CACHE = use_cache
    CACHE_DIR = cache_dir
    FLUSH_CACHE = flush_cache
    USE_MEMMAP = use_memmap

    if USE_CACHE:
        if CACHE_DIR:
            # Set cache directory to given name, expand any environment
            # variables in the name
            cachedir = os.path.abspath(
                os.path.expandvars(os.path.expanduser(CACHE_DIR))
            )
        else:
            # Create directory with random name
            from tempfile import mkdtemp

            cachedir = mkdtemp(prefix="xsec_")

        if USE_MEMMAP:
            # Set memmap mode 'read-only'
            memmap_mode = "r"
        else:
            # Disable memmapping
            memmap_mode = None

        # Create a Joblib Memory object managing the cache
        global CACHE_MEMORY
        CACHE_MEMORY = joblib.Memory(
            location=cachedir, mmap_mode=memmap_mode, verbose=0
        )
        print("Cache folder: " + str(cachedir))


def check_process_input(process):
    """
    Check the input format of a process tuple. Returns True if valid,
    raises an error if invalid.
    """
    # Check if process exists (right format, known sparticles)
    if len(process) != 2:
        raise ValueError(
            "The entered process tuple ({input}) does not consist of "
            "exactly _two_ particle IDs from the following list: "
            "\n {ids}".format(input=process, ids=parameters.SPARTICLE_IDS)
        )
    if not all((pid in parameters.SPARTICLE_IDS) for pid in process):
        raise ValueError(
            "One or more particle IDs entered ({input}) are not in the"
            " allowed set of IDs: \n {ids}".format(
                input=process, ids=parameters.SPARTICLE_IDS
            )
        )
    return True


def get_processes():
    """
    Return the current list of processes to be evaluated.
    """
    return PROCESSES


###############################################
# Loading functions                           #
###############################################


def reconstruct_gp(model_file):
    """
    Decompress and reconstruct an individual GP model stored in a
    .gproc file.

    Parameters
    ----------
    model_file : str
        Path to the .gproc file.

    Returns
    -------
    gp_reco : dict
        Dictionary containing all information about an individual GP.
    """
    # Open the stored GP model file of a single expert
    with open(model_file, "rb") as file_object:
        # Unzip the binary file with Joblib, yielding dict
        gp_model = joblib.load(file_object)
        # If necessary, reconvert float32 arrays to float64 for
        # higher-precision computations, filling a new dict gp_reco
        gp_reco = {}
        gp_reco["X_train"] = gp_model["X_train"].astype("float64")
        gp_reco["alpha"] = gp_model["alpha"].astype("float64")
        L_inv = gp_model["L_inv"].astype("float64")
        # Compute K_inv from L_inv and store it in the dict
        gp_reco["K_inv"] = L_inv.dot(L_inv.T)
        # Read GP expert index from last part of file name
        gp_reco["index"] = int(
            os.path.splitext(os.path.basename(model_file))[0].split("_")[-1]
        )
        # Kernel lambda function reconstructed later since not picklable
        gp_reco["kernel_string"] = gp_model["kernel_string"]
    return gp_reco


def load_single_process(process_xstype, energy):
    """
    Given a single process, cross-section type (e.g. gluino-gluino at
    the central scale) and COM energy, load the relevant trained GP
    models for all experts and return them in a list of dictionaries,
    one per expert.

    Parameters
    ----------
    process_xstype : tuple
        The input argument process_xstype is a 3-tuple
        (process[0], process[1], var) where the first two components
        are integers specifying the process and the last component is a
        string from utils.XSTYPES.
        Example: (1000021, 1000021, 'centr')

    energy : int
        Integer value specifying the COM energy in GeV.

    Returns
    -------
    model_dict : dict of dict
        Dictionary containing one dictionary per expert trained on the
        process and cross-section type specified in process_xstype. The
        keys of model_dict are the index of the GP expert, starting from
        1 for the communications expert. Each GP dictionary has keys
        'X_train', 'K_inv', 'alpha' and 'kernel'. These components are
        all the information needed to make the predictions of the expert
        with GP_predict(). If using cache, a reference to the location
        of the cached data is stored in model_dict, indexed by GP
        expert.
    """
    assert len(process_xstype) == 3

    if energy not in parameters.ALLOWED_ENERGIES:
        raise ValueError(
            "Please ensure the center-of-mass energy is set with set_energy(),"
            " before loading the GP models.\n"
            "Currently the only available CoM energies are 7000/8000/13000/"
            "14000 GeV. (The requested CoM energy was [{energy}] GeV.)".format(
                energy=parameters.COM_ENERGY
            )
        )

    # Construct location of GP models for the specified process and
    # cross-section type, using global data directory variable DATA_DIR
    process_dir = os.path.join(
        DATA_DIR, utils.get_processdir_name(process_xstype, energy)
    )

    # Collect the GP model data file locations (and avoid loading
    # transform.py or __init__.py from the model directory)
    if os.path.isdir(process_dir):
        candidate_model_files = [
            os.path.join(process_dir, f) for f in os.listdir(process_dir)
        ]
        model_files = []
        for a_file in candidate_model_files:
            if os.path.isfile(a_file):
                # Require .gproc extension for GP model files
                if a_file.lower().endswith((".gproc")):
                    model_files.append(a_file)
        # Raise error if list of GP model files stays empty
        if not model_files:
            raise IOError(
                "No GP data files (*.gproc) found at {}.".format(process_dir)
            )
    else:
        raise IOError("No valid directory found at {}.".format(process_dir))

    # Initialise dict of GP model dicts
    model_dict = OrderedDict()

    # Loop over experts and add one GP model dict per expert to model_dict
    if USE_CACHE:
        # Decorate reconstruct_gp() such that its output can be
        # cached using the Joblib Memory object and will only be computed
        # for new input arguments; otherwise the cached value is used
        reconstruct_gp_cache = CACHE_MEMORY.cache(reconstruct_gp, verbose=0)
    for model_file in model_files:
        if USE_CACHE:
            gp_reco = reconstruct_gp_cache.call_and_shelve(model_file)
            # Reconstruct kernel function (as a lambda function)
            KERNEL_DICT[
                (process_xstype, gp_reco.get()["index"])
            ] = kernels.get_kernel(gp_reco.get()["kernel_string"])
            # Add the cache reference to the model dictionary, with the
            # GP expert index as key
            model_dict[gp_reco.get()["index"]] = gp_reco
        else:
            gp_reco = reconstruct_gp(model_file)
            # Reconstruct kernel function (as a lambda function)
            KERNEL_DICT[
                (process_xstype, gp_reco["index"])
            ] = kernels.get_kernel(gp_reco["kernel_string"])
            # Key the model dictionary by the GP expert index
            model_dict[gp_reco["index"]] = gp_reco

    # Add transform.py file corresponding to the process_xstype to the
    # modules dictionary (keyword: process, xstype; value: corresponding
    # transform module)
    transform_file_path = os.path.join(process_dir, "transform.py")
    process = utils.get_process_from_process_id(process_xstype)
    xstype = utils.get_xstype_from_process_id(process_xstype)
    try:
        TRANSFORM_MODULES[(process, xstype)] = imp.load_source(
            "transform_" + utils.get_str_from_process_id(process_xstype),
            transform_file_path,
        )
    except IOError:
        raise IOError(
            "Could not find transform.py file in {dir}.".format(
                dir=process_dir
            )
        )
    return model_dict


def load_processes(process_list):
    """
    Given a list of sparticle production processes, load all relevant
    trained GP models into memory, or into a cache folder on disk if
    using cache. The function calls load_single_process() for each
    process in process_list, looping over all cross-section types in
    utils.XSTYPES. It stores each returned list of models in the global
    dictionary PROCESS_DICT, indexed with key 'process_xstype'.

    Parameters
    ----------
    process_list : list of tuple
        The input argument is a list of 2-tuples
        (process[0], process[1]), where the components are integers
        specifying the process. For example, gluino-gluino production
        corresponds to the tuple (1000021, 1000021).
    """
    import xsec.features as features

    if not process_list:
        raise ValueError("List of processes to be evaluated cannot be empty.")

    # Get the requested COM energy from the parameters
    # This requires setting the parameters BEFORE loading processes!
    energy = parameters.COM_ENERGY

    # Loop over specified processes
    for process in process_list:
        # Check validity of input
        check_process_input(process)
        # Convert to trained process code!
        trained_process = features.get_trained_process(*process)
        # Search for all directories with same process, accounting for
        # different cross-section types
        for xstype in utils.XSTYPES:
            process_xstype = utils.get_process_id(trained_process, xstype)
            # Loaded GP models (or their cache reference) are stored in
            # PROCESS_DICT
            PROCESS_DICT[process_xstype] = load_single_process(
                process_xstype, energy
            )

        # Add the process to the global list of processes if all went well
        if trained_process not in PROCESSES:
            PROCESSES.append(trained_process)

        # Add literature references for process to list
        utils.get_references(*process)


def unload_processes(process_list=None):
    """
    Given a list of sparticle production processes, unload all relevant
    trained GP models from memory, or remove them from a cache folder on
    disk if using cache. This is particularly useful to minimise the
    memory load when making predictions for many different processes.
    If called without argument, all loaded processes are removed.

    Parameters
    ----------
    process_list : list of tuple, optional
        The input argument is a list of 2-tuples
        (process[0], process[1]), where the components are integers
        specifying the process. For example, gluino-gluino production
        corresponds to the tuple (1000021, 1000021).
        If None (default), all loaded processes are removed.
    """
    import xsec.features as features

    if not process_list:
        del PROCESSES[:]
        PROCESS_DICT.clear()
    else:
        for process in process_list:
            # Check if process tuple has right length
            if len(process) != 2:
                warnings.warn(
                    "The entered process tuple ({input}) does not consist of "
                    "exactly _two_ particle IDs and will be ignored.".format(
                        input=process
                    )
                )
                continue
            else:
                # Retrieve trained process and attempt removal
                trained_process = features.get_trained_process(*process)
                try:
                    PROCESSES.remove(trained_process)
                    # Remove each of the xstypes in PROCESS_DICT
                    for xstype in utils.XSTYPES:
                        process_xstype = utils.get_process_id(
                            trained_process, xstype
                        )
                        del PROCESS_DICT[process_xstype]
                except (KeyError, ValueError):
                    warnings.warn(
                        "An entered process {input} was ignored as it is "
                        "not present in the list of loaded processes: {loaded}"
                        "".format(input=process, loaded=PROCESSES)
                    )
                    continue
        # Force Python to collect garbage
        gc.collect()


###############################################
# Finalisation                                #
###############################################


def finalise():
    """
    Function to finalise a run.
    Currently clears cache (if used) and writes references to a file.
    """
    # Clear cache. Inactive if cache not used
    clear_cache()
    # Write references to file (overwrite if existing)
    ref_file = "xsec.bib"
    with open(ref_file, "w") as file_object:
        utils.print_references(file_object)
    print(
        "A list of references that form the basis of the results in\n"
        "this run has been written to {file}.".format(file=ref_file)
    )


def clear_cache():
    """
    Clear cache memory if it was used.
    """
    if USE_CACHE and FLUSH_CACHE:
        # Flush the cache completely
        CACHE_MEMORY.clear(warn=False)
    return 0
