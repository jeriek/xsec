"""
Initialisation and GP loading functions.
"""

from __future__ import print_function

import os
import joblib       # Needs v0.12.2 or later

import utils
import parameters

###############################################
# Global variables                            #
###############################################

# Reset the GP model data directory. By default, DATA_DIR gets set to
# the /data directory inside the xsec installation folder when init() is
# executed. This can be overwritten by manually specifying DATA_DIR in
# the run script, either through accessing the DATA_DIR global variable
# explicitly, or through specifying the data_dir keyword inside init().
# The latter takes precedence if both methods are used simultaneously.
DATA_DIR = ''

# List of selected processes (2-tuples of sparticle ids), to be set by
# the user
PROCESSES = []  # e.g. [(1000021, 1000021), (1000021, 1000002)]

# For each selected process, store trained GP model dictionaries here
# (or a list of their cache locations):
PROCESS_DICT = {}

# Initialise default settings for using Joblib memory caching, can be
# modified by user with init()
USE_CACHE = False
CACHE_DIR = ''
FLUSH_CACHE = True
USE_MEMMAP = True
CACHE_MEMORY = None


###############################################
# Initialisation                              #
###############################################


def init(data_dir='', use_cache=False, cache_dir='', flush_cache=True,
         use_memmap=True):
    """
    Initialise run settings for the program. In particular, whether to
    use a cache, i.e. a temporary disk directory to store loaded GP
    models for use in predictive functions. This could be useful if the
    memory load needs to be decreased, when loading many large models.

    Parameters
    ----------
    data_dir : str, optional
        Specify the path of the top directory containing all of the GP
        model data directories. (default '')
    use_cache : bool, optional
        Specify whether to cache data on disk (default False).
    cache_dir: str, optional
        Specify a disk directory for the cache. Inactive if use_cache is
        False. If '', a random directory is created. (default '')
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

    # Reset the data directory, if the given string isn't empty
    # TODO: try/except
    global DATA_DIR
    if data_dir:
        DATA_DIR = os.path.expandvars(os.path.expanduser(data_dir))

    # If, as default, DATA_DIR was not set manually before init(), nor
    # with the data_dir keyword inside init(), then the data directory
    # is the default /data folder within the xsec installation
    # directory. Fix DATA_DIR to that default location now and import
    # the inverse_transform function. Else, DATA_DIR is already fixed,
    # just need to import inverse_transform() now.
    # TODO: try/except
    # TODO: all of this will change with the new transform.py structure!
    if not DATA_DIR:
        global inverse_transform
        xsec_dir = os.path.dirname(os.path.realpath(__file__))
        try:
            from gprocs.transform import inverse_transform
            DATA_DIR = os.path.join(xsec_dir, 'gprocs')
        except ImportError:
            raise ImportError(
                'Please check that the /gprocs directory is located inside '
                '{dir}, and that it contains the file transform.py'.format(
                    dir=xsec_dir
                )
            )
    else:
        # Execute the transform module and add its functions to the
        # global scope
        transform_file = os.path.join(
            os.path.abspath(DATA_DIR), 'transform.py')
        with open(transform_file) as f:
            transform_code = compile(f.read(), transform_file, 'exec')
            exec(transform_code, globals(), globals())  # globals(), locals())

    # Fix global variables coordinating the use of caching
    global USE_CACHE, CACHE_DIR, FLUSH_CACHE, USE_MEMMAP
    USE_CACHE = use_cache
    CACHE_DIR = cache_dir
    FLUSH_CACHE = flush_cache
    USE_MEMMAP = use_memmap

    if USE_CACHE:
        if CACHE_DIR:
            # Set cache directory to given name, expand any environment
            # variables in the name
            cachedir = os.path.expandvars(os.path.expanduser(CACHE_DIR))
        else:
            # Create directory with random name
            from tempfile import mkdtemp
            cachedir = mkdtemp(prefix='xsec_')
        if USE_MEMMAP:
            # Set memmap mode 'copy on write'
            memmap_mode = 'c'
        else:
            # Disable memmapping
            memmap_mode = None

        # Create a Joblib Memory object managing the cache
        global CACHE_MEMORY
        CACHE_MEMORY = joblib.Memory(location=cachedir,
                                     mmap_mode=memmap_mode,
                                     verbose=0)
        print("Cache folder: "+str(cachedir))

    return 0


def set_processes(tuple_list):
    """
    Set the list of processes to be evaluated, and load the GP models
    for those processes.
    """
    # Check if process exists (right format, known sparticles)
    for process in tuple_list:
        if len(process) == 2:
            if all((pid in parameters.SPARTICLE_IDS) for pid in process):
                continue
            else:
                raise ValueError(
                    "One or more particle IDs are not in the allowed set of "
                    "IDs: \n", parameters.SPARTICLE_IDS
                )
        else:
            raise ValueError(
                "Each entered process tuple should consist of exactly _two_ "
                "particle IDs from the following list: \n",
                parameters.SPARTICLE_IDS
            )
    # Only set PROCESSES and load the GPs if all checks were passed
    global PROCESSES
    PROCESSES = tuple_list
    load_processes(PROCESSES)


def get_processes():
    """
    Return the current list of processes to be evaluated.
    """
    return PROCESSES


###############################################
# Loading functions                           #
###############################################


def load_single_process(process_xstype):
    """
    Given a single process and cross-section type (e.g. gluino-gluino at
    the central scale), load the relevant trained GP models for all
    experts and return them in a list of dictionaries, one per expert.

    Parameters
    ----------
    process_xstype : tuple of str
        The input argument process_xstype is a 3-tuple
        (process[0], process[1], var) where the first two components
        are integers specifying the process and the last component is a
        string from XSTYPES.
        Example: (1000021, 1000021, 'centr')

    Returns
    -------
    model_list : list of dict
        List containing one dictionary per expert trained on the
        process and cross-section type specified in process_xstype.
        Each dictionary has keys 'X_train', 'K_inv', 'alpha' and
        'kernel'. These components are all the information needed to
        make the predictions of the expert with GP_predict().
    """

    assert len(process_xstype) == 3

    # Construct location of GP models for the specified process and
    # cross-section type, using global data directory variable DATA_DIR
    # process_dir = os.path.join(os.path.abspath(DATA_DIR),
    #                            get_processdir_name(process_xstype))
    process_dir = os.path.join(
        DATA_DIR, utils.get_processdir_name(process_xstype))

    # Collect the GP model file locations for all the experts
    model_files = [os.path.join(process_dir, f) for f in
                   os.listdir(process_dir)]
    #if os.path.isfile(os.path.join(process_dir, f))]

    # Initialise list of GP model dicts
    model_list = []

    # Loop over experts and append one GP model dict per expert to
    # model_list
    for model_file in model_files:
        # Open the stored GP model file of a single expert
        with open(model_file, "rb") as fo:
            # Unzip the binary file with Joblib, yielding dict
            gp_model = joblib.load(fo)
            # Reconvert float32 arrays to float64 for higher-precision
            # computations, filling a new dict gp_reco
            gp_reco = {}
            gp_reco['X_train'] = gp_model['X_train'].astype('float64')
            gp_reco['L_inv'] = gp_model['L_inv'].astype('float64')
            gp_reco['alpha'] = gp_model['alpha'].astype('float64')
            gp_reco['kernel'] = gp_model['kernel']  # kernel parameters
            # Compute K_inv from L_inv and store it in the dict
            gp_reco['K_inv'] = gp_reco['L_inv'].dot(gp_reco['L_inv'].T)

            model_list.append(gp_reco)

    return model_list


def load_processes(process_list=PROCESSES):
    """
    Given a list of sparticle production processes, load all relevant
    trained GP models into memory, or into a cache folder on disk if
    using cache. The function calls load_single_process() for each
    process in process_list, looping over all cross-section types in
    XSTYPES. It stores each returned list of models in the global
    dictionary PROCESS_DICT, indexed with key 'process_xstype'. If
    using cache, a reference to the location of the cached data is
    stored in PROCESS_DICT, indexed in the same way.

    Parameters
    ----------
    process_list : list of tuple
        The input argument is a list of 2-tuples
        (process[0], process[1]), where the components are integers
        specifying the process. For example, gluino-gluino production
        corresponds to the tuple (1000021, 1000021).
    """
    # TODO: Remove process_list argument, just use PROCESSES (and add
    # set_processes() function)
    if USE_CACHE:
        # Decorate load_single_process() such that its output can be
        # cached using the Joblib Memory object
        load_single_process_cache = CACHE_MEMORY.cache(load_single_process)

    # Loop over specified processes
    for process in process_list:
        assert len(process) == 2
        # Search for all directories with same process, accounting for
        # cross-sections calculated at varied parameters
        for xstype in utils.XSTYPES:
            process_xstype = utils.get_process_id(process, xstype)
            if USE_CACHE:
                # If using cache, PROCESS_DICT only keeps a reference
                # to the data stored in a disk folder ('shelving')
                PROCESS_DICT[process_xstype] = (
                    load_single_process_cache.call_and_shelve(process_xstype))
            else:
                # Loaded GP models are stored in PROCESS_DICT
                PROCESS_DICT[process_xstype] = (
                    load_single_process(process_xstype))


def clear_cache():
    if USE_CACHE and FLUSH_CACHE:
        # Flush the cache completely
        CACHE_MEMORY.clear(warn=False)
    return 0