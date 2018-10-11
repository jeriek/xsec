"""
This program evaluates cross-sections for the production of
supersymmetric particles, using Distributed Gaussian Processes trained
by NIMBUS.
"""

# Import packages
from __future__ import print_function
import os
import sys
# import imp
# import warnings
import collections
# from itertools import product

import numpy as np  # Needs v1.14 or later
import joblib       # Needs v0.12.2 or later

# Need to import all setters to allow access upon importing only
# 'evaluation'!
from parameters import PARAMS, MEAN_INDEX, set_parameters, set_parameter, set_mean_mass
from features import get_features, get_features_dict
import kernels

# print('Numpy version ' + np.__version__)
# print('Joblib version ' + joblib.__version__)


###############################################
# Global variables                            #
###############################################

# Specify GP model data directory (can be reset in the run script)
DATA_DIR = './data'

# Link internal cross-section type (xstype) identifiers here to the
# corresponding Nimbus file suffixes for each trained xstype
XSTYPE_FILESUFFIX = {
    'centr': '',  # xsection @ central scale
    'sclup': '_2',  # xsection @ higher scale (2 x central scale)
    'scldn': '_05',  # xsection @ lower scale (0.5 x central scale)
    'pdf': '_pdf',  # xsection error due to pdf variation
    'aup': '_aup',  # xsection @ higher alpha_s
    'adn': '_adn'  # xsection @ lower alpha_s
}
# List of the internal xstype identifiers
XSTYPES = XSTYPE_FILESUFFIX.keys()

# List of selected processes (2-tuples of sparticle ids), to be set by
# the user
# TODO: add set_processes function
PROCESSES = []  # e.g. [(1000021, 1000021), (1000021, 1000002)]

# Definition of sparticle PDG ids
SQUARK_IDS = [1000001, 1000002, 1000003, 1000004,
              2000001, 2000002, 2000003, 2000004]
GLUINO_ID = 1000021

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

    # Set the data directory, if the given string isn't empty
    # TODO: try/except
    if data_dir:
        global DATA_DIR
        DATA_DIR = data_dir

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
            cachedir = os.path.expandvars(CACHE_DIR)
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

    # Can only import Nimbus data transformation code after DATA_DIR is
    # set definitively, need to import from path and add to global scope
    # - Works in Python 2.x:
    # global datatransform
    # datatransform = imp.load_source('', os.path.join(DATA_DIR, 'transform.py'))

    # - Works in any Python version, despite being ugly:
    # Execute the module and add its functions to the global scope
    transform_file = os.path.join(os.path.abspath(DATA_DIR), 'transform.py')
    with open(transform_file) as f:
        transform_code = compile(f.read(), transform_file, 'exec')
        exec(transform_code, globals(), globals())  # globals(), locals())

    # https://stackoverflow.com/questions/67631/how-to-import-a-module-given-the-full-path
    # https://stackoverflow.com/questions/6347588/is-it-possible-to-import-to-the-global-scope-from-inside-a-function-python
    # from data.transform import inverse_transform

    return 0


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
    process_dir = os.path.join(os.path.abspath(DATA_DIR),
                               get_processdir_name(process_xstype))

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
            gp_reco['kernel'] = gp_model['kernel'] # kernel parameters
            # Compute K_inv from L_inv and store it in the dict
            gp_reco['K_inv'] = gp_reco['L_inv'].dot(gp_reco['L_inv'].T)

            model_list.append(gp_reco)

    return model_list


def load_processes(process_list):
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

    if USE_CACHE:
        # Decorate load_single_process() such that its output can be
        # cached using the Joblib Memory object
        load_single_process_cache = CACHE_MEMORY.cache(load_single_process)

    # Loop over specified processes
    for process in process_list:
        assert len(process) == 2
        # Search for all directories with same process, accounting for
        # cross-sections calculated at varied parameters
        for xstype in XSTYPES:
            process_xstype = get_process_id(process, xstype)
            if USE_CACHE:
                # If using cache, PROCESS_DICT only keeps a reference
                # to the data stored in a disk folder ('shelving')
                PROCESS_DICT[process_xstype] = (
                    load_single_process_cache.call_and_shelve(process_xstype))
            else:
                # Loaded GP models are stored in PROCESS_DICT
                PROCESS_DICT[process_xstype] = (
                    load_single_process(process_xstype))


def set_kernel(kernel_params):
    """
    Construct a kernel function from its parameters. In particular, the
    returned functions are functions of X and Y (optional), and have the
    form
        k(X, Y) = WhiteKernel(X, Y, noise_level) +
                  prefactor*MaternKernel(X, Y, length_scale, nu).

    Parameters
    ----------
    kernel_params : dict
        Parameter dictionary with keys 'matern_prefactor',
        'matern_lengthscale', 'matern_nu', and 'whitekernel_noiselevel',
        with corresponding double-precision numerical values.
        This input format corresponds to the output from the NIMBUS
        training routines.

    Returns
    -------
    kernel_function(X, Y=None) : function
        Kernel function that is a linear combination of a white kernel
        and a Matern kernel. If Y = None, kernel_function(X, X) is
        returned.
    """

    # Define a function object to return (requires loading 'kernels'
    # module for kernel definitions)
    def kernel_function(X, Y=None):
        """
        Return the Gaussian Process kernel k(X, Y), a linear combination
        of a white kernel and a Matern kernel. The implementation is
        based on scikit-learn v0.19.2.

        Parameters
        ----------
        X : array, shape (n_samples_X, n_features)
            Left argument of the returned kernel k(X, Y)
        Y : array, shape (n_samples_Y, n_features), (optional,
                                                     default=None)
            Right argument of the returned kernel k(X, Y). If None,
            k(X, X) is evaluated instead.

        Returns
        -------
        K : array, shape (n_samples_X, n_samples_Y)
            Kernel k(X, Y).
        """

        # Extract parameters from input dictionary
        noise_level = kernel_params['whitekernel_noiselevel']
        prefactor = kernel_params['matern_prefactor']
        nu = kernel_params['matern_nu']
        length_scale = kernel_params['matern_lengthscale']

        # Return sum of white kernel and (prefactor times) Matern kernel
        if Y is None:
            kernel_sum = (kernels.WhiteKernel(X, noise_level=noise_level)
                          + prefactor*kernels.MaternKernel(
                              X, length_scale=length_scale, nu=nu))
        else:
            kernel_sum = (kernels.WhiteKernel(X, Y, noise_level=noise_level)
                          + prefactor*kernels.MaternKernel(
                              X, Y, length_scale=length_scale, nu=nu))

        return kernel_sum

    return kernel_function


###############################################
# Helper functions                            #
###############################################

def get_processdir_name(process_xstype):
    # Get the partons of the process
    parton1, parton2, xstype = get_process_id_split(process_xstype)

    # Decide process name
    # Check if one of the partons is a gluino
    if parton1 == GLUINO_ID:
        processdir_name = str(parton1)+'_'+str(parton2)+'_NLO'
    elif parton2 == GLUINO_ID:
        processdir_name = str(parton2)+'_'+str(parton1)+'_NLO'

    # Otherwise name starts with the largest parton PID
    elif abs(parton1) >= abs(parton2):
        processdir_name = str(parton1)+'_'+str(parton2)+'_NLO'
    elif abs(parton1) < abs(parton2):
        processdir_name = str(parton2)+'_'+str(parton1)+'_NLO'

    # Add scale like '05','2' or other variation parameter like 'aup', 'adn'
    try:
        processdir_name += XSTYPE_FILESUFFIX[xstype]
    except KeyError:
        print('Error: ', xstype, ' is not a valid variation parameter!')

    return processdir_name


def get_process_id(process, xstype):
    assert len(process) == 2
    process_xstype = (process[0], process[1], xstype)

    return process_xstype


def get_process_id_str(process, xstype):
    assert len(process) == 2
    process_xstype_str = (str(process[0]) + '_' + str(process[1]) + '_'
                           + xstype)

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


def get_process_list_str(process_list):
    process_str_list = []
    for process in process_list:
        process_str = str(process[0]) + '_' + str(process[1])
        process_str_list.append(process_str)

    return process_str_list


# Function to test consistency of parameters.
# This will check that all necessary parameters are set and that they are
# internally consistent, e.g. the mean mass.
def check_parameters(process_list, params):
    # Check that the parameters for all required feature have been supplied
    for process in process_list:
        features = get_features(process[0],process[1])
        for feature in features:
            if params[feature] == None:
                raise ValueError('The feature {feature} used in this cross section evaulation has not been set!'.format(feature=feature))
    # Check that the mean squark mass has been set consistently
    mean = 0
    nsquark = 0
    for key in MEAN_INDEX:
        if params[key] != None:
            mean += params[key]
            nsquark += 1
    mean = mean/8.
    if nsquark == 8 and abs(params['mean'] - mean) > 0.1:
        raise ValueError('The squark masses mean {mean1} is not equal to the mean mass feature used {mean2}!'.format(mean1=mean,mean2=params['mean']))


###############################################
# Main functions                              #
###############################################

# Evaluation of cross sections
def eval_xsection(verbose=True, check_consistency=True):

    """
    Evaluates cross sections for processes in global list PROCESSES using
    parameter values stored in global dictionary PARAMS.
    
    The function has two options:
    verbose:    Turns on and off printing of values to terminal
    check_consistency:  Forces a consistency check of the paramters in PARAMS
    """

    ##################################################
    # Build feature vector                           #
    ##################################################

    # Get local variable to avoid multiple slow lookups in global namespace
    processes = PROCESSES

    for process in processes:
        assert len(process) == 2

    params = PARAMS

    # Sanity check parameter inputs
    if check_consistency:
        check_parameters(processes, params)

    # Build feature vectors, depending on production channel
    features = get_features_dict(processes)

    ###################################################
    # Do DGP evaluation                               #
    ###################################################

    # Call a DGP for each process_xstype, store results as lists of
    # (mu_dgp, sigma_dgp) in dictionary with key xstype; i-th element of
    # list dgp_results[xstype] gives DGP result for
    # process = processes[i] and the specified xstype.
    # Immediately corrected for any data transformation during training
    #  with Nimbus.

    # Dictionary of PROCESSES-ordered lists
    dgp_results = {
        xstype: [
            inverse_transform(process, xstype, params,
                              *DGP(process, xstype, features[process]))
            for process in processes
            ]
        for xstype in XSTYPES
        }

    # All returned errors are defined to be deviations from 1

    # -- Central-scale xsection and regression error (= standard
    #    deviation) in fb.
    xsection_central, reg_err = map(
        np.array, zip(*(mu_sigma_dgp for mu_sigma_dgp in dgp_results['centr']))
        )
    # xsection_central, reg_err = map(
    #     np.array, zip(*(moments_lognormal(*mu_sigma_dgp)
    #                     for mu_sigma_dgp in dgp_results['centr']))
    #     )
    # (zip() splits list of (mu,sigma) tuples into two tuples, one for
    # mu and one for sigma values -- then convert to arrays by mapping)
    # NOTE: Result arrays are now ordered in the user-specified order
    # from the global PROCESSES variable!

    # -- Xsection deviating one (lognormal) regression error away
    #    from the central-scale xsection, relative to the latter.
    regdown_rel = 1. - reg_err/xsection_central # numpy array
    regup_rel = 1. + reg_err/xsection_central  # numpy array

    # -- Xsection at lower and higher scale (0.5x and 2x central scale),
    #    relative to the central-scale xsection. To prevent that the
    #    unusual case with xsection_scaleup > xsection_scaledown causes
    #    errors, min/max ensures scaledown_rel always gives the lower
    #    bound and scaleup_rel the higher one.
    #    NOTE: This means scaledown_rel generally doesn't correspond to
    #    the xsection value at the lower scale, but at the higher one,
    #    and vice versa for scaleup_rel.
    # Get the DGP means, discard regression errors on the variations
    mu_dgp_scldn, _ = np.array(zip(*dgp_results['scldn']))
    mu_dgp_sclup, _ = np.array(zip(*dgp_results['sclup']))

    scaledown_rel = np.array(map(np.min, zip(mu_dgp_scldn, mu_dgp_sclup)))
    scaleup_rel = np.array(map(np.max, zip(mu_dgp_scldn, mu_dgp_sclup)))

    # -- Xsection deviating one pdf error away from the
    #    central-scale xsection, relative to the latter.
    # Get the DGP means, discard regression errors on the variations
    delta_pdf_rel, _ = np.array(zip(*dgp_results['pdf']))

    pdfdown_rel = 1. - delta_pdf_rel
    pdfup_rel = 1. + delta_pdf_rel

    # -- Xsection deviating one symmetrised alpha_s error away from
    #    the central-scale xsection, relative to the latter.
    # Get the DGP means, discard regression errors on the variations
    mu_dgp_adn, _ = np.array(zip(*dgp_results['scldn']))
    mu_dgp_aup, _ = np.array(zip(*dgp_results['sclup']))

    delta_alphas_rel = np.array([0.5*(abs(aup-1.) + abs(1.-adn))
                                for (aup, adn) in zip(mu_dgp_aup, mu_dgp_adn)])

    alphasdown_rel = 1. - delta_alphas_rel
    alphasup_rel = 1. + delta_alphas_rel

    # Collect values for output in Numpy array
    return_array = np.array([
        xsection_central, regdown_rel, regup_rel,
        scaledown_rel, scaleup_rel, pdfdown_rel,
        pdfup_rel, alphasdown_rel, alphasup_rel
        ])
    # print(return_array)

    if verbose:
        print(
            "\t    _/      _/    _/_/_/  _/_/_/_/    _/_/_/   \n"
            "\t     _/  _/    _/        _/        _/          \n"
            "\t      _/        _/_/    _/_/_/    _/           \n"
            "\t   _/  _/          _/  _/        _/            \n"
            "\t_/      _/  _/_/_/    _/_/_/_/    _/_/_/       \n"
        )
        nr_dec = 4
        np.set_printoptions(precision=nr_dec)
        print("* Processes requested, in order: \n  ",
            *get_process_list_str(processes))
        for process in processes:
            print("* Input features: \n  ", process, ": ",
                  zip(get_features(*process),
                  get_features_dict(PROCESSES)[process]))
        print("* xsection_central (fb):", xsection_central)
        print("* regdown_rel:", regdown_rel)
        print("* regup_rel:", regup_rel)
        print("* scaledown_rel:", scaledown_rel)
        print("* scaleup_rel:", scaleup_rel)
        print("* pdfdown_rel:", pdfdown_rel)
        print("* pdfup_rel:", pdfup_rel)
        print("* alphasdown_rel:", alphasdown_rel)
        print("* alphasup_rel:", alphasup_rel)
        print("**************************************************************")

    # NOTE: plot just for comparison during testing
    # plot_lognormal(mu_dgp, sigma_dgp)

    # print return_array
    return return_array


def DGP(process, xstype, features):
    """
        Evaluate a set of distributed Gaussian processes (DGPs)
    """
    assert len(process) == 2
    process_xstype = (process[0], process[1], xstype)

    # Decide which GP to use depending on 'scale'
    processdir_name = get_processdir_name(process_xstype)
    # print processdir_name

    # List all trained experts in the chosen directory

    models = os.listdir(os.path.join(DATA_DIR, processdir_name))
    n_experts = len(models)

    # print 'Models:', models

    # Empty arrays where all predicted numbers are stored
    mus = np.zeros(n_experts)
    sigmas = np.zeros(n_experts)
    sigma_priors = np.zeros(n_experts)

    # Loop over GP models/experts
    for i in range(len(models)):
        # model = models[i]
        mu, sigma, sigma_prior = GP_predict(process_xstype, features,
                                            index=i, return_std=True)
        mus[i] = mu
        sigmas[i] = sigma
        sigma_priors[i] = sigma_prior
        # print "-- Resulting mu, sigma, sigma_prior:", mu, sigma, sigma_prior

    ########################################
    # Assume here that mus, sigmas and
    # sigma_priors are full arrays

    N = len(mus)

    # Find weight (beta) for each expert
    betas = 0.5*(  2*np.log(sigma_priors) - 2*np.log(sigmas) )

    # Final mean and variance
    mu_dgp = 0.
    var_dgp_inv = 0. # (sigma^2)^-1

    # Combine sigmas
    for i in range(N):
        var_dgp_inv += (betas[i] * sigmas[i]**(-2)
                        + (1./n_experts - betas[i])
                        * sigma_priors[i]**(-2))

    # Combine mus
    for i in range(N):
        mu_dgp +=  var_dgp_inv**(-1) * (betas[i] * sigmas[i]**(-2) * mus[i])


    # Return mean and std
    return mu_dgp, np.sqrt(var_dgp_inv**(-1))



def GP_predict(process_xstype, features, index=0, return_std=True, return_cov=False):
    """
    Gaussian process evaluation for the individual experts. Takes as
    input arguments the produced partons, an array of new test features,
    and the index number of the expert. Requires running
    load_processes() first.

    Returns a list of numpy arrays containing the mean value (the
    predicted cross-section), the GP standard deviation (or full
    covariance matrix), and the square root of the prior variance on the
    test features.

    Based on GaussianProcessRegressor.predict() from scikit-learn
    v0.19.2 and algorithm 2.1 of Gaussian Processes for Machine Learning
    by Rasmussen and Williams.
 
    """

    if return_std and return_cov:
        raise RuntimeError("Cannot return both standard deviation "
                           "and full covariance.")

    try:
        if USE_CACHE:
            # Get list of loaded models for the specified process
            gp_model = PROCESS_DICT[process_xstype].get()[index]
        else:
            gp_model = PROCESS_DICT[process_xstype][index]

        kernel = gp_model['kernel']
        X_train = gp_model['X_train']
        alpha = gp_model['alpha']
        L_inv = gp_model['L_inv']
        K_inv = gp_model['K_inv']
        kernel = set_kernel(gp_model['kernel'])

    except KeyError, e:
        print(KeyError, e)
        print("No trained GP models loaded for: " + str(process_xstype))
        return None

    X = np.atleast_2d(features)

    K_trans = kernel(X, X_train) # transpose of K*
    y_mean = K_trans.dot(alpha) # Line 4 (y_mean = f_star)

    prior_variance = kernel(X) # Note: 1x1 if just 1 new test point!]

    if return_std:
        # Compute variance of predictive distribution Note: =
        # prior_variance if 1x1, deep copy else prior_variance = y_var
        # alias
        y_var = np.diag(prior_variance.copy())
        y_var.setflags(write=True)  # else this array is read-only
        y_var -= np.einsum("ij,ij->i", np.dot(K_trans, K_inv),
                           K_trans, optimize=True)

        # Check if any of the variances is negative because of numerical
        # issues. If yes: set the variance to absolute value, to keep
        # the rough order of magnitude right.
        y_var_negative = y_var < 0
        if np.any(y_var_negative):
            # warnings.warn("Predicted some variance(s) smaller than 0. "
                        #   "Approximating these with their absolute value.")
            y_var[y_var_negative] = np.abs(y_var[y_var_negative])
        y_std = np.sqrt(y_var)
        prior_std = np.sqrt(prior_variance.flatten())
        return y_mean, y_std, prior_std

    elif return_cov:
        v = L_inv.dot(K_trans.T) # Line 5
        y_cov = prior_variance - K_trans.dot(v)  # Line 6
        return [y_mean, y_cov, prior_variance]

    else:
        return y_mean


def clear_cache():
    if USE_CACHE and FLUSH_CACHE:
        # Flush the cache completely
        CACHE_MEMORY.clear(warn=False)
    return 0
