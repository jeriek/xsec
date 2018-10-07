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

from parameters import param, set_parameters, mean_mass
from features import get_features
import kernels

import numpy as np # v1.14 or later
import joblib  # v0.12.2 or later

# from data.transform import inverse_transform
# NOTE: currently no relative import, meaning DATA_DIR should stay
# ./data, else transform.py won't be found (if not ./data, would need to
# append to PYTHONPATH even?)

# print('Numpy version ' + np.__version__)
# print('Joblib version ' + joblib.__version__)


###############################################
# Global variables                            #
###############################################

# Specify GP model data directory (can be reset in the run script)
DATA_DIR = './data'

# Specify available variation parameters (NOTE: to be changed!)
# XSTYPES = ['', '05', '2', '3', '4', '5']

# Link internal cross-section type (xstype) identifiers here to the
# corresponding Nimbus file suffixes for each trained xstype
# TODO: this should replace part of get_processdir_name()
XSTYPE_FILESUFFIX = {
    'centr' : '', # xsection @ central scale
    'sclup' : '_2', # xsection @ higher scale (2 x central scale)
    'scldn' : '_05', # xsection @ lower scale (0.5 x central scale)
    'pdf' : '_pdf', # xsection error due to pdf variation
    'aup' : '_aup', # xsection @ higher alpha_s
    'adn' : '_adn' # xsection @ lower alpha_s
    }
# List of the internal xstype identifiers
XSTYPES = XSTYPE_FILESUFFIX.keys()

# List of selected processes (2-tuples of sparticle ids), to be set by
# the user
# TODO: should be XSECTIONS for consistency!
XSECTIONS = [] # e.g. [(1000021, 1000021), (1000021, 1000002)]

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

def init(use_cache=False, cache_dir='', flush_cache=True,
         use_memmap=True):
    """
    Initialise run settings for the program. In particular, whether to
    use a cache, i.e. a temporary disk directory to store loaded GP
    models for use in predictive functions. This could be useful if the
    memory load needs to be decreased, when loading many large models.

    Parameters
    ----------
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
        exec(transform_code, globals(), globals()) # globals(), locals())

    # https://stackoverflow.com/questions/67631/how-to-import-a-module-given-the-full-path
    # https://stackoverflow.com/questions/6347588/is-it-possible-to-import-to-the-global-scope-from-inside-a-function-python
    # from data.transform import inverse_transform

    return 0


###############################################
# Loading functions                           #
###############################################

def load_single_process(xsection_xstype):
    """
    Given a single process and cross-section type (e.g. gluino-gluino at
    the central scale), load the relevant trained GP models for all
    experts and return them in a list of dictionaries, one per expert.

    Parameters
    ----------
    xsection_xstype : tuple of str
        The input argument xsection_xstype is a 3-tuple
        (xsection[0], xsection[1], var) where the first two components
        are integers specifying the process and the last component is a
        string from XSTYPES.
        Example: (1000021, 1000021, 'centr')

    Returns
    -------
    model_list : list of dict
        List containing one dictionary per expert trained on the
        process and cross-section type specified in xsection_xstype.
        Each dictionary has keys 'X_train', 'K_inv', 'alpha' and
        'kernel'. These components are all the information needed to
        make the predictions of the expert with GP_predict().
    """

    assert len(xsection_xstype) == 3

    # Construct location of GP models for the specified process and
    # cross-section type, using global data directory variable DATA_DIR
    process_dir = os.path.join(os.path.abspath(DATA_DIR), 
                               get_processdir_name(xsection_xstype))

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


def load_processes(xsections_list):
    """
    Given a list of sparticle production processes, load all relevant
    trained GP models into memory, or into a cache folder on disk if
    using cache. The function calls load_single_process() for each
    process in xsections_list, looping over all cross-section types in
    XSTYPES. It stores each returned list of models in the global
    dictionary PROCESS_DICT, indexed with key 'xsection_xstype'. If
    using cache, a reference to the location of the cached data is
    stored in PROCESS_DICT, indexed in the same way.

    Parameters
    ----------
    xsections_list : list of tuple
        The input argument is a list of 2-tuples
        (xsection[0], xsection[1]), where the components are integers
        specifying the process. For example, gluino-gluino production
        corresponds to the tuple (1000021, 1000021).
    """

    if USE_CACHE:
        # Decorate load_single_process() such that its output can be
        # cached using the Joblib Memory object
        load_single_process_cache = CACHE_MEMORY.cache(load_single_process)

    # Loop over specified processes
    for xsection in xsections_list:
        assert len(xsection) == 2
        # Search for all directories with same process, accounting for
        # cross-sections calculated at varied parameters
        for xstype in XSTYPES:
            xsection_xstype = get_process_id(xsection, xstype)
            if USE_CACHE:
                # If using cache, PROCESS_DICT only keeps a reference
                # to the data stored in a disk folder ('shelving')
                PROCESS_DICT[xsection_xstype] = \
                    load_single_process_cache.call_and_shelve(xsection_xstype)
            else:
                # Loaded GP models are stored in PROCESS_DICT
                PROCESS_DICT[xsection_xstype] = \
                    load_single_process(xsection_xstype)


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

# Produce a dictionary of features and their masses for each process in xsections_list
def get_feature_dict(xsections_list):

    all_features = range(len(xsections_list)) # initialise dummy list
    # As dict
    all_features_dict = {}

    # Find mean squark mass. TODO: Move out of here to separate function.
    mean_index = ['m1000004', 'm1000003', 'm1000001', 'm1000002',
                  'm2000002', 'm2000001', 'm2000003', 'm2000004']
    mean_mass = sum([param[key] for key in mean_index])/float(len(mean_index))
    param['mean'] = mean_mass
    #mean = mean_mass()
    
    # xsections_list has a list of proxesses we want features for so loop
    for i in range(len(xsections_list)):

        # Extract current process
        xsection = xsections_list[i]

        # Find features for this process
        features_index = get_features(xsection[0],xsection[1])

        # Make a feature dictionary
        # features_dict = {key : masses[key] for key in features_index}
        features_dict = collections.OrderedDict()
        for key in features_index:
            features_dict[key] = param[key]

        features = features_dict.values()

        # Add features to feature array
        all_features[i] = features
        all_features_dict.update({xsection : features})

    # Return feature array for all processes
    #return np.asarray(all_features)
    return all_features_dict


def get_processdir_name(xsection_xstype):
    # Get the partons of the process
    parton1, parton2, xstype = get_process_id_split(xsection_xstype)

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


def get_process_id(xsection, xstype):
    assert len(xsection) == 2
    xsection_xstype = (xsection[0], xsection[1], xstype)

    return xsection_xstype


def get_process_id_str(xsection, xstype):
    assert len(xsection) == 2
    xsection_xstype_str = (str(xsection[0]) + '_' + str(xsection[1]) + '_'
                           + xstype)

    return xsection_xstype_str


def get_process_id_split(xsection_xstype):
    assert len(xsection_xstype) == 3
    parton1 = xsection_xstype[0]
    parton2 = xsection_xstype[1]
    xstype = xsection_xstype[2]

    return parton1, parton2, xstype


def get_xsection_from_process_id(xsection_xstype):
    parton1 = xsection_xstype[0]
    parton2 = xsection_xstype[1]

    return (parton1, parton2)


def get_xstype_from_process_id(xsection_xstype):
    xstype = xsection_xstype[-1]

    return xstype


def get_xsections_list_str(xsections_list):
    xsection_str_list = []
    for xsection in xsections_list:
        xsection_str = str(xsection[0]) + '_' + str(xsection[1])
        xsection_str_list.append(xsection_str)

    return xsection_str_list


###############################################
# Main functions                              #
###############################################

# Evaluation of cross sections for processes stored in global variable XSECTIONS
# TODO: Remove dependence on masses to global parameter object
def eval_xsection(m1000021, m2000006=None, m2000005=None, m2000004=None, m2000003=None,
                  m2000002=None, m2000001=None, m1000006=None, m1000005=None, m1000004=None,
                  m1000003=None, m1000002=None, m1000001=None):

    """
    Read masses and parameters from slha-file and evaluate
    cross sections
    """

    ##################################################
    # Check masses                                   #
    ##################################################

    if m2000003 is not None:

        # If more than two two first masses are provided, then
        # all squark masses must be provided.

        try:
            m1000001+m1000002+m2000002+m2000001+m2000003+m2000004 # If you try to add a number and a None, you get False
        except TypeError:
            print('Error! Masses must either be given as two masses (mg, mq), \n \
                or as all nine masses (mg, mcR, msR, muR, mdR, mcL, msL, muL, mdL).')
            sys.exit()

    else:

        # If only the two first masses are given the squark masses
        # are degenerate, and set to m2000004
        m1000004 = m2000004
        m1000003 = m2000004
        m1000002 = m2000004
        m1000001 = m2000004
        m2000003 = m2000004
        m2000002 = m2000004
        m2000001 = m2000004


    # Put masses in dictionary
    masses = {'m1000021' : m1000021, 'm1000006' : m1000006,
              'm1000005' : m1000005, 'm1000004' : m1000004,
              'm1000003' : m1000003, 'm1000001' : m1000001,
              'm1000002' : m1000002, 'm2000002' : m2000002,
              'm2000001' : m2000001, 'm2000003' : m2000003,
              'm2000004' : m2000004, 'm2000005' : m2000005,
              'm2000006' : m2000006, 'mean' : 0}
    # Store in param dictionary
    set_parameters(masses)


    ##################################################
    # Build feature vector                           #
    ##################################################

    # Get local variable to avoid multiple slow lookups in global namespace
    xsections = XSECTIONS 

    for xsection in xsections:
        assert len(xsection) == 2
        # print 'The production type of ', xsection, 'is ', types[xsection]

    # Build feature vectors, depending on production channel
    features = get_feature_dict(xsections)
    # print 'The features are ', features


    ###################################################
    # Do DGP regression                               #
    ###################################################

    # Call a DGP for each xsection_xstype, store results as lists of
    # (mu_dgp, sigma_dgp) in dictionary with key xstype; i-th element of
    # list dgp_results[xstype] gives DGP result for
    # xsection = xsections[i] and the specified xstype.
    # Immediately corrected for any data transformation during training
    #  with Nimbus.

    # Dictionary of xsections-ordered lists
    dgp_results = {
        xstype: [
            inverse_transform(DGP(xsection, xstype, features[xsection]),
                              (xsection, xstype), m1000021=m1000021,
                              m2000004=m2000004, m2000003=m2000003,
                              m2000002=m2000002, m2000001=m2000001,
                              m1000004=m1000004, m1000003=m1000003,
                              m1000002=m1000002, m1000001=m1000001)
            for xsection in xsections
            ]
        for xstype in XSTYPES
        }

    # All returned errors are defined to be deviations from 1

    # -- Central-scale xsection and regression error (= standard
    #    deviation) in fb.
    xsection_central, reg_err = map(
        np.array, zip(*(moments_lognormal(*mu_sigma_dgp)
                        for mu_sigma_dgp in dgp_results['centr']))
        )
    # (zip() splits list of (mu,sigma) tuples into two tuples, one for
    # mu and one for sigma values -- then convert to arrays by mapping)
    # NOTE: Result arrays are now ordered in the user-specified order
    # from the global xsections variable!

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

    print("************** NEW XSEC OUTPUT FORMAT ******************")
    nr_dec = 4
    np.set_printoptions(precision=nr_dec)
    print("* Processes requested, in order: \n", "  ",
          *get_xsections_list_str(xsections))
    print("* xsection_central (fb):", xsection_central)
    print("* regdown_rel:", regdown_rel)
    print("* regup_rel:", regup_rel)
    print("* scaledown_rel:", scaledown_rel)
    print("* scaleup_rel:", scaleup_rel)
    print("* pdfdown_rel:", pdfdown_rel)
    print("* pdfup_rel:", pdfup_rel)
    print("* alphasdown_rel:", alphasdown_rel)
    print("* alphasup_rel:", alphasup_rel)
    print("********************************************************")

    # NOTE: plot just for comparison during testing
    # plot_lognormal(mu_dgp, sigma_dgp)

    # print return_array
    return return_array


def DGP(xsection, xstype, features):

    assert len(xsection) == 2
    xsection_xstype = (xsection[0], xsection[1], xstype)

    # Decide which GP to use depending on 'scale'
    processdir_name = get_processdir_name(xsection_xstype)
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
        mu, sigma, sigma_prior = GP_predict(xsection_xstype, features,
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



def GP_predict(xsection_xstype, features, index=0, return_std=True, return_cov=False):
    """
    Gaussian process regression for the individual experts. Takes as
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
            gp_model = PROCESS_DICT[xsection_xstype].get()[index]
        else:
            gp_model = PROCESS_DICT[xsection_xstype][index]

        kernel = gp_model['kernel']
        X_train = gp_model['X_train']
        alpha = gp_model['alpha']
        L_inv = gp_model['L_inv']
        K_inv = gp_model['K_inv']
        kernel = set_kernel(gp_model['kernel'])

        # if xsection_xstype[2] == '':
        #     print("- Do GP regression for: " + get_processdir_name(xsection_xstype) + " at scale 1.0")
        # else:
        #     print ("- Do GP regression for: " + get_processdir_name(xsection_xstype) + " with variation parameter " + xsection_xstype[2])
    except KeyError, e:
        print(KeyError, e)
        print("No trained GP models loaded for: " + str(xsection_xstype))
        return None

    X = np.atleast_2d(features)

    K_trans = kernel(X, X_train) # transpose of K*
    y_mean = K_trans.dot(alpha) # Line 4 (y_mean = f_star)

    prior_variance = kernel(X) # Note: 1x1 if just 1 new test point!]

    if return_std:
        # Compute variance of predictive distribution
        y_var = np.diag(prior_variance.copy()) # Note: = prior_variance if 1x1, deep copy else prior_variance = y_var alias
        y_var.setflags(write=True) # else this array is read-only
        y_var -= np.einsum("ij,ij->i", np.dot(K_trans, K_inv), K_trans, optimize=True)

        # Check if any of the variances is negative because of
        # numerical issues. If yes: set the variance to 0.
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

# ----------------------------------------------------------------------
# -------- NOTE: ONLY FOR TESTING, TO BE REMOVED LATER -----------------

def moments_lognormal(mu_DGP, sigma_DGP):
    """
    Given the output mean and std of a DGP trained on log10(x), return
    the expectation value and standard deviation of x.
    """
    # Parameters of the lognormal pdf
    mu_lognorm = mu_DGP*np.log(10.)
    sigma_lognorm = sigma_DGP*np.log(10.)
    # Moments of the lognormal pdf
    mean_lognorm = np.exp(mu_lognorm + 0.5*sigma_lognorm**2)
    std_lognorm = mean_lognorm * np.sqrt(np.exp(sigma_lognorm**2) - 1)

    # NOTE: The 'skewness' ratio is only printed for test purposes!
    # print("~DEBUG OUTPUT: ", "std_lognorm/mean_lognorm = ",
        # round(std_lognorm/mean_lognorm, 6))

    return mean_lognorm, std_lognorm

def plot_lognormal(mu_DGP, sigma_DGP):
    # NOTE: Just for testing, not for release.

    from scipy.stats import lognorm, norm
    import matplotlib.pyplot as plt

    mu_lognorm = mu_DGP*np.log(10.)
    sigma_lognorm = sigma_DGP*np.log(10.)

    fig, (ax1, ax2) = plt.subplots(1, 2)
    mean, var, _, _  = lognorm.stats(
        s=sigma_lognorm, scale=np.exp(mu_lognorm), moments='mvsk')
    startx = lognorm.ppf(0.01, s=sigma_lognorm, scale=np.exp(mu_lognorm))
    endx = lognorm.ppf(0.99, s=sigma_lognorm, scale=np.exp(mu_lognorm))
    x = np.linspace(startx, endx, 1000)
    y = lognorm.pdf(x, s=sigma_lognorm, scale=np.exp(mu_lognorm))
    ax1.plot(x, y, 'r-', lw=4, alpha=0.7, color='k', label='lognormal pdf')

    ax1.axvline(x=np.exp(mu_lognorm-sigma_lognorm**2), color='r', label='mode')
    ax1.axvline(x=np.exp(mu_lognorm), color='g', label='median')
    ax1.axvline(x=np.exp(mu_lognorm+0.5*sigma_lognorm**2), color='b', label='mean')

    # lowbound = ax1.axvline(x=np.exp(mu_lognorm+0.5*sigma_lognorm**2)-np.sqrt(var), color='m', label='mean - std')
    # upbound = ax1.axvline(x=np.exp(mu_lognorm+0.5*sigma_lognorm**2)+np.sqrt(var), color='m', label='mean + std')
    lowbound = np.exp(mu_lognorm+0.5*sigma_lognorm**2) - np.sqrt(var)
    upbound = np.exp(mu_lognorm+0.5*sigma_lognorm**2) + np.sqrt(var)
    x_errorrange = np.linspace(lowbound, upbound, 1000)
    y_pdfrange = lognorm.pdf(x_errorrange, s=sigma_lognorm,
        scale=np.exp(mu_lognorm))

    ax1.fill_between(x_errorrange, -1.1*min(y_pdfrange), y_pdfrange, facecolor='c', alpha=0.4,
        label=r'mean $\pm$ std', interpolate=True)

    ax1.set_xlabel('xsection in fb')
    ax1.set_ylabel('lognormal pdf')
    ax1.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
    ax1.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
    ax1.legend()
    ax1.set_ylim([-1.05*min(y), 1.1*max(y)])

    startx = norm.ppf(0.01, loc=mu_DGP, scale=sigma_DGP)
    endx = norm.ppf(0.99, loc=mu_DGP, scale=sigma_DGP)
    x = np.linspace(startx, endx, 100)
    y = norm.pdf(x, loc=mu_DGP, scale=sigma_DGP)
    ax2.plot(x, y, 'r-', lw=4, alpha=0.7, color='k', label='normal pdf')

    lowbound = mu_DGP - sigma_DGP
    upbound = mu_DGP + sigma_DGP
    x_errorrange = np.linspace(lowbound, upbound, 100)
    y_pdfrange = norm.pdf(x_errorrange, scale=sigma_DGP, loc=mu_DGP)
    # ax2.axhspan(x_errorrange, lowbound, upbound)

    ax2.fill_between(x_errorrange, -1.1*min(y_pdfrange), y_pdfrange, facecolor='c', alpha=0.4,
                     label=r'mean $\pm$ std', interpolate=True)

    ax2.set_ylim([-1.05*min(y), 1.1*max(y)])

    ax2.axvline(x=mu_DGP, color='b', label='mean/median/mode')
    ax2.set_xlabel(r"$\log_{10}$(xsection/fb)")
    ax2.set_ylabel('normal pdf')
    ax2.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
    ax2.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
    ax2.legend()
    plt.show()
