"""
This program evaluates cross-sections
using DGP models trained by NIMBUS.
"""

# Import packages
import os
import sys
import warnings
import collections

import numpy as np # v1.14 or later
import joblib  # v0.12.2 or later

import kernels

print('Numpy version ' + np.__version__)
print('Joblib version ' + joblib.__version__)


###############################################
# Global variables                            #
###############################################

# Specify GP model data directory (can be reset in the run script)
DATA_DIR = './data'


# Specify available variation parameters (NOTE: to be changed!)
XSTYPES = ['','05','2','3','4','5']

# Link internal cross-section type (xstype) identifiers here to the
# corresponding Nimbus file suffixes for each trained xstype
# TODO: this should replace part of get_process_name()
XSTYPE_FILESUFFIX = {
    'centr' : '', # xsection @ central scale
    'sclup' : '_2', # xsection @ higher scale (2 x central scale)
    'scldn' : '_05', # xsection @ lower scale (0.5 x central scale)
    'pdf' : '_pdf', # xsection error due to pdf variation
    'aup' : '_aup', # xsection @ higher alpha_s
    'adn' : '_adn' # xsection @ lower alpha_s
    }
# List of the internal xstype identifiers
# XSTYPES = XSTYPE_FILESUFFIX.keys()

# List of selected processes (2-tuples of sparticle ids), to be set by
# the user
# xsections = [] # e.g. [(1000021,1000021), (1000021,1000002)]

# Definition of sparticle PDG ids
SQUARK_IDS = [1000001, 1000002, 1000003, 1000004,
              2000001, 2000002, 2000003, 2000004]
GLUINO_ID = 1000021

# For each selected process, store trained GP model dictionaries here
# (or a list of their cache locations):
PROCESS_DICT = {}


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
    cache_dir: string, optional
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


    global USE_CACHE, CACHE_DIR, FLUSH_CACHE, USE_MEMMAP
    USE_CACHE = use_cache
    CACHE_DIR = cache_dir
    FLUSH_CACHE = flush_cache
    USE_MEMMAP = use_memmap

    if USE_CACHE:
        if CACHE_DIR:
            cachedir = os.path.expandvars(CACHE_DIR) # expand environment variables
        else: # create directory with random name
            from tempfile import mkdtemp
            cachedir = mkdtemp(prefix='xsec_')
        if USE_MEMMAP:
            memmap_mode = 'c' # memmap mode: copy on write
        else:
            memmap_mode = None

        global memory
        memory = joblib.Memory(location=cachedir, mmap_mode=memmap_mode,
                               verbose=0)
        print("Cache folder: "+str(cachedir))

    return 0


###############################################
# Loading functions                           #
###############################################

def load_single_process(xsection_var):
    """
    Given a single process and cross-section type (e.g. gluino-gluino at
    the central scale), load the relevant trained GP models for all
    experts and return them in a list of dictionaries, one per expert.
    The input argument xsection_var is a 3-tuple (xsection[0],
    xsection[1], var) where the first two numbers represent the process
    and the last component is a string ('','05','2',...) from
    XSTYPES.
    """

    assert len(xsection_var) == 3 # ensure 3-component input
    # Construct location of GP models for the specified process and
    # cross-section type, using global data directory variable DATA_DIR
    process_dir = os.path.join(DATA_DIR, get_process_name(xsection_var))

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
        with open(model_file,"rb") as fo:
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

def load_processes(xsections):
    """
    Given a list of processes, load all relevant trained GP models into
    a cache folder on disk.
    """

    if USE_CACHE:
        load_single_process_cache = memory.cache(load_single_process)

    for xsection in xsections:
        # Search for all directories with same process, but
        # cross-section calculated at varied parameters
        for var in XSTYPES:
            xsection_var = (xsection[0],xsection[1],var)
            if USE_CACHE:
                PROCESS_DICT[xsection_var] = load_single_process_cache.call_and_shelve(xsection_var)
            else:
                PROCESS_DICT[xsection_var] = load_single_process(xsection_var)

    return 0

def set_kernel(params):
    """
    Given a parameter dictionary with keys {'matern_prefactor',
    'matern_lengthscale', 'matern_nu', 'whitekernel_noiselevel'}, return
    the corresponding (curried) Matern+WhiteKernel function of X and Y
    (default Y=None).
    """

    def kernel_function(X, Y=None):
        """
        Return the kernel function k(X, Y).
        """

        noise_level = params['whitekernel_noiselevel']
        prefactor = params['matern_prefactor']
        nu = params['matern_nu']
        length_scale = params['matern_lengthscale']

        if Y is None:
            sum = (kernels.WhiteKernel(X, noise_level=noise_level)
                  + prefactor*kernels.MaternKernel(X, length_scale=length_scale, nu=nu))
        else:
            sum = (kernels.WhiteKernel(X, Y, noise_level=noise_level)
                  + prefactor*kernels.MaternKernel(X, Y, length_scale=length_scale, nu=nu))

        return sum

    return kernel_function



###############################################
# Helper functions                            #
###############################################

def get_type(xsections):
    process_type = {}

    # Calculate alpha for wanted production channels
    for xsection in xsections:

        if xsection[0] in SQUARK_IDS and xsection[1] in SQUARK_IDS:
            process_type.update({xsection : 'qq'})
        elif xsection[0] in SQUARK_IDS and xsection[1] == GLUINO_ID:
            process_type.update({xsection : 'gq'})
        elif xsection[1] in SQUARK_IDS and xsection[0] == GLUINO_ID:
            process_type.update({xsection : 'gq'})
        elif xsection[0] in SQUARK_IDS and - xsection[1] in SQUARK_IDS:
            process_type.update({xsection : 'qqbar'})
        elif xsection[1] in SQUARK_IDS and - xsection[0] in SQUARK_IDS:
            process_type.update({xsection : 'qqbar'})
        elif xsection[0] == GLUINO_ID and xsection[1] == GLUINO_ID:
            process_type.update({xsection : 'gg'})

    return process_type


def get_features(masses, xsections, types):

    all_features = range(len(xsections)) # initialise dummy list
    # As dict
    all_features_dict = {}

    # print all_features
    mean_index = ['m1000004', 'm1000003','m1000001','m1000002',
                  'm2000002', 'm2000001','m2000003','m2000004']

    # No pandas
    mean_mass = sum([masses[key] for key in mean_index])/float(len(mean_index))

    for i in range(len(xsections)):

        xsection = xsections[i]

        if types[xsection] == 'gg':
            features_index = ['m1000021', 'm2000004', 'm2000003',
                              'm2000002','m2000001', 'm1000004',
                              'm1000003','m1000002','m1000001']

        elif types[xsection] == 'gq':
            features_index = ['m1000021', 'm'+str(xsection[1])] # Given that gluino is first

        elif types[xsection] == 'qq':

            # Largest index first, so convention
            # mcR, msR, muR, mdR, mcL, msL, muL, mdL
            if xsection[0] == xsection[1]:
                # Only use one mass if the partons are identical
                features_index = ['m1000021', 'm'+str(max(xsection))]
            else:
                features_index = ['m1000021', 'm'+str(max(xsection)), 'm'+str(min(xsection)) ]

        elif types[xsection] == 'qqbar':

            # Particle before antiparticle
            if abs(xsection[0]) == abs(xsection[1]):
                features_index = features_index = ['m1000021', 'm'+str(max(xsection))]
            else:
                features_index = ['m1000021', 'm'+str(max(xsection)), 'm'+str(abs(min(xsection)))]


        # Make a feature dictionary
        # features_dict = {key : masses[key] for key in features_index}
        features_dict = collections.OrderedDict()
        for key in features_index:
            features_dict[key] = masses[key]

        # Add mean squark mass to mass dict
        features_dict.update({'mean' : mean_mass})
        features = features_dict.values()

        # Add features to feature array
        all_features[i] = features
        all_features_dict.update({xsection : features})

    # Return feature array for all processes
    #return np.asarray(all_features)
    return all_features_dict


def get_process_name(xsection_var):
    # Get the partons of the process
    assert len(xsection_var) == 3
    parton1 = xsection_var[0]
    parton2 = xsection_var[1]
    param = xsection_var[2]

    # Decide process name
    # Check if one of the partons is a gluino
    if parton1 == GLUINO_ID:
        process_name = str(parton1)+'_'+str(parton2)+'_NLO'
    elif parton2 == GLUINO_ID:
        process_name = str(parton2)+'_'+str(parton1)+'_NLO'

    # Otherwise name starts with the largest parton PID
    elif abs(parton1) >= abs(parton2):
        process_name = str(parton1)+'_'+str(parton2)+'_NLO'
    elif abs(parton1) < abs(parton2):
        process_name =  str(parton2)+'_'+str(parton1)+'_NLO'

    # Add scale like '05','2' or other variation parameter like 'aup', 'adn'
    if param in XSTYPES and param is not '': # '' is also in XSTYPES
        if param == '05':
            process_name += '_05'
        elif param == '2':
            process_name += '_2'
        elif param == '3':
            process_name += '_pdf'
        elif param == '4':
            process_name += '_aup'
        elif param == '5':
            process_name += '_adn'
        else:
            print param, ' is not a valid variation parameter!'

    return process_name


###############################################
# Main functions                              #
###############################################

def eval_xsection(m1000021, m2000004, m2000003=None,
                  m2000002=None, m2000001=None, m1000004=None,
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
                print 'Error! Masses must either be given as two masses (mg, mq), \n \
                 or as all nine masses (mg, mcR, msR, muR, mdR, mcL, msL, muL, mdL).'
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
        masses = {'m1000021' : m1000021,'m1000004' : m1000004,
                  'm1000003' : m1000003, 'm1000001': m1000001,
                  'm1000002' : m1000002, 'm2000002' : m2000002,
                  'm2000001' : m2000001, 'm2000003' : m2000003, 'm2000004' : m2000004}


        ##################################################
        # Build feature vector                           #
        ##################################################

        # Decide the production channel type
        types = get_type(xsections)

        for xsection in xsections:
            assert len(xsection) == 2
            # print 'The production type of ', xsection, 'is ', types[xsection]

        # Build feature vectors, depending on production channel type
        features = get_features(masses, xsections, types)
        # print 'The features are ', features


        ###################################################
        # Do DGP regression                               #
        ###################################################

        for xsection in xsections: # alternatively: write a 2nd loop over var in XSTYPES
            mu_dgp, sigma_dgp = DGP(xsection, features[xsection], scale=1.0)
            scale_05_dgp, sigma_05_dgp = DGP(
                xsection, features[xsection], scale=0.5)
            scale_2_dgp, sigma_2_dgp = DGP(
                xsection, features[xsection], scale=2.0)
            scale_3_dgp, sigma_3_dgp = DGP(
                xsection, features[xsection], scale=3.0)
            scale_4_dgp, sigma_4_dgp = DGP(
                xsection, features[xsection], scale=4.0)
            scale_5_dgp, sigma_5_dgp = DGP(
                xsection, features[xsection], scale=5.0)
            # scale_05_dgp, _ = DGP(xsection, features[xsection], scale=0.5)
            # scale_2_dgp, _ = DGP(xsection, features[xsection], scale=2.0)
            # scale_3_dgp, _ = DGP(xsection, features[xsection], scale=3.0)
            # scale_4_dgp, _ = DGP(xsection, features[xsection], scale=4.0)
            # scale_5_dgp, _ = DGP(xsection, features[xsection], scale=5.0)

            print "DGP, scale 1:", mu_dgp, sigma_dgp
            print "DGP, scale 0.5:", scale_05_dgp, sigma_05_dgp
            print "DGP, scale 2:", scale_2_dgp, sigma_2_dgp
            print "DGP, pdf:", scale_3_dgp, sigma_3_dgp
            print "DGP, aup:", scale_4_dgp, sigma_4_dgp
            print "DGP, adn:", scale_5_dgp, sigma_5_dgp
        
        # All returned numbers defined to be deviations from 1

        # -- Central-scale xsection and regression error (= standard
        #    deviation) in fb.
        xsection_central, reg_err = moments_lognormal(mu_dgp, sigma_dgp)  
        # -- Xsection deviating one (lognormal) regression error away
        #    from the central-scale xsection, relative to the latter.
        regdown_rel = 1 - reg_err/xsection_central  
        regup_rel = 1 + reg_err/xsection_central 
        # -- Xsection at lower and higher scale (0.5x and 2x central
        #    scale), relative to the central-scale xsection. To prevent
        #    that the unusual case with xsection_scaleup >
        #    xsection_scaledown causes errors, min/max ensures
        #    scaledown_rel gives the lower bound and scaleup_rel the
        #    higher one.
        scaledown_rel = np.minimum(scale_05_dgp, scale_2_dgp)
        scaleup_rel = np.maximum(scale_05_dgp, scale_2_dgp)
        # -- Xsection deviating one pdf error away from the
        #    central-scale xsection, relative to the latter.
        delta_pdf_rel = scale_3_dgp
        pdfdown_rel = 1 - delta_pdf_rel
        pdfup_rel = 1 + delta_pdf_rel
        # -- Xsection deviating one symmetrised alpha_s error away from
        #    the central-scale xsection, relative to the latter.
        delta_alphas_rel = 0.5 * (abs(scale_4_dgp-1) 
                                  + abs(1-scale_5_dgp))
        alphasdown_rel = 1 - delta_alphas_rel 
        alphasup_rel = 1 + delta_alphas_rel

        # Collect values for output in Numpy array
        return_array = np.array([
            xsection_central, regdown_rel, regup_rel, 
            scaledown_rel, scaleup_rel, pdfdown_rel, 
            pdfup_rel, alphasdown_rel, alphasup_rel
            ])

        print "************** NEW OUTPUT FORMAT ******************"
        nr_dec = 4
        print "* xsection_central (fb):", '{:0.4e}'.format(xsection_central)
        print "* regdown_rel, regup_rel:", round(
            regdown_rel, nr_dec), round(regup_rel, nr_dec)
        print "* scaledown_rel, scaleup_rel:", round(scaledown_rel, nr_dec), round(scaleup_rel, nr_dec)
        print "* pdfdown_rel, pdfup_rel:", round(pdfdown_rel, nr_dec), round(pdfup_rel, nr_dec)
        print "* alphasdown_rel, alphasup_rel:", round(alphasdown_rel, nr_dec), round(alphasup_rel, nr_dec)
        print "***************************************************"
        # NOTE: plot just for comparison during testing
        plot_lognormal(mu_dgp, sigma_dgp, m1000021)

        # xsec_array = np.asarray([mu_dgp, sigma_dgp, scale_05_dgp, sigma_05_dgp,
                                #  scale_2_dgp, sigma_2_dgp, scale_3_dgp, sigma_3_dgp,
                                #  scale_4_dgp, sigma_4_dgp, scale_5_dgp, sigma_5_dgp])

        # print return_array
        return return_array


def DGP(xsection, features, scale):
    # Get name of production channel, and name
    # of model folder

    if scale==0.5:
        xsection_var = (xsection[0], xsection[1], '05')
    elif scale==2.0:
        xsection_var = (xsection[0], xsection[1], '2')
    elif scale==3.0:
        xsection_var = (xsection[0], xsection[1], '3')
    elif scale==4.0:
        xsection_var = (xsection[0], xsection[1], '4')
    elif scale==5.0:
        xsection_var = (xsection[0], xsection[1], '5')
    else: # scale==1.0
        xsection_var = (xsection[0], xsection[1], '')

    # Decide which GP to use depending on 'scale'
    process_name = get_process_name(xsection_var)
    # print process_name

    # List all trained experts in the chosen directory

    models = os.listdir(os.path.join(DATA_DIR, process_name))
    n_experts = len(models)

    # print 'Models:', models

    # Empty arrays where all predicted numbers are stored
    mus = np.zeros(n_experts)
    sigmas = np.zeros(n_experts)
    sigma_priors = np.zeros(n_experts)

    # Loop over GP models/experts
    for i in range(len(models)):
        # model = models[i]
        mu, sigma, sigma_prior = GP_predict(xsection_var, features, index=i, return_std=True)
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
    mu_DGP = 0.
    var_DGP_inv = 0. # (sigma^2)^-1

    # Combine sigmas
    for i in range(N):
        var_DGP_inv += (betas[i] * sigmas[i]**(-2)
                        + (1./n_experts - betas[i])
                        * sigma_priors[i]**(-2))

    # Combine mus
    for i in range(N):
        mu_DGP +=  var_DGP_inv**(-1) * (betas[i] * sigmas[i]**(-2) * mus[i])

    # Transform back to cross section
    # production_type = get_type([xsection])


    # Return mean and std
    return mu_DGP, np.sqrt(var_DGP_inv**(-1))



def GP_predict(xsection_var, features, index=0, return_std=True, return_cov=False):
    """
    Gaussian process regression for the individual experts. Takes as
    input arguments the produced partons, an array of new test features,
    and the index number of the expert. Requires running
    load_processes(...) first.

    Returns a list of numpy arrays containing the mean value (the
    predicted cross-section), the GP standard deviation (or full
    covariance matrix), and the square root of the prior variance on the
    test features.

    Based on GaussianProcessRegressor.predict(...) from scikit-learn and
    algorithm 2.1 of Gaussian Processes for Machine Learning by
    Rasmussen and Williams.

    """

    if return_std and return_cov:
        raise RuntimeError("Cannot return both standard deviation "
                           "and full covariance.")

    try:
        if USE_CACHE:
            # model_list = PROCESS_DICT[xsection_var].get() # list of loaded models for the specified process
            gp_model = PROCESS_DICT[xsection_var].get()[index]
        else:
            gp_model = PROCESS_DICT[xsection_var][index]

        kernel = gp_model['kernel']
        X_train = gp_model['X_train']
        alpha = gp_model['alpha']
        L_inv = gp_model['L_inv']
        K_inv = gp_model['K_inv']
        kernel = set_kernel(gp_model['kernel'])

        # if xsection_var[2] == '':
        #     print("- Do GP regression for: " + get_process_name(xsection_var) + " at scale 1.0")
        # else:
        #     print ("- Do GP regression for: " + get_process_name(xsection_var) + " with variation parameter " + xsection_var[2])
    except KeyError, e:
        print(KeyError, e)
        print("No trained GP models loaded for: " + str(xsection_var))
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
            warnings.warn("Predicted some variance(s) smaller than 0. "
                          "Approximating these with their absolute value.")
            y_var[y_var_negative] = np.abs(y_var[y_var_negative]) # not set to 0 or 1e-99, but to approximated right size
        y_std = np.sqrt(y_var)
        prior_std = np.sqrt(prior_variance.flatten())
        return y_mean, y_std, prior_std

    elif return_cov:
        v = L_inv.dot(K_trans.T) # Line 5
        y_cov = prior_variance - K_trans.dot(v)  # Line 6
        return [y_mean, y_cov, prior_variance]

    else:
        return y_mean

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
    print "std_lognorm/mean_lognorm = ", std_lognorm/mean_lognorm

    return mean_lognorm, std_lognorm

def plot_lognormal(mu_DGP, sigma_DGP, m_gluino):
    # NOTE: Just for testing, not for release.

    from scipy.stats import lognorm, norm
    import matplotlib.pyplot as plt
    
    # Correction for "mult mg"
    # m_gluino = 1000.
    mu_DGP -= 2*np.log(m_gluino)/np.log(10)

    mu_lognorm = mu_DGP*np.log(10.)
    sigma_lognorm = sigma_DGP*np.log(10.)

    # mu_lognorm += 2*np.log(m_gluino)

    fig, (ax1, ax2) = plt.subplots(1, 2)
    mean, var, skew, kurt = lognorm.stats(
        s=sigma_lognorm, scale=np.exp(mu_lognorm), moments='mvsk')
    startx = lognorm.ppf(0.001, s=sigma_lognorm, scale=np.exp(mu_lognorm))
    endx = lognorm.ppf(0.999, s=sigma_lognorm, scale=np.exp(mu_lognorm))
    x = np.linspace(startx, endx, 100)
    y = ax1.plot(x, lognorm.pdf(x, s=sigma_lognorm, scale=np.exp(mu_lognorm)),
        'r-', lw=4, alpha=0.7, color='k', label='lognormal pdf')

    ax1.axvline(x=np.exp(mu_lognorm-sigma_lognorm**2), color='r', label='mode')
    ax1.axvline(x=np.exp(mu_lognorm), color='g', label='median')
    ax1.axvline(x=np.exp(mu_lognorm+0.5*sigma_lognorm**2), color='b', label='mean')
    
    # lowbound = ax1.axvline(x=np.exp(mu_lognorm+0.5*sigma_lognorm**2)-np.sqrt(var), color='m', label='mean - std')
    # upbound = ax1.axvline(x=np.exp(mu_lognorm+0.5*sigma_lognorm**2)+np.sqrt(var), color='m', label='mean + std')
    lowbound = np.exp(mu_lognorm+0.5*sigma_lognorm**2) - np.sqrt(var)
    upbound = np.exp(mu_lognorm+0.5*sigma_lognorm**2) + np.sqrt(var)
    x_errorrange = np.linspace(lowbound, upbound, 100)
    y_pdfrange = lognorm.pdf(x_errorrange, s=sigma_lognorm, 
        scale=np.exp(mu_lognorm))

    ax1.fill_between(x_errorrange, -1.1*min(y_pdfrange), y_pdfrange, facecolor='c', alpha=0.4, 
        label=r'mean $\pm$ std', interpolate=True)

    ax1.set_xlabel('xsection in fb')
    ax1.set_ylabel('lognormal pdf')
    ax1.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
    ax1.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
    ax1.legend()
    tot_yrange = lognorm.pdf(x, s=sigma_lognorm, scale=np.exp(mu_lognorm))
    ax1.set_ylim([-1.05*min(tot_yrange), 1.1*lognorm.pdf(np.exp(mu_lognorm),
        s=sigma_lognorm, scale=np.exp(mu_lognorm))])

    startx = norm.ppf(0.001, loc=mu_DGP, scale=sigma_DGP)
    endx = norm.ppf(0.999, loc=mu_DGP, scale=sigma_DGP)
    x = np.linspace(startx, endx, 100)
    ax2.plot(x, norm.pdf(x, loc=mu_DGP, scale=sigma_DGP),
             'r-', lw=4, alpha=0.7, color='k', label='normal pdf')

    lowbound = mu_DGP - sigma_DGP
    upbound = mu_DGP + sigma_DGP
    x_errorrange = np.linspace(lowbound, upbound, 100)
    y_pdfrange = norm.pdf(x_errorrange, scale=sigma_DGP, loc=mu_DGP)
    # ax2.axhspan(x_errorrange, lowbound, upbound)
    tot_yrange = norm.pdf(x, scale=sigma_DGP, loc=mu_DGP)
    ax2.fill_between(x_errorrange, -1.1*min(y_pdfrange), y_pdfrange, facecolor='c', alpha=0.4,
                     label=r'mean $\pm$ std', interpolate=True)
                     
    ax2.set_ylim([-1.05*min(tot_yrange), 1.1 *
                  norm.pdf(mu_DGP, scale=sigma_DGP, loc=mu_DGP)])

    ax2.axvline(x=mu_DGP, color='b', label='mean/median/mode')
    ax2.set_xlabel(r"$\log_{10}$(xsection/fb)")
    ax2.set_ylabel('normal pdf')
    ax2.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
    ax2.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
    ax2.legend()
    plt.show()


def clear_cache():
    if USE_CACHE and FLUSH_CACHE:
        # Flush the cache completely
        memory.clear(warn=False)
    return 0
