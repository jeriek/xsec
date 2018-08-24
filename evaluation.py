"""
This program evaluates cross sections
using DGP models trained by NIMBUS.
"""

# Import packages
import os, sys
import socket
import numpy as np
import joblib
from tempfile import mkdtemp
import warnings
import kernels

print('Numpy version ' + np.__version__)
print('Joblib version ' + joblib.__version__)


# Specify GP model data directory
HOSTNAME = socket.gethostname()
if HOSTNAME == 'Lorien':
    DATA_DIR = '/home/jeriek/Documents/prospino-PointSampler/NIMBUS/GPdata'
elif HOSTNAME == 'audrey3':
    DATA_DIR = '/home/ingrid/Documents/VitAs/xsec'
elif HOSTNAME == 'yourhostname123':
    DATA_DIR = '/your/GP/model/data/directory/'
else:
    print("Error: unknown hostname -- specify your data directory first")
    sys.exit()

# Specify available variation parameters (the exact GP model directory suffixes)
VARIATION_PAR = ['','05','2'] # 'aup','adn', ... ('' for scale 1.0)


# TO DO:
# - pyslha input 
# - change asserts

###############################################
# Global variables                            #
###############################################

xsections = [(1000021,1000021), (1000021,1000002)] # This is the preferred input now

squarks = [1000004, 1000003, 1000001, 1000002, 2000002, 2000001, 2000003, 2000004]
gluino = 1000021


# Temporary directory to store loaded GP models for use in predictive functions
cachedir = mkdtemp()
# memory = joblib.Memory(cachedir=cachedir, mmap_mode='r')
memory = joblib.Memory(cachedir=cachedir, mmap_mode='c',verbose=0) # memmap mode: copy on write
print("Cache folder: "+str(cachedir))

# cachedir parameter deprecated in joblib 0.12, check new examples on GitHub
# location = './cachedir'
# memory = joblib.Memory(location, mmap_mode='r')


# For each selected process, store a reference to a cached list of trained GP model dictionaries 
process_dict = {} 


###############################################
# Loading functions                           #
###############################################

@memory.cache
def load_single_process(xsection_var):
    """
    Given a single process, load the relevant trained GP models (saved as dictionaries)
    and return them in a list. The input argumen xsection_var is a 3-tuple (xsection[0],xsection[1],var)
    where the first two numbers represent the process and the last component is a string ('','05','2',...)
    from VARIATION_PAR, the list of 
    """

    assert len(xsection_var) == 3
    process_dir = os.path.join(DATA_DIR, get_process_name(xsection_var))

    model_files = [os.path.join(process_dir, f) for f in os.listdir(process_dir) 
                    if os.path.isfile(os.path.join(process_dir, f))]
    # print(model_files)

    model_list = [] # list of GP model dictionaries 

    for model_file in model_files:
        with open(model_file,"rb") as fo:
            gp_model = joblib.load(fo)
            # Reconvert float32 arrays to float64 for higher-precision computations
            gp_reco = {}
            gp_reco['X_train'] = gp_model['X_train'].astype('float64')
            # gp_reco['y_train'] = gp_model['y_train'].astype('float64')
            gp_reco['L_inv'] = gp_model['L_inv'].astype('float64') 
            gp_reco['alpha'] = gp_model['alpha'].astype('float64')
            gp_reco['K_inv'] = gp_reco['L_inv'].dot(gp_reco['L_inv'].T)
            # Change kernel parameter dictionary to Matern+WhiteKernel function
            # gp_reco['kernel'] = set_kernel(gp_model['kernel']) # NOTE: not working since joblib won't memmap complex callable objects
            gp_reco['kernel'] = gp_model['kernel'] # for now, initialise kernel functions in GP_predict()
            model_list.append(gp_reco)

    return model_list

def load_processes(xsections):
    """
    Given a list of processes, load all relevant trained GP models 
    into a cache folder on disk.
    """

    for xsection in xsections:
        # Search for all directories with same process, but cross-section calculated at varied parameters
        for var in VARIATION_PAR:
            xsection_var = (xsection[0],xsection[1],var)
            process_dict[xsection_var] = load_single_process.call_and_shelve(xsection_var)
    return 0

def set_kernel(params):
    """
    Given a parameter dictionary with keys {'matern_prefactor', 'matern_lengthscale', 
    'matern_nu', 'whitekernel_noiselevel'}, return the corresponding (curried) Matern+WhiteKernel 
    function of X and Y (default Y=None).
    """

    def kernel_function(X, Y=None):
        """
        Return the kernel function k(X, Y). Not using a kernel class object in this version.
        """

        noise_level = params['whitekernel_noiselevel']
        prefactor = params['matern_prefactor']
        nu = params['matern_nu']
        length_scale = params['matern_lengthscale']

        if Y is None:
            sum = kernels.WhiteKernel(X, noise_level=noise_level) + prefactor*kernels.MaternKernel(X, length_scale=length_scale, nu=nu)
        else:
            sum = kernels.WhiteKernel(X, Y, noise_level=noise_level) + prefactor*kernels.MaternKernel(X, Y, length_scale=length_scale, nu=nu)

        return sum

    return kernel_function



###############################################
# Helper functions                            #
###############################################

def get_type(xsections):
    process_type = {}

    # Calculate alpha for wanted production channels
    for xsection in xsections:
        
        if xsection[0] in squarks and xsection[1] in squarks:
            process_type.update({xsection : 'qq'})
        elif xsection[0] in squarks and xsection[1] == gluino:
            process_type.update({xsection : 'gq'})
        elif xsection[1] in squarks and xsection[0] == gluino:
            process_type.update({xsection : 'gq'})                
        elif xsection[0] in squarks and - xsection[1] in squarks:
            process_type.update({xsection : 'qqbar'})
        elif xsection[1] in squarks and - xsection[0] in squarks:
            process_type.update({xsection : 'qqbar'})                
        elif xsection[0] == gluino and xsection[1] == gluino:
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
            features_index = ['m1000021', 'm1000004', 'm1000003',
                              'm1000001','m1000002', 'm2000002',
                              'm2000001','m2000003','m2000004']

        elif types[xsection] == 'gq':
            features_index = ['m1000021', 'm'+str(xsection[1])] # Given that gluino is first

        elif types[xsection] == 'qq':
            
            # Largest index first, so convention
            # mcR, msR, muR, mdR, mcL, msL, muL, mdL
            if xsection[0] == xsection[1]:
                # Only use one mass if the partons are identical 
                features_index = ['m1000021', 'm'+str(max(xsection)) ]
            else:
                features_index = ['m1000021', 'm'+str(max(xsection)), 'm'+str(min(xsection)) ]

        elif types[xsection] == 'qqbar':
            
            # Particle before antiparticle
            if abs(xsection[0]) == abs(xsection[1]):
                features_index = features_index = ['m1000021', 'm'+str(max(xsection))]
            else:
                features_index = ['m1000021', 'm'+str(max(xsection)), 'm'+str(abs(min(xsection)))]


        # Make a feature dictionary
        features_dict = {key : masses[key] for key in features_index}

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
    if parton1 == 1000021:
        process_name = str(parton1)+'_'+str(parton2)+'_NLO'
    elif parton2 == 1000021:
        process_name = str(parton2)+'_'+str(parton1)+'_NLO'

    # Otherwise name starts with the largest parton PID
    elif abs(parton1) >= abs(parton2):
        process_name = str(parton1)+'_'+str(parton2)+'_NLO'
    elif abs(parton1) < abs(parton2):
        process_name =  str(parton2)+'_'+str(parton1)+'_NLO'
    
    # Add scale like '05','2' or other variation parameter like 'aup', 'adn'
    if param in VARIATION_PAR and param is not '': # '' is also in VARIATION_PAR 
        process_name += '_'+param

    return process_name


###############################################
# Main functions                              #
###############################################
            
def eval_xsection(m1000021, m1000004, m1000003=None,
                  m1000001=None, m1000002=None, m2000002=None,
                  m2000001=None, m2000003=None, m2000004=None):
        """
        Read masses and parameters from slha-file and evaluate 
        cross sections
        """

        ##################################################
        # Check masses                                   #
        ##################################################

        if m1000003 is not None:

            # If more than two two first masses are provided, then
            # all squark masses must be provided. 
            
            try:
                m1000001+m1000002+m2000002+m2000001+m2000003+m2000004 # If you try to add a number and a None, you get False
            except TypeError:
                print 'Error! Masses must either be given as two masses (mg, mq),\n or as all nine masses (mg, mcL, msL, mdL, muL, muR, mdR, msR, mcR)'
                sys.exit()
            
        else:

            # If only the two first masses are given the squark masses
            # are degenerate, and set to m1000004
            
            print 'This is the common mass'
            m1000001 = m1000004
            m1000002 = m1000004
            m2000002 = m1000004
            m2000001 = m1000004
            m2000003 = m1000004
            m2000004 = m1000004

            
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
            print 'The production type of ', xsection, 'is ', types[xsection]
        
        # Build feature vectors, depending on production channel type
        features = get_features(masses, xsections, types)
        print 'The features are ', features


        ###################################################
        # Do DGP regression                               #
        ###################################################
        
        load_processes(xsections)

        for xsection in xsections: # alternatively: write a 2nd loop over var in VARIATION_PAR
            mu_dgp, sigma_dgp = DGP(xsection, features[xsection], scale=1.0)
            scale_05_dgp, sigma_05_dgp = DGP(xsection, features[xsection], scale=0.5)
            scale_2_dgp, sigma_2_dgp = DGP(xsection, features[xsection], scale=2.0)
            print "DGP, scale 1:", mu_dgp, sigma_dgp
            print "DGP, scale 0.5:", scale_05_dgp, sigma_05_dgp
            print "DGP, scale 2:", scale_2_dgp, sigma_2_dgp
            # Here we put in PDF and alpha variations

        # Flush the cache completely  
        memory.clear(warn=True)

        return 0


    

def DGP(xsection, features, scale):
    # Get name of production channel, and name
    # of model folder
    
    if scale==0.5:
        # process_name = process_name+'_05'
        xsection_var = (xsection[0],xsection[1],'05')
    elif scale==2.0:
        # process_name = process_name+'_2'
        xsection_var = (xsection[0],xsection[1],'2')
    else:
        xsection_var = (xsection[0],xsection[1],'')

    # Decide which GP to use depending on 'scale'
    process_name = get_process_name(xsection_var)
    print process_name
    
    # List all trained experts in the chosen directory

    models = os.listdir(os.path.join(DATA_DIR, process_name))
    n_experts = len(models)

    print 'Models:', models

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
        print "-- Resulting mu, sigma:", mu, sigma

    ########################################
    # Assume here that mus, sigmas and
    # sigma_priors are full arrays

    N = len(mus)
    
    # Find weight (beta) for each expert
    betas = 0.5*( 0.5*( np.log(sigma_priors) - np.log(sigmas) ) )

    # Final mean and variance
    mu_DGP = 0
    sigma_DGP_neg = 0 # (sigma^2)^-1

    # Combine sigmas
    for i in range(N):
        sigma_DGP_neg += betas[i] * sigmas[i]**(-1)+(1./n_experts - betas[i]) * sigma_priors[i]**(-1)

    # Combine mus
    for i in range(N):
        mu_DGP +=  sigma_DGP_neg**(-1) * ( betas[i] * sigmas[i]**(-1) * mus[i] )


    # Transform back to cross section
    production_type = get_type([xsection])
    
        
    # Return mean and variance, maybe change this to std
    return mu_DGP, sigma_DGP_neg**(-1) 

        

def GP_predict(xsection_var, features, index=0, return_std=True, return_cov=False):
    """
    Gaussian process regression for the individual experts.
    Takes as input arguments the produced partons, an array of new
    test features, and the index number of the expert. 
    Requires running load_processes(...) first.

    Returns a list of numpy arrays containing the mean value (the predicted 
    cross-section), the GP standard deviation (or full covariance matrix), 
    and the square of the prior variance on the test features.

    Based on GaussianProcessRegressor.predict(...) from scikit-learn and 
    algorithm 2.1 of Gaussian Processes for Machine Learning by Rasmussen
    and Williams.

    """
    
    if return_std and return_cov:
        raise RuntimeError("Cannot return both standard deviation " 
                           "and full covariance.")

    try:
        # model_list = process_dict[xsection_var].get() # list of loaded models for the specified process 
        gp_model = process_dict[xsection_var].get()[index]

        kernel = gp_model['kernel']
        X_train = gp_model['X_train']
        alpha = gp_model['alpha']
        # y_train = gp_model['y_train'] # not needed if we don't normalise?
        L_inv = gp_model['L_inv']
        K_inv = gp_model['K_inv']
        # kernel = gp_model['kernel'] # NOTE: not working since joblib won't memmap complex callable objects
        kernel = set_kernel(gp_model['kernel'])

        if xsection_var[2] == '':
            print("- Do GP regression for: " + get_process_name(xsection_var) + " at scale 1.0")
        else:
            print ("- Do GP regression for: " + get_process_name(xsection_var) + " with variation parameter " + xsection_var[2])
    except KeyError, e:
        print(KeyError, e)
        print("No trained GP models loaded for: " + str(xsection_var)) 
        return None
    
    X = np.atleast_2d(features) # not needed if just 1 new test point!

    K_trans = kernel(X, X_train) # transpose of K*
    y_mean = K_trans.dot(alpha) # Line 4 (y_mean = f_star)
    # y_mean = y_train_mean + y_mean # Only use if normalisation performed

    prior_variance = kernel(X)

    if return_cov:
        v = L_inv.dot(K_trans.T) # Line 5
        y_cov = prior_variance - K_trans.dot(v)  # Line 6
        return [y_mean, y_cov, prior_variance]
    elif return_std:
        # Compute variance of predictive distribution
        y_var = np.diag(kernel(X)) # NOTE: can be optimised in class object with kernel.diag(X)!
        y_var.setflags(write=True) # somehow this array is set to read-only
        y_var -= np.einsum("ij,ij->i", np.dot(K_trans, K_inv), K_trans)

        # Check if any of the variances is negative because of
        # numerical issues. If yes: set the variance to 0.
        y_var_negative = y_var < 0
        if np.any(y_var_negative):
            warnings.warn("Predicted variances smaller than 0. "
                          "Setting those variances to 0.")
            y_var[y_var_negative] = 0.0
        return y_mean, np.sqrt(y_var), np.sqrt(prior_variance.flatten())
    else:
        return y_mean


