"""
This program evaluates cross sections
using DGP models trained by NIMBUS.
"""

# Return exp(mu)*exp(0.5*log(10)*sigma**2)

# Check that joblib and pyslha are installed
# OBS: This does not work with pip 10.0.1 ! 
import pip

pkgs = ['joblib', 'pyslha', 'pandas']
for package in pkgs:
    try:
        import package
    except ImportError, e:
        pip.main(['install', package])

# Import packages
import numpy as np
import pandas as pd # Not pandas?
import sys

###############################################
# Module begins here                          #
###############################################

xsections = [(1000021,1000021), (1000021,1000002)] # This is the preferred input now
squarks = [1000004, 1000003, 1000001, 1000002, 2000002, 2000001, 2000003, 2000004]
gluino = 1000021



def load_models(xsections):
    # Jeriek

    # Load models, Ls, kernels; Xtrain, ytrain
    # Make persistent, so you can load in GP function


    # path_dict = {'gluino_gluino' : 'path to L', 'gluino_squark' : 'path to L',}
    #L[xsection] = ...
    
    return 0


def get_type(xsections):
    process_type = {}

    # Calculate alpha for wanted production channels
    for i in range(len(xsections)):
        xsection = xsections[i]
        
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

    all_features = range(len(xsections))
    # As dict
    all_features_dict = {}

    print all_features
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

        






# Main function
            
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
        

        for xsection in xsections:
            mu_dgp, sigma_dgp = DGP(xsection, features[xsection], scale=1.0)
            scale_05_dgp, sigma_05_dgp = DGP(xsection, features[xsection], scale=0.5)
            scale_2_dgp, sigma_2_dgp = DGP_2(xsection, features[xsection], scale=2.0)
            
        return 0


    

def DGP(xsection, features, scale):
    # Get name of production channel, and name
    # of model folder
    
    process_name = get_process_name(xsection)
    # Decide which GP to use depending on 'scale'
    
    # List all trained experts in the chosen directory

    import os
    models = os.listdir(process_name)
    n_experts = len(models)

    # Empty arrays where all predicted numbers are stored
    mus = np.zeros(n_experts)
    sigmas = np.zeros(n_experts)
    sigma_priors = np.zeros(n_experts)

    
    # Loop over GP models/experts
    for i in range(len(models)):
        model = models[i]
        print model
        mu, sigma, sigma_prior = GP(model, process_name, features)
        mus[i] = mu
        sigmas[i] = sigma
        sigma_priors[i] = sigma_prior


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


        

        

    


# Jerieks part
        
def GP(model, xsection, features):
    """
    The function that does Gaussian process regression
    for the individual experts.

    Takes as input arguments the produced partons, and
    a feature array. 

    Returns a mean value (the predicted cross section), 
    the GP standard deviation and the prior variance: 

    prior variance = kernel( x_test, x_test)

    """
    #Linv = Linv[xsection]
    
    print 'Do GP regression using', model
    return [1,1,1]




def get_process_name(process_index):
    # Get the partons of the process
    parton1 = process_index[0]
    parton2 = process_index[1]

    # Decide process name

    # Check if one of the partons is a gluino
    if parton1 == 1000021:
        process_name = str(parton1)+'_'+str(parton2)+'_NLO'
    elif parton2 == 1000021:
        process_name = str(parton2)+'_'+str(parton1)+'_NLO'

    # Otherwise name starts with the largest parton PID
    elif parton1 > parton2:
        process_name = str(parton1)+'_'+str(parton2)+'_NLO'
    elif parton2 < parton1:
        proess_name =  str(parton2)+'_'+str(parton1)+'_NLO'

    return process_name
