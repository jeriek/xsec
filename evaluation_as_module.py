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
import pandas as pd
import sys

###############################################
# Module begins here                          #
###############################################

xsections = [(1000021,1000021), (1000021,1000002)] # This is the preferred input now
squarks = [1000004, 1000003, 1000001, 1000002, 2000002, 2000001, 2000003, 2000004]
gluino = 1000021



# assert : check that tuple has length 2

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
    print all_features
    mean_index = ['m1000004', 'm1000003','m1000001','m1000002',
                  'm2000002', 'm2000001','m2000003','m2000004']
    
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
            
            features_index = ['m1000021', 'm'+str(max(xsection)), 'm'+str(min(xsection)) ]

        elif types[xsection] == 'qqbar':
            
            # Particle before antiparticle
            
            features_index = ['m1000021', 'm'+str(max(xsection)), 'm'+str(abs(min(xsection)))]

        # Create mass features
        features_nomean = masses[features_index].values
        m_mean = masses[mean_index].mean(axis=1).values.ravel()

        features = np.concatenate( ( features_nomean, m_mean.reshape(-1, 1) ), axis=1 )

        all_features[i] = features
        
    return np.asarray(all_features)
        






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

        # Make mass DataFrame
        df_masses = pd.DataFrame(masses, index=[0])

        ##################################################
        # Build feature vector                           #
        ##################################################
        
        # Decide the production channel type
        types = get_type(xsections)
        
        for xsection in xsections:
            assert len(xsection) == 2
            print 'The production type of ', xsection, 'is ', types[xsection]
        
        # Build feature vectors, depending on production channel type
        features = get_features(df_masses, xsections, types)
        print 'The features are ', features[0], features[1]


        ###################################################
        # Do DGP regression                               #
        ###################################################
        

        return 0

        # [sigma] = fb
        # (double, double, -1, -1)
        # return x,y,z,w # tuple? list? array?
        #print 'sigma, sigma_1/2, sigma_2, sigma_n'

        # Return a 4 dim- array for every production channel,
        # including sigma, sigma_1/2, sigma_2 and sigma_n
