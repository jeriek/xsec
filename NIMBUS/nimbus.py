"""
 NIMBUS: A class for building DGP models of supersymmetric
 cross sections at the LHC.

 The class must be called in an initialize.py file, and 
 takes a data file and a configuration file as inputs.

 Squarks are the gluino are indexed by their PDG index, 
 and antiparticles are prefixed '-'. The indices are

   gluino : 1000021
   cL     : 1000004
   sL     : 1000003
   dL     : 1000001
   uL     : 1000002
   uR     : 2000002
   dR     : 2000001
   sR     : 2000003
   cR     : 2000004

 @author: Ingrid A V Holm
"""

import pandas as pd
import numpy as np
import sys

import sklearn
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import WhiteKernel, Matern, ConstantKernel as C
from sklearn.model_selection import train_test_split
from sklearn.externals import joblib
from joblib import Parallel, delayed

class nimbus:
    """
     A class to train distributed Gaussian process models for strong 
     supersymmetric processes at the LHC. The class takes as input: 

         1) A datafile containing masses and cross sections for training.

         2) A configuration file with the following information for all 
            desired processes: 
             -   Type of process: 'gg' (gluino-gluino), 'qq' (squark-squark), 
                 'gq' (gluino-squark) or 'qqbar' (squark-antisquark)
    
             -   Target transformation: 'div mg' (divide the cross sections
                 by the gluino mass squared), 'mul mg' (multiply the cross
                 by the gluino mass squared) or 0 (no changes)
    
             -   Outliers: Whether to remove cross sections set to zero by
                 Prospino: 'Remove' or 'Keep'.
    
             -   Cut on cross sections: Remove cross sections below the given
                 value, either a float or 0.

             -   Experts: Number of experts to use in training the DGP
             
             -   Number of points: Number of training points per expert
    
             -   Kernel: Kernel to use in Gaussian process regression, 
                 'M' (Matern kernel) or 'RBF' (RBF kernel)
    
             -   Noise in data: Whether to estimate the noise level in the
                 data using a WhiteKernel, or setting it manually by setting
                 a float.
    """

    def __init__(self, configfile, datafile):
        """
        Initialize an instance with a datafile and configuration file.
        """        
        print 'NIMBUS: A class for training DGP models for \n cross section evaluation.'
        
        self.df_config = pd.read_table(configfile, sep='\t')        
        self.df_data = pd.read_csv(datafile, sep=' ', skipinitialspace=True)


        
    def setup(self):
        """
        The function that readies everything for the DGP learning and training. 
        Calls the 'data_transformation' function for every process given in the
        configuration file. 

        Returns/gives an array of features and an array of targets for every 
        process.
        """
        print "Setting up..."

        # The two first columns in the configuration file are the
        # parton indices
        self.df_config = self.df_config.rename(columns = {'Unnamed: 0':'Parton1'})
        self.df_config = self.df_config.rename(columns = {'Unnamed: 1':'Parton2'})


        # Create empty lists for features and targets
        self.features = range(len(self.df_config.index))
        self.target = range(len(self.df_config.index))

        # Transform the data for every process
        for process_index in self.df_config.index:
            self.data_transformation(process_index)

        # Turn lists of features and targets into arrays
        self.features = np.asarray(self.features)
        self.target = np.asarray(self.target)

        print 'Models are trained for the production of:'
        for process_index in self.df_config.index:
            print self.df_config.loc[process_index]['Parton1'], ' and ', self.df_config.loc[process_index]['Parton2'] 

            
    def run(self):
        """
        Runs the DGP training for all processes. Calls the 'DGP' function for
        every process given in the configuration file. 

        The function 'run' must be called after 'setup', so that every process 
        has received a feature and target array.
        """
        # Give error message if 'setup' has not initialized the feature array
        try:
            self.features
        except AttributeError:
            print "Error: Set up before running!"
            sys.exit()

        # Train DGP models for all processes in the configuration file
        for process_index in self.df_config.index:
            self.DGP(process_index)



            
    def data_transformation(self, process_index):
        """
        This function transforms the data and creates feature 
        and target arrays, that are stored in a DataFrame. Specifications
        for the transformations are in the config-file.

        Input is the index of the process, and the function returns an 
        instance of self.
        """

        # Whether outliers should be removed
        target_outliers = self.df_config.loc[process_index]['Outliers']
        # Lower boundary on included cross sections
        target_cut = self.df_config.loc[process_index]['Cut']

        # Get the partons in the process
        parton1 = self.df_config.loc[process_index]['Parton1']
        parton2 = self.df_config.loc[process_index]['Parton2']

        # Decide the process
        process_name = str(parton1)+'_'+str(parton2)+'_NLO'

        parton1_mass = 'm'+str(parton1)
        parton2_mass = 'm'+str(parton2)
        
        # Remove outliers if the config file says so
        if target_outliers == 'Remove':
            mask_outlier = self.df_data[process_name] != 0
            self.df_data_inuse = self.df_data[mask_outlier]
        else:
            self.df_data_inuse = self.df_data

        # Introduce lower limit if the config file says so
        if target_cut > 0:
            mask_cut = self.df_data[process_name] > target_cut
            self.df_data_inuse = self.df_data_inuse[mask_cut]
        else: 
            mask_cut = self.df_data[process_name] > 0
            self.df_data_inuse = self.df_data_inuse[mask_cut]

        # Build the feature and target arrays
        inside_features = self.build_features(process_index)
        inside_target = self.build_target(process_index)

        # Add the features and target to the lists
        self.target[process_index] = inside_target
        self.features[process_index] = inside_features
        

        
    def build_target(self, process_index):

        # Get transformation information from config file
        transformation = self.df_config.loc[process_index]['Target']

        # Get the partons of the process
        parton1 = self.df_config.loc[process_index]['Parton1']
        parton2 = self.df_config.loc[process_index]['Parton2']

        # Decide the process
        process_name = str(parton1)+'_'+str(parton2)+'_NLO'

        target = self.df_data_inuse[process_name].values.ravel()
        mgluino = self.df_data_inuse['m1000021'].values.ravel()
        
        if transformation:
            if transformation == 'mult mg':
                target = target*mgluino**2
                
            elif transformation == 'div mg':
                target = target/mgluino**2
            #elif: something else you might want to do

        return target


    
        
    def build_features(self, process_index):

        squark_list = [1000004, 1000003, 1000001, 1000002,
                       2000002, 2000001, 2000003, 2000004]
        squark_mass_list = ['m1000004', 'm1000003', 'm1000001', 'm1000002',
                            'm2000002', 'm2000001', 'm2000003', 'm2000004']
        gluino = 'm1000021'
        
        # Get feature specification from config file
        parton1 = self.df_config.loc[process_index]['Parton1']
        parton2 = self.df_config.loc[process_index]['Parton2']
        
        production_type = self.df_config.loc[process_index]['Type']

        # Find array with mean squark mass
        m_mean = self.df_data_inuse[squark_mass_list].mean(axis=1).values.ravel()

        # Set feature index lists depending on production type
        if production_type == 'gg':
            # For gluino production use all squark masses, the gluino
            # mass and the mean squark mass
            feature_list = squark_mass_list
            feature_list.append(gluino)
            
        elif production_type == 'qq':
            # For squark pair production use the two masses, the gluino
            # mass and the mean squark mass
            if parton1 == parton2:
                feature_list = ['m'+str(parton1), gluino]
            else: 
                feature_list = ['m'+str(parton1), 'm'+str(parton2), gluino]
                
        elif production_type == 'qqbar':
            # For squark-antisquark production use the two masses, the
            # gluino mass and the mean squark mass

            if parton1 == -parton2:
                feature_list = ['m'+str(abs(parton1)), gluino]
            else:
                feature_list = ['m'+str(abs(parton1)), 'm'+str(abs(parton2)), gluino]
                
        elif production_type == 'gq':
            feature_list = ['m'+str(parton1), 'm'+str(parton2)]

        # Get array of features
        features_nomean = self.df_data_inuse[feature_list].values
        # Add mean squark mass to features
        features = np.concatenate( ( features_nomean, m_mean.reshape(-1, 1) ) , axis=1 )
        
        return features

    def DGP(self, process_index):
        """
        Function that does distributed Gaussian processes to train a model, 
        given features and target. Number of points and number of experts is 
        taken from the configuration file.

        DGP can be done in sequence for all processes by calling 'run()', or 
        in parallel by looping over DGP for the process indices from the 
        external program, e.g.
        
        >> nimbus2000 = nimbus('configfile.dat', 'datafile.dat')
        >> nimbus2000.setup()
        >> for process_index in nimbus2000.df_config.index: # MPI here
        >>         nimbus2000.DGP(process_index)
        """

        # Get features and target from array
        features_inside = self.features[process_index]
        target_inside = self.target[process_index]

        # Get number of experts and points per expert from config file
        n_experts = self.df_config.loc[process_index]['Experts']
        n_points = self.df_config.loc[process_index]['Points']

        # Split into training and test set
        X_train, X_test, y_train, y_test = train_test_split(features_inside, target_inside, random_state=42, train_size=n_points*n_experts)

        X_train_subsets = range(n_experts)
        y_train_subsets = range(n_experts)
        
        # Divide features and target into subsets
        for i in range(n_experts):
            X_train_subsets[i] = X_train[i*n_points:(i+1)*n_points]
            y_train_subsets[i] = y_train[i*n_points:(i+1)*n_points]

        X_train_subsets = np.asarray(X_train_subsets)
        y_train_subsets = np.asarray(y_train_subsets)
        
        # Set kernel name
        kernel_name = self.df_config.loc[process_index]['Kernel']

        # Set model name
        # Use time to create unique folder names
        import time
        
        folder = str(process_index)+'_'+str(time.time())+'/' 
        #folder = 'somefolder2/'
        model_name = folder + str(self.df_config.loc[process_index]['Parton1'])+'_'+str(self.df_config.loc[process_index]['Parton2'])
        
        print 'I do distributed Gaussian processes for', self.df_config.loc[process_index]['Parton1'], self.df_config.loc[process_index]['Parton2']
        print 'using %.f experts with %.f points each' % (n_experts, n_points)
        
        # Change n_jobs to n_experts
        out = Parallel(n_jobs=n_experts, verbose=4)(delayed(GP)(i, X_train_subsets[i], y_train_subsets[i], kernel_name, model_name) for i in range(n_experts))

        print 'NIMBUS: Checking out'


# To use joblib.Parallel the function must be defined outside the class
def GP(i, features, target, kernel_name, model_name):

    ###############################################
    # Kernel                                      #
    ###############################################

    # One length scale per feature
    char_len = np.zeros(len(features[0]))+1000
    # Set kernel for GP from config file

    if kernel_name == 'M':
        kernel = C(10, (10, 1000))*Matern(char_len,(10, 1e6), nu=1.5) + WhiteKernel(1e-3, (1e-7, 1e-2))
    
    ###############################################
    # Gaussian process                            #
    ###############################################
        
    my_gp = GaussianProcessRegressor(kernel=kernel, random_state=42)

    # Fit GP to training data
    my_gp.fit(features, target)


    ################################################
    # Save model                                   #
    ################################################
    # Expert number
    j = i+1
    filepath = model_name+'_'+str(len(features))+'_'+str(j)
    
    # If directory does not exists, create
    import os
    directory = os.path.dirname(filepath)
    if not os.path.exists(directory):
        os.makedirs(directory)


    # Save the model using joblib here, this can be changed
    joblib.dump(my_gp, filepath, compress=True)

    mybool = True
    return mybool
