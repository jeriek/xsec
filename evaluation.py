"""
This program evaluates cross sections
using DGP models trained by NIMBUS.
"""

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
        
class evaluation:
    def __init__(self, configfile):
        """
        Take the configuration file that contains the production
        channels you want to evaluate.  
        """
        self.df = pd.DataFrame(configfile)

        # Calculate alpha for wanted production channels
