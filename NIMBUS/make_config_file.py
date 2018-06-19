"""
 A program that generates a configuration file to be used in 
 NIMBUS to calculate cross sections. The program generates a pandas
 DataFrame. 

 @author: Ingrid A V Holm
"""
import pandas as pd
import numpy as np

"""
 The indexes in the file are: 

 Experts: The number of experts, an int

 Points p : The number of points per experts, an int or a np.array
 of the same dim as Experts

 Features : (m1, m2, mg, all, mean), include feature = 1, don't include = 0

 Kernel : Set the kernel to be used, M = Matern, RBF = RBF

 Noise : Whether noise should be determined by the kernel or set 
 to a fixed level. WK = decided by WhiteKernel, int = fixed level.

 Cut : A lower cut on the values of cross sections, int or False

 Target : Target transformations, 'mult mg' multiplies the cross section
 by mg^2, 'div mg' divides the cross section by mg^2 and False does nothing 

 Values are retrieved using: 
 df.loc[process][setting]
 E.g. 
 >> kernel_gg = df.loc['gg']['Kernel']

"""

squarks = ['1000004', '1000003', '1000001', '1000002', '2000002', '2000001', '2000003', '2000004']
gluino = '1000021'

A = {}

for i in range(len(squarks)):
    for j in range(i, len(squarks)):
        A.update({(squarks[i], squarks[j]) : {'Type' : 'qq',
                          'Experts' : 4 ,
                          'Points' : 1000,
                          'Kernel' : 'M',
                          'Target' : 'div mg',
                          'Outliers' : 'Remove',
                          'Cut' : 1e-16,
                          'Noise' : 'WK'}})

for i in range(len(squarks)):
    for j in range(i, len(squarks)):
        A.update({(squarks[i], str('-'+squarks[j])) : {'Type' : 'qqbar',
                          'Experts' : 4 ,
                          'Points' : 1000,
                          'Kernel' : 'M',
                          'Target' : 0,
                          'Outliers' : 'Remove',
                          'Cut' : 1e-16,
                          'Noise' : 'WK'}})    
for i in range(len(squarks)):
    A.update({(1000021, squarks[i]) : {'Type' : 'gq',
                      'Experts' : 4 ,
                      'Points' : 1000,
                      'Kernel' : 'M',
                      'Target' : 0,
                      'Outliers' : 'Remove',
                      'Cut' : 0,
                      'Noise' : 'WK'}})

A.update({(1000021, 1000021) : {'Type' : 'gg',
                                    'Experts' : 4 ,
                                    'Points' : 1000,
                                    'Kernel': 'M',
                                    'Target' : 'mult mg',
                                    'Outliers' : 'Remove',
                                    'Cut' : 0,
                                    'Noise' : 'WK'}})

df = pd.DataFrame.from_dict(A, orient='index')
#print 'I want the kernel to be', df.loc['gg']['Features']

df.to_csv('configfile.dat', orient='index', sep='\t')
print 'Created configuration file "configfile.dat"'
