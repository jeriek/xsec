"""
 A program that calls the NIMBUS class. 
 
 @author: Ingrid A V Holm
"""

from nimbus import nimbus

datafile = '../Data/MSSM-24/nimbus/nimbus_lin_1'
configfile = 'configfile.dat'

nimbus2000 = nimbus(configfile, datafile)

nimbus2000.setup()
nimbus2000.run()

