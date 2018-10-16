#!/usr/bin/env python2

# @todo Switch shebang to "!/usr/bin/env python" when the code works with both python2 and python3

"""
Simple script for downloading trained Gaussian Processes 
for use with the cross-section evaluation code

@author Anders Kvellestad
        anders.kvellestad@fys.uio.no

Usage:
  ./download_gprocs.py 
"""


# Import packages
from __future__ import print_function
import urllib2  
# @todo Import and use urllib.request for python3
import sys


# @todo Let the user choose specific GPs to download via command line options.

#
# If no GPs are specified at the command line, download all of them.
# Each download is specified as a (url,filename) tuple in the list below.
#
# @todo Replace the coding cats with the proper GP files
dowload_list = [
    ("https://media.giphy.com/media/JIX9t2j0ZTN9S/giphy.gif", "coding_cat.gif"),
    ("https://media.giphy.com/media/3oKIPnAiaMCws8nOsE/giphy.gif", "another_coding_cat.gif"),
] 

#
# Download all files in download_list
#
print()
for url,filename in dowload_list:
    with open(filename,'wb') as f:

        # python3
        # print("- Downloading file", filename, "from", url, "...", end="", flush=True) # Python3

        # python2
        print("- Downloading file", filename, "from", url, "...", end="")
        sys.stdout.flush()

        #
        # @todo Put this in a try,except block to catch download errors
        #
        f.write(urllib2.urlopen(url).read())
        f.close()
    print("done.")

print()
print("Done.")
print()
