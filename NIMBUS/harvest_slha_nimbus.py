#! /usr/bin/python

######################################
#                                    #
# Program for collecting information #
# from a set of SLHA files           #
#                                    #
######################################
#
# Usage:
# ------
#     python harvest_slha.py [output_file] [directory_to_search] [file_tag]
#
#
# Details:
# --------
# The program will only include files that have [file_tag] as part of the filename.
# If [file_tag] is set to '', all files in the search directory are included
#
#

#from mpi4py import MPI
import os
import sys
from modules import pyslha
from collections import OrderedDict


###########################
#  What data to collect?  #
###########################

# Should this really be ordered? Try to change
datadict = OrderedDict ([])

# Squark masses
datadict['m1000021']  = {'block': 'MASS',     'element': (1000021,1),   'abs' : True}
datadict['m1000004'] = {'block' : 'MASS', 'element': (1000004, 1), 'abs' : True}
datadict['m1000003']= {'block': 'MASS',     'element': (1000003,1),   'abs' : True}
datadict['m1000001'] = {'block': 'MASS',     'element': (1000001,1),   'abs' : True}
datadict['m1000002'] = {'block': 'MASS',     'element': (1000002,1),   'abs' : True}
datadict['m2000002']= {'block': 'MASS',     'element': (2000002,1),   'abs' : True}
datadict['m2000001'] = {'block': 'MASS',     'element': (2000001,1),   'abs' : True}
datadict['m2000003']= {'block': 'MASS',     'element': (2000003,1),   'abs' : True}
datadict['m2000004']= {'block': 'MASS',     'element': (2000004,1),   'abs' : True}

# gluino-gluino cross section

datadict['1000021_1000021_NLO']       = {'block': 'PROSPINO_OUTPUT', 'element': ((1000021, 1000021, 8000),7)}

# squark-squark cross sections

squarks = ['1000004', '1000003', '1000001', '1000002', '2000002', '2000001', '2000003', '2000004']
gluino = '1000021'

for i in range(len(squarks)):
    for j in range(i, len(squarks)):
        name = str(squarks[i])+'_'+str(squarks[j])+'_NLO'
        datadict[name]       = {'block': 'PROSPINO_OUTPUT', 'element': ((int(squarks[i]), int(squarks[j]), 8000),7)}

# squark-gluino cross sections

for i in range(len(squarks)):
    name = str(1000021)+'_'+str(squarks[i])+'_NLO'
    datadict[name]       = {'block': 'PROSPINO_OUTPUT', 'element': ((int(1000021), int(squarks[i]), 8000),7)}

# squark-antisquark cross sections

for i in range(len(squarks)):
    for j in range(i, len(squarks)):
        name = str(squarks[i])+'_-'+str(squarks[j])+'_NLO'
        datadict[name]       = {'block': 'PROSPINO_OUTPUT', 'element': ((int(squarks[i]), - int(squarks[j]), 8000),7)}




##############################
#  Initial setup and checks  #
##############################

# set output prefix
outpref = sys.argv[0] + ' : '

# check input arguments:
if len(sys.argv) < 4:
    sys.stdout.write("%s Missing input arguments \n" % (outpref))
    sys.exit()
for arg in sys.argv[1:]:
    if type(arg) != str:
        sys.stdout.write("%s Wrong input format \n" % (outpref))
        sys.exit()

# assign input arguments to variables    
outfile = sys.argv[1]
filetag = sys.argv[-1]
if isinstance(sys.argv[2:-1], list):
    filedirs = sys.argv[2:-1]
elif isinstance(sys.argv[2:-1], str):
    filedirs = [ sys.argv[2:-1] ]
else:
    sys.stdout.write("%s Problem with given input directories \n" % (outpref))
    sys.exit()
    

# print info
sys.stdout.write("%s Searching in directories %s for files with names containing '%s'.\n" % (outpref,filedirs,filetag))



#####################################
#  File search and data collection  #
#####################################

# store all requested files in a dict of lists
# (one dict entry per search directory) 
filedict = dict((key,[]) for key in filedirs)
for directory in filedict.keys():
    for filename in os.listdir(directory):
        if filetag in filename:
            filedict[directory].append(filename)

# output file count
n_files_total = 0
for directory in filedict.keys():
    n_files = len(filedict[directory])
    sys.stdout.write("%s Found %d files in %s\n" % (outpref, n_files, directory))
    n_files_total += n_files
sys.stdout.write("%s In total %d files\n" % (outpref, n_files_total))


# sort the list according to first letter/number in filename
# ...should make this more sofisticated...
for directory in filedict.keys():
    filedict[directory].sort()

# output info
sys.stdout.write("%s Collecting data...\n" % (outpref))

# open outfile for output
f = open(outfile, "w")



# add tag for filename columns
tagline = ''
tag = '1.file'
tagline += ( tag+' '*(max(1,25-len(tag))) ) 

for i,tag in enumerate(datadict.keys()): # IH: Remove the tag for masses and xsections
    n = i+2
    complete_tag = tag # IH
    #complete_tag = '%i.%s' % (n,tag)
    tagline += ( complete_tag+' '*(max(1,25-len(complete_tag))) ) 


tagline += '\n'
print 'The tagline, ', tagline
f.write(tagline)

# collect data from each file and write to outfile
count = 0
lines = ''
for directory in filedict.keys():
    for filename in filedict[directory]:
        count += 1
        filepath = os.path.join(directory,filename)

        slha_dict = pyslha.readSLHAFile(filepath)
        
        datalist = []

        accepted_file = True
        number_of_unaccepted_sxections = 0 #IH

        for key in datadict.keys():

  
            if 'block' in datadict[key].keys():

                if not datadict[key]['block'] in slha_dict.keys():
                    accepted_file = False
                    sys.stdout.write("%s Problem encountered when looking for block %s in file %s. File ignored.\n" % (outpref, datadict[key]['block'], filepath))
                    break

                

                if ('abs' in datadict[key].keys()) and (datadict[key]['abs'] == True):


                    datalist.append( abs( slha_dict[ datadict[key]['block'] ].getElement( *datadict[key]['element'] ) ) )
                else:
                    
                    try: 
                        slha_dict[ datadict[key]['block'] ].getElement( *datadict[key]['element'] ) #IH
                        #print slha_dict[ datadict[key]['block'] ].getElement( *datadict[key]['element'] )
                        datalist.append( slha_dict[ datadict[key]['block'] ].getElement( *datadict[key]['element'] ) )
                    except: 
                        number_of_unaccepted_sxections += 1
                        datalist.append(-1)



            elif 'decay' in datadict[key].keys():

                if not datadict[key]['decay'] in slha_dict.keys():
                    accepted_file = False
                    sys.stdout.write("%s Problem encountered when looking for decay %s in file %s. File ignored.\n" % (outpref, datadict[key]['decay'], filepath))
                    break

                datalist.append( slha_dict[ datadict[key]['decay'] ].getBR( list( datadict[key]['element'] ) ) )


            # got_it = True
            # try:
            #     datalist.append( slha_dict[ datadict[key]['block'] ].getElement( *datadict[key]['element'] ) )

            # except Exception as err:
            #     sys.stdout.write("%s %s \n" % (outpref, err.message))
            #     sys.stdout.write("%s Problem encountered when harvesting data from file %s. File ignored.\n" % (outpref, filepath))
            #     continue

        if not accepted_file:
            continue


        datatuple = tuple(datalist)
        lines += ('%s' + ' '*(max(1,24-len(filename))) ) % filename
        lines += ( ('% .8e' + ' '*10)*len(datatuple) + '\n') % datatuple

        # write to file once per 1000 files read
        if count%1000 == 0:
            sys.stdout.write("%s %d of %d files read\n" % (outpref, count, n_files_total))
            f.write(lines)
            lines = ''


# Remove final endline and write remaining lines
lines = lines.rstrip('\n')
f.write(lines)

##############
#  Finalize  #
##############

# output info
sys.stdout.write("%s ...done!\n" % (outpref))

# close the outfile
f.close()

# print some output
sys.stdout.write("%s Summary written to the file %s \n" % (outpref, outfile))
