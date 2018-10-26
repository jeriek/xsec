"""
Internal helper functions.
"""

import parameters

###############################################
# Global variables                            #
###############################################

# Link internal cross-section type (xstype) identifiers to the
# corresponding Nimbus file suffixes for each pre-trained xstype
XSTYPE_FILESUFFIX = {
    'centr': '',  # xsection @ central scale
    'sclup': '_2',  # xsection @ higher scale (2 x central scale)
    'scldn': '_05',  # xsection @ lower scale (0.5 x central scale)
    'pdf': '_pdf',  # xsection error due to pdf variation
    'aup': '_aup',  # xsection @ higher alpha_s
    'adn': '_adn'  # xsection @ lower alpha_s
}

# List of the internal xstype identifiers
XSTYPES = XSTYPE_FILESUFFIX.keys()


###############################################
# Helper functions                            #
###############################################

def get_processdir_name(process_xstype):
    # Get the partons of the process
    parton1, parton2, xstype = get_process_id_split(process_xstype)

    # Decide process name
    # Check if one of the partons is a gluino
    if parton1 == parameters.GLUINO_ID:
        processdir_name = str(parton1)+'_'+str(parton2)+'_NLO'
    elif parton2 == parameters.GLUINO_ID:
        processdir_name = str(parton2)+'_'+str(parton1)+'_NLO'

    # Otherwise name starts with the largest parton PID
    elif abs(parton1) >= abs(parton2):
        processdir_name = str(parton1)+'_'+str(parton2)+'_NLO'
    elif abs(parton1) < abs(parton2):
        processdir_name = str(parton2)+'_'+str(parton1)+'_NLO'

    # Add scale like '05','2' or other variation parameter like 'aup'
    try:
        processdir_name += XSTYPE_FILESUFFIX[xstype]
    except KeyError:
        print('Error: ', xstype, ' is not a valid variation parameter!')

    return processdir_name


def get_process_id(process, xstype):
    assert len(process) == 2
    process_xstype = (process[0], process[1], xstype)

    return process_xstype


def get_process_id_str(process, xstype):
    assert len(process) == 2
    process_xstype_str = (str(process[0]) + '_' + str(process[1]) + '_'
                          + xstype)

    return process_xstype_str


def get_process_id_split(process_xstype):
    assert len(process_xstype) == 3
    parton1 = process_xstype[0]
    parton2 = process_xstype[1]
    xstype = process_xstype[2]

    return parton1, parton2, xstype


def get_process_from_process_id(process_xstype):
    parton1 = process_xstype[0]
    parton2 = process_xstype[1]

    return (parton1, parton2)


def get_xstype_from_process_id(process_xstype):
    xstype = process_xstype[-1]

    return xstype


def get_process_list_str(process_list):
    process_str_list = []
    for process in process_list:
        process_str = str(process[0]) + '_' + str(process[1])
        process_str_list.append(process_str)

    return process_str_list
