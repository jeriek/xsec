"""
Module containing a dictionary of features for processes keyed by
particle ID (PID) tuple pairs.
"""

from __future__ import print_function

import collections

import xsec.utils as utils
import xsec.parameters as parameters
from xsec.parameters import (GLUINO_ID, ALL_SQUARK_IDS, ALL_GEN3_IDS,
                             EWINO_IDS)

###############################################
# Global variables                            #
###############################################

# fmt: off
FEATURES_LIST = {
    # Process identification key: PID tuple (sorted from low to high)

    # --- Gluino--gluino
    # [Order: gluino, squarks from low to high PID, mean squark mass]
    (1000021, 1000021): [
        "m1000021",
        "m2000004", "m2000003", "m2000002", "m2000001",
        "m1000004", "m1000003", "m1000002", "m1000001",
        "mean",
    ],
    # --- Gluino--squark
    # [Order: gluino, squark, mean]
    (1000001, 1000021): ["m1000021", "m1000001", "mean"],
    (1000002, 1000021): ["m1000021", "m1000002", "mean"],
    (1000003, 1000021): ["m1000021", "m1000003", "mean"],
    (1000004, 1000021): ["m1000021", "m1000004", "mean"],
    (1000021, 2000001): ["m1000021", "m2000001", "mean"],
    (1000021, 2000002): ["m1000021", "m2000002", "mean"],
    (1000021, 2000003): ["m1000021", "m2000003", "mean"],
    (1000021, 2000004): ["m1000021", "m2000004", "mean"],
    # --- Squark--squark
    # [Order: gluino, squarks from low to high PID, mean squark mass]
    (1000001, 1000001): ["m1000021", "m1000001", "mean"],
    (1000001, 1000002): ["m1000021", "m1000001", "m1000002", "mean"],
    (1000001, 1000003): ["m1000021", "m1000001", "m1000003", "mean"],
    (1000001, 1000004): ["m1000021", "m1000001", "m1000004", "mean"],
    (1000001, 2000001): ["m1000021", "m1000001", "m2000001", "mean"],
    (1000001, 2000002): ["m1000021", "m1000001", "m2000002", "mean"],
    (1000001, 2000003): ["m1000021", "m1000001", "m2000003", "mean"],
    (1000001, 2000004): ["m1000021", "m1000001", "m2000004", "mean"],
    (1000002, 1000002): ["m1000021", "m1000002", "mean"],
    (1000002, 1000003): ["m1000021", "m1000002", "m1000003", "mean"],
    (1000002, 1000004): ["m1000021", "m1000002", "m1000004", "mean"],
    (1000002, 2000001): ["m1000021", "m1000002", "m2000001", "mean"],
    (1000002, 2000002): ["m1000021", "m1000002", "m2000002", "mean"],
    (1000002, 2000003): ["m1000021", "m1000002", "m2000003", "mean"],
    (1000002, 2000004): ["m1000021", "m1000002", "m2000004", "mean"],
    (1000003, 1000003): ["m1000021", "m1000003", "mean"],
    (1000003, 1000004): ["m1000021", "m1000003", "m1000004", "mean"],
    (1000003, 2000001): ["m1000021", "m1000003", "m2000001", "mean"],
    (1000003, 2000002): ["m1000021", "m1000003", "m2000002", "mean"],
    (1000003, 2000003): ["m1000021", "m1000003", "m2000003", "mean"],
    (1000003, 2000004): ["m1000021", "m1000003", "m2000004", "mean"],
    (1000004, 1000004): ["m1000021", "m1000004", "mean"],
    (1000004, 2000001): ["m1000021", "m1000004", "m2000001", "mean"],
    (1000004, 2000002): ["m1000021", "m1000004", "m2000002", "mean"],
    (1000004, 2000003): ["m1000021", "m1000004", "m2000003", "mean"],
    (1000004, 2000004): ["m1000021", "m1000004", "m2000004", "mean"],
    (2000001, 2000001): ["m1000021", "m2000001", "mean"],
    (2000001, 2000002): ["m1000021", "m2000001", "m2000002", "mean"],
    (2000001, 2000003): ["m1000021", "m2000001", "m2000003", "mean"],
    (2000001, 2000004): ["m1000021", "m2000001", "m2000004", "mean"],
    (2000002, 2000002): ["m1000021", "m2000002", "mean"],
    (2000002, 2000003): ["m1000021", "m2000002", "m2000003", "mean"],
    (2000002, 2000004): ["m1000021", "m2000002", "m2000004", "mean"],
    (2000003, 2000003): ["m1000021", "m2000003", "mean"],
    (2000003, 2000004): ["m1000021", "m2000003", "m2000004", "mean"],
    (2000004, 2000004): ["m1000021", "m2000004", "mean"],
    # --- Squark--anti-squark
    # [Order: gluino, squarks from low to high PID, mean]
    (-1000001, 1000001): ["m1000021", "m1000001", "mean"],
    (-1000001, 1000002): ["m1000021", "m1000001", "m1000002", "mean"],
    (-1000001, 1000003): ["m1000021", "m1000001", "m1000003", "mean"],
    (-1000001, 1000004): ["m1000021", "m1000001", "m1000004", "mean"],
    (-1000001, 2000001): ["m1000021", "m1000001", "m2000001", "mean"],
    (-1000001, 2000002): ["m1000021", "m1000001", "m2000002", "mean"],
    (-1000001, 2000003): ["m1000021", "m1000001", "m2000003", "mean"],
    (-1000001, 2000004): ["m1000021", "m1000001", "m2000004", "mean"],
    (-1000002, 1000002): ["m1000021", "m1000002", "mean"],
    (-1000002, 1000003): ["m1000021", "m1000002", "m1000003", "mean"],
    (-1000002, 1000004): ["m1000021", "m1000002", "m1000004", "mean"],
    (-1000002, 2000001): ["m1000021", "m1000002", "m2000001", "mean"],
    (-1000002, 2000002): ["m1000021", "m1000002", "m2000002", "mean"],
    (-1000002, 2000003): ["m1000021", "m1000002", "m2000003", "mean"],
    (-1000002, 2000004): ["m1000021", "m1000002", "m2000004", "mean"],
    (-1000003, 1000003): ["m1000021", "m1000003", "mean"],
    (-1000003, 1000004): ["m1000021", "m1000003", "m1000004", "mean"],
    (-1000003, 2000001): ["m1000021", "m1000003", "m2000001", "mean"],
    (-1000003, 2000002): ["m1000021", "m1000003", "m2000002", "mean"],
    (-1000003, 2000003): ["m1000021", "m1000003", "m2000003", "mean"],
    (-1000003, 2000004): ["m1000021", "m1000003", "m2000004", "mean"],
    (-1000004, 1000004): ["m1000021", "m1000004", "mean"],
    (-1000004, 2000001): ["m1000021", "m1000004", "m2000001", "mean"],
    (-1000004, 2000002): ["m1000021", "m1000004", "m2000002", "mean"],
    (-1000004, 2000003): ["m1000021", "m1000004", "m2000003", "mean"],
    (-1000004, 2000004): ["m1000021", "m1000004", "m2000004", "mean"],
    (-2000001, 2000001): ["m1000021", "m2000001", "mean"],
    (-2000001, 2000002): ["m1000021", "m2000001", "m2000002", "mean"],
    (-2000001, 2000003): ["m1000021", "m2000001", "m2000003", "mean"],
    (-2000001, 2000004): ["m1000021", "m2000001", "m2000004", "mean"],
    (-2000002, 2000002): ["m1000021", "m2000002", "mean"],
    (-2000002, 2000003): ["m1000021", "m2000002", "m2000003", "mean"],
    (-2000002, 2000004): ["m1000021", "m2000002", "m2000004", "mean"],
    (-2000003, 2000003): ["m1000021", "m2000003", "mean"],
    (-2000003, 2000004): ["m1000021", "m2000003", "m2000004", "mean"],
    (-2000004, 2000004): ["m1000021", "m2000004", "mean"],
    # --- Sbottom--anti-sbottom
    # [Order: gluino, squark, mean, mixing]
    (-1000005, 1000005): ["m1000021", "m1000005", "mean", "sbotmix11"],
    (-2000005, 2000005): ["m1000021", "m2000005", "mean", "sbotmix11"],
    # --- Stop--anti-stop
    # [Order: gluino, squark, mean, mixing]
    (-1000006, 1000006): ["m1000021", "m1000006", "mean", "stopmix11"],
    (-2000006, 2000006): ["m1000021", "m2000006", "mean", "stopmix11"],
    # --- Neutralino and chargino production
    # [Order: neutralinos, gluino, squarks (low to high PIDs), mean, mixings]
    (1000023, 1000023): ["m1000023",
                         "m1000021",
                         "m1000001", "m1000002", "m1000003", "m1000004",
                         "m2000001", "m2000002", "m2000003", "m2000004",
                         "mean",
                         "nmix21", "nmix22", "nmix23", "nmix24"],
    (1000022, 1000023): ["m1000022", "m1000023",
                         "m1000021",
                         "m1000001", "m1000002", "m1000003", "m1000004",
                         "m2000001", "m2000002", "m2000003", "m2000004",
                         "mean",
                         "nmix11", "nmix12", "nmix13", "nmix14",
                         "nmix21", "nmix22", "nmix23", "nmix24"],
    (1000022, 1000022): ["m1000022",
                         "m1000021",
                         "m1000001", "m1000002", "m1000003", "m1000004",
                         "m2000001", "m2000002", "m2000003", "m2000004",
                         "mean",
                         "nmix11", "nmix12", "nmix13", "nmix14"],
}
# fmt: on
TRAINED_PROCESSES = FEATURES_LIST.keys()

###############################################
# Get functions                               #
###############################################


def get_features(pid1, pid2):
    """
    Function that provides features for a PID pair. The order of the
    PIDs is irrelevant, and will return the charge conjugate process if
    that exists instead.

    The function will raise an error if the key is not found.
    """

    try:
        return FEATURES_LIST[get_trained_process(pid1, pid2)]
    except KeyError:
        raise KeyError(
            "The entered process ({pid1}, {pid2}) is not in the "
            "list of allowed processes!".format(
                pid1=pid1, pid2=pid2
                )
        )
    # TODO: remove _if_ no longer needed!
    #     return FEATURES_LIST[(pid1, pid2)]
    #     try:
    #         return FEATURES_LIST[(pid2, pid1)]
    #     except KeyError:
    #         try:
    #             return FEATURES_LIST[(get_cc(pid1), get_cc(pid2))]
    #         except KeyError:
    #             try:
    #                 return FEATURES_LIST[(get_cc(pid2), get_cc(pid1))]
    #             except KeyError:
    #                 raise KeyError(
    #                     "The entered process ({pid1}, {pid2}) is not in the "
    #                     "list of allowed processes!".format(
    #                         pid1=pid1, pid2=pid2
    #                         )
    #                 )


def get_features_dict(process_list):
    """
    Produce a dictionary of processes and values for their features, for
    each process in process_list.
    """
    # Dictionary {process : [ordered feature values list]}
    all_features_dict = {}

    # Loop over list of processes we want features for
    for process in process_list:
        assert len(process) == 2
        # Find features for this process (Note: the features in this
        # list have a specific order, which must be the same as used
        # during training ... the X_train data are ordered!)
        features_index = get_features(*process)

        # Make a feature dictionary
        # features_dict = {key : PARAMS[key] for key in features_index}
        features_dict = collections.OrderedDict()
        for param in features_index:
            features_dict[param] = parameters.get_parameter(param)

        # Ordered list since derived from ordered dict!
        features = features_dict.values()

        # Add features to feature array
        all_features_dict.update({process: features})

    # Return feature dict for all processes
    return all_features_dict


def get_unique_features(process_list):
    """
    Return a list of required, unique features for a list of processes.
    The features in the returned list have no specific order, so the
    function should only be used when the order is of no importance,
    e.g. when checking the consistency of each parameter individually.
    """
    # List to be filled with (non-duplicate) features
    feature_list = []
    # Loop over list of processes we want features for
    for process in process_list:
        assert len(process) == 2
        features = get_features(*process)
        # Add to feature list without duplicates
        feature_list = list(set(feature_list + features))
    return feature_list


def get_cc(pid):
    """
    Returns the charge conjugate of a PID code or the same code if no
    charge conjugate exists.
    """
    if abs(pid) in parameters.SQUARK_IDS:
        return -pid
    elif pid in parameters.CHARGINO_IDS:
        return -pid
    else:
        return pid


def sort_pids(pid1, pid2):
    """
    Returns the PID tuple sorted from lowest to highest numerical value.
    """
    input_pids = (pid1, pid2)
    sorted_pids = (min(input_pids), max(input_pids))
    return sorted_pids


def get_trained_process(pid1, pid2):
    """
    Find the process that has actually been trained, given the PID pair
    (tuple) of the final state. Returns the ordered PID pair (tuple) of
    the process for which training data is available (see full list in
    FEATURES_LIST), possibly the charge-conjugate of the input process
    or a process with different chiralities, nevertheless guaranteed by
    symmetries to have the same cross section.

    Used in features.get_features() and gploader.set_processes().
    Largely replaces utils.get_processdir_name() by placing the right
    PIDs in the right order matching the trained process already.

    """
    # Sort PIDs, lowest first
    sorted_pids = sort_pids(pid1, pid2)

    # Try to find it in the process list, else try charge-conjugate
    if sorted_pids in TRAINED_PROCESSES:
        return sorted_pids
    else:
        sorted_pids_cc = sort_pids(*get_cc(sorted_pids))
        if sorted_pids_cc in TRAINED_PROCESSES:
            return sorted_pids_cc
        else:
            raise utils.unknown_process_error(pid1, pid2)

    # TODO: update Nimbus side
    # TODO: remove
    # # - Processes with one or more gluinos
    # if GLUINO_ID in input_pids:
    #     # -- ~g ~g (only 1 process)
    #     if pid1 == pid2:
    #         return (pid1, pid2)
    #     # -- ~g ~t/b(*)
    #     elif pid1 in ALL_GEN3_IDS or pid2 in ALL_GEN3_IDS:
    #         raise utils.unknown_process_error(pid1, pid2)
    #     # -- ~g ~q(*) (not 3rd gen.)
    #     elif pid1 in ALL_SQUARK_IDS:
    #         # ~g always first, same output as for c.c.
    #         return (pid2, abs(pid1))
    #     elif pid2 in ALL_SQUARK_IDS:
    #         # ~g always first, same output as for c.c.
    #         return (pid1, abs(pid2))
    #     else:
    #         raise utils.unknown_process_error(pid1, pid2)
    # # - Processes with 3rd gen. squarks
    # elif pid1 in ALL_GEN3_IDS or pid2 in ALL_GEN3_IDS:
    #     # -- ~b ~b* and ~t ~t* only
    #     if pid1 == -pid2:
    #         return (max(pid1, pid2), min(pid1, pid2))
    #     else:
    #         raise utils.unknown_process_error(pid1, pid2)
    # # - Processes with squarks and/or antisquarks (not 3rd gen.)
    # elif pid1 in ALL_SQUARK_IDS and pid2 in ALL_SQUARK_IDS:
    #     # -- ~q ~q != ~q* ~q*; always sum of c.c. xsections returned
    #     if pid1*pid2 > 0:
    #         return (max(abs(pid1), abs(pid2)), min(abs(pid1), abs(pid2)))
    #     # -- ~q ~q*; always sum of c.c. xsections returned
    #     else:
    #         return (max(abs(pid1), abs(pid2)), -min(abs(pid1), abs(pid2)))
    # # - Processes with neutralinos and/or charginos
    # elif pid1 in EWINO_IDS and pid2 in EWINO_IDS:
    #     return (max(abs(pid1), abs(pid2)), min(abs(pid1), abs(pid2)))
    # # All other combinations
    # raise utils.unknown_process_error(pid1, pid2)
