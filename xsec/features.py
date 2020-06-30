"""
Module containing a dictionary of features for processes keyed by
particle ID (PID) tuple pairs, and functions to access it.
"""

from __future__ import print_function

import collections

import xsec.utils as utils
import xsec.parameters as parameters
import xsec.gploader as gploader


###############################################
# Global variables                            #
###############################################

# fmt: off
FEATURES_LIST = {
    # Process identification key: PID tuple
    # - sorted in ascending order
    # - start with most negative PID if ~q~q*: (-1000004, 1000003)
    # --- Gluino--gluino
    # [Order: gluino, squarks (ascending PID), mean 1st/2nd gen. squark mass]
    (1000021, 1000021): [
        "m1000021",
        "m1000001", "m1000002", "m1000003", "m1000004",
        "m2000001", "m2000002", "m2000003", "m2000004",
        "mean",
    ],
    # --- Gluino--squark
    # [Order: gluino, squark, mean 1st/2nd gen. squark mass]
    (1000001, 1000021): ["m1000021", "m1000001", "mean"],
    (1000002, 1000021): ["m1000021", "m1000002", "mean"],
    (1000003, 1000021): ["m1000021", "m1000003", "mean"],
    (1000004, 1000021): ["m1000021", "m1000004", "mean"],
    (1000021, 2000001): ["m1000021", "m2000001", "mean"],
    (1000021, 2000002): ["m1000021", "m2000002", "mean"],
    (1000021, 2000003): ["m1000021", "m2000003", "mean"],
    (1000021, 2000004): ["m1000021", "m2000004", "mean"],
    # --- Squark--squark
    # [Order: gluino, squarks (d/u/s/c, L/R), mean 1st/2nd gen. squark mass]
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
    (1000002, 2000001): ["m1000021", "m2000001", "m1000002", "mean"],
    (1000002, 2000002): ["m1000021", "m1000002", "m2000002", "mean"],
    (1000002, 2000003): ["m1000021", "m1000002", "m2000003", "mean"],
    (1000002, 2000004): ["m1000021", "m1000002", "m2000004", "mean"],
    (1000003, 1000003): ["m1000021", "m1000003", "mean"],
    (1000003, 1000004): ["m1000021", "m1000003", "m1000004", "mean"],
    (1000003, 2000001): ["m1000021", "m2000001", "m1000003", "mean"],
    (1000003, 2000002): ["m1000021", "m2000002", "m1000003", "mean"],
    (1000003, 2000003): ["m1000021", "m1000003", "m2000003", "mean"],
    (1000003, 2000004): ["m1000021", "m1000003", "m2000004", "mean"],
    (1000004, 1000004): ["m1000021", "m1000004", "mean"],
    (1000004, 2000001): ["m1000021", "m2000001", "m1000004", "mean"],
    (1000004, 2000002): ["m1000021", "m2000002", "m1000004", "mean"],
    (1000004, 2000003): ["m1000021", "m2000003", "m1000004", "mean"],
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
    # [Order: gluino, squarks (d/u/s/c, L/R), mean 1st/2nd gen. squark mass]
    (-2000004, 2000004): ["m1000021", "m2000004", "mean"],
    (-2000004, 2000003): ["m1000021", "m2000003", "m2000004", "mean"],
    (-2000004, 2000002): ["m1000021", "m2000002", "m2000004", "mean"],
    (-2000004, 2000001): ["m1000021", "m2000001", "m2000004", "mean"],
    (-2000004, 1000004): ["m1000021", "m1000004", "m2000004", "mean"],
    (-2000004, 1000003): ["m1000021", "m1000003", "m2000004", "mean"],
    (-2000004, 1000002): ["m1000021", "m1000002", "m2000004", "mean"],
    (-2000004, 1000001): ["m1000021", "m1000001", "m2000004", "mean"],
    (-2000003, 2000003): ["m1000021", "m2000003", "mean"],
    (-2000003, 2000002): ["m1000021", "m2000002", "m2000003", "mean"],
    (-2000003, 2000001): ["m1000021", "m2000001", "m2000003", "mean"],
    (-2000003, 1000004): ["m1000021", "m1000004", "m2000003", "mean"],
    (-2000003, 1000003): ["m1000021", "m1000003", "m2000003", "mean"],
    (-2000003, 1000002): ["m1000021", "m1000002", "m2000003", "mean"],
    (-2000003, 1000001): ["m1000021", "m1000001", "m2000003", "mean"],
    (-2000002, 2000002): ["m1000021", "m2000002", "mean"],
    (-2000002, 2000001): ["m1000021", "m2000001", "m2000002", "mean"],
    (-2000002, 1000004): ["m1000021", "m1000004", "m2000002", "mean"],
    (-2000002, 1000003): ["m1000021", "m1000003", "m2000002", "mean"],
    (-2000002, 1000002): ["m1000021", "m1000002", "m2000002", "mean"],
    (-2000002, 1000001): ["m1000021", "m1000001", "m2000002", "mean"],
    (-2000001, 2000001): ["m1000021", "m2000001", "mean"],
    (-2000001, 1000004): ["m1000021", "m1000004", "m2000001", "mean"],
    (-2000001, 1000003): ["m1000021", "m1000003", "m2000001", "mean"],
    (-2000001, 1000002): ["m1000021", "m1000002", "m2000001", "mean"],
    (-2000001, 1000001): ["m1000021", "m1000001", "m2000001", "mean"],
    (-1000004, 1000004): ["m1000021", "m1000004", "mean"],
    (-1000004, 1000003): ["m1000021", "m1000003", "m1000004", "mean"],
    (-1000004, 1000002): ["m1000021", "m1000002", "m1000004", "mean"],
    (-1000004, 1000001): ["m1000021", "m1000001", "m1000004", "mean"],
    (-1000003, 1000003): ["m1000021", "m1000003", "mean"],
    (-1000003, 1000002): ["m1000021", "m1000002", "m1000003", "mean"],
    (-1000003, 1000001): ["m1000021", "m1000001", "m1000003", "mean"],
    (-1000002, 1000002): ["m1000021", "m1000002", "mean"],
    (-1000002, 1000001): ["m1000021", "m1000001", "m1000002", "mean"],
    (-1000001, 1000001): ["m1000021", "m1000001", "mean"],
    # --- Sbottom--anti-sbottom
    # [Order: gluino, squark, mean 1st/2nd gen. squark mass, mixing]
    (-1000005, 1000005): [
        "m1000021", "m1000005", "m2000005", "mean", "sbotmix11",
        ],
    (-2000005, 2000005): [
        "m1000021", "m1000005", "m2000005", "mean", "sbotmix11",
        ],
    # --- Stop--anti-stop
    # [Order: gluino, squark, mean 1st/2nd gen. squark mass, mixing]
    (-1000006, 1000006): [
        "m1000021", "m1000006", "m2000006", "mean", "stopmix11",
        ],
    (-2000006, 2000006): [
        "m1000021", "m1000006", "m2000006", "mean", "stopmix11",
        ],
    # --- Neutralino and chargino production
    # [Order: neutralinos, gluino, squarks (ascending PID), mean
    # 1st/2nd gen. squark mass, mixings]
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

# Keep track of which processes have not yet been validated.
# These will not be returned by the utils.list_all_xsec_processes() function
UNVALIDATED_PROCESSES = [
    # --- Neutralino and chargino production
    (1000023, 1000023),
    (1000022, 1000023),
    (1000022, 1000022),
]

# fmt: on
TRAINED_PROCESSES = list(FEATURES_LIST.keys())

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
            "list of allowed processes!".format(pid1=pid1, pid2=pid2)
        )


def get_features_dict(process_list, auto_normf=True):
    """
    Produce a dictionary of processes and values for their features, for
    each process in process_list. If the auto_normf flag is set to True,
    the NORMF flag in the transform modules decides whether normalised
    features should be returned. If False, no normalisation occurs.
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

        # Check if features were normalised before training
        # This assumes features are normalised for all xstypes
        if auto_normf:
            try:
                feature_norm_flag = gploader.TRANSFORM_MODULES[
                    (process, "centr")
                ].NORMF
            except AttributeError:  # Kept for backward compatibility
                feature_norm_flag = False
        else:
            feature_norm_flag = False

        # Make a feature dictionary
        # features_dict = {key : PARAMS[key] for key in features_index}
        features_dict = collections.OrderedDict()
        for param in features_index:
            # Check if features should be normalised
            if feature_norm_flag:
                features_dict[param] = parameters.get_normalised_parameter(
                    param
                )
            else:
                features_dict[param] = parameters.get_parameter(param)

        # Ordered list since derived from ordered dict!
        features = list(features_dict.values())

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
    FEATURES_LIST), possibly the charge-conjugate of the input process.
    The PID pair is sorted in ascending order.
    """
    # Sort PIDs, lowest first
    sorted_pids = sort_pids(pid1, pid2)

    # Try to find it in the process list, else try charge-conjugate
    if sorted_pids in TRAINED_PROCESSES:
        return sorted_pids
    else:
        sorted_pids_cc = sort_pids(get_cc(pid1), get_cc(pid2))
        if sorted_pids_cc in TRAINED_PROCESSES:
            return sorted_pids_cc
        else:
            raise utils.unknown_process_error(pid1, pid2)
