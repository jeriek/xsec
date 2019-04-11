"""
Module containing a dictionary of features for processes keyed by pid
tuple pairs.
"""

from __future__ import print_function

import collections

import xsec.parameters as parameters

###############################################
# Global variables                            #
###############################################

FEATURES_LIST = {
    # --- Gluino--gluino
    (1000021, 1000021): [
        "m1000021",
        "m2000004",
        "m2000003",
        "m2000002",
        "m2000001",
        "m1000004",
        "m1000003",
        "m1000002",
        "m1000001",
        "mean",
    ],
    # --- Gluino--squark
    (1000021, 1000001): ["m1000021", "m1000001", "mean"],
    (1000021, 1000002): ["m1000021", "m1000002", "mean"],
    (1000021, 1000003): ["m1000021", "m1000003", "mean"],
    (1000021, 1000004): ["m1000021", "m1000004", "mean"],
    (1000021, 2000001): ["m1000021", "m2000001", "mean"],
    (1000021, 2000002): ["m1000021", "m2000002", "mean"],
    (1000021, 2000003): ["m1000021", "m2000003", "mean"],
    (1000021, 2000004): ["m1000021", "m2000004", "mean"],
    # --- Squark--squark
    (1000004, 1000004): ["m1000021", "m1000004", "mean"],
    (1000004, 1000003): ["m1000021", "m1000004", "m1000003", "mean"],
    (1000004, 1000001): ["m1000021", "m1000004", "m1000001", "mean"],
    (1000004, 1000002): ["m1000021", "m1000004", "m1000002", "mean"],
    (1000004, 2000002): ["m1000021", "m1000004", "m2000002", "mean"],
    (1000004, 2000001): ["m1000021", "m1000004", "m2000001", "mean"],
    (1000004, 2000003): ["m1000021", "m1000004", "m2000003", "mean"],
    (1000004, 2000004): ["m1000021", "m1000004", "m2000004", "mean"],
    (1000003, 1000003): ["m1000021", "m1000003", "mean"],
    (1000003, 1000001): ["m1000021", "m1000003", "m1000001", "mean"],
    (1000003, 1000002): ["m1000021", "m1000003", "m1000002", "mean"],
    (1000003, 2000002): ["m1000021", "m1000003", "m2000002", "mean"],
    (1000003, 2000001): ["m1000021", "m1000003", "m2000001", "mean"],
    (1000003, 2000003): ["m1000021", "m1000003", "m2000003", "mean"],
    (1000003, 2000004): ["m1000021", "m1000003", "m2000004", "mean"],
    (1000001, 1000001): ["m1000021", "m1000001", "mean"],
    (1000001, 1000002): ["m1000021", "m1000001", "m1000002", "mean"],
    (1000001, 2000002): ["m1000021", "m1000001", "m2000002", "mean"],
    (1000001, 2000001): ["m1000021", "m1000001", "m2000001", "mean"],
    (1000001, 2000003): ["m1000021", "m1000001", "m2000003", "mean"],
    (1000001, 2000004): ["m1000021", "m1000001", "m2000004", "mean"],
    (1000002, 1000002): ["m1000021", "m1000002", "mean"],
    (1000002, 2000002): ["m1000021", "m1000002", "m2000002", "mean"],
    (1000002, 2000001): ["m1000021", "m1000002", "m2000001", "mean"],
    (1000002, 2000003): ["m1000021", "m1000002", "m2000003", "mean"],
    (1000002, 2000004): ["m1000021", "m1000002", "m2000004", "mean"],
    (2000002, 2000002): ["m1000021", "m2000002", "mean"],
    (2000002, 2000001): ["m1000021", "m2000002", "m2000001", "mean"],
    (2000002, 2000003): ["m1000021", "m2000002", "m2000003", "mean"],
    (2000002, 2000004): ["m1000021", "m2000002", "m2000004", "mean"],
    (2000001, 2000001): ["m1000021", "m2000001", "mean"],
    (2000001, 2000003): ["m1000021", "m2000001", "m2000003", "mean"],
    (2000001, 2000004): ["m1000021", "m2000001", "m2000004", "mean"],
    (2000003, 2000003): ["m1000021", "m2000003", "mean"],
    (2000003, 2000004): ["m1000021", "m2000003", "m2000004", "mean"],
    (2000004, 2000004): ["m1000021", "m2000004", "mean"],
    # --- Squark--anti-squark
    (1000004, -1000004): ["m1000021", "m1000004", "mean"],
    (1000004, -1000003): ["m1000021", "m1000004", "m1000003", "mean"],
    (1000004, -1000001): ["m1000021", "m1000004", "m1000001", "mean"],
    (1000004, -1000002): ["m1000021", "m1000004", "m1000002", "mean"],
    (1000004, -2000002): ["m1000021", "m1000004", "m2000002", "mean"],
    (1000004, -2000001): ["m1000021", "m1000004", "m2000001", "mean"],
    (1000004, -2000003): ["m1000021", "m1000004", "m2000003", "mean"],
    (1000004, -2000004): ["m1000021", "m1000004", "m2000004", "mean"],
    (1000003, -1000003): ["m1000021", "m1000003", "mean"],
    (1000003, -1000001): ["m1000021", "m1000003", "m1000001", "mean"],
    (1000003, -1000002): ["m1000021", "m1000003", "m1000002", "mean"],
    (1000003, -2000002): ["m1000021", "m1000003", "m2000002", "mean"],
    (1000003, -2000001): ["m1000021", "m1000003", "m2000001", "mean"],
    (1000003, -2000003): ["m1000021", "m1000003", "m2000003", "mean"],
    (1000003, -2000004): ["m1000021", "m1000003", "m2000004", "mean"],
    (1000001, -1000001): ["m1000021", "m1000001", "mean"],
    (1000001, -1000002): ["m1000021", "m1000001", "m1000002", "mean"],
    (1000001, -2000002): ["m1000021", "m1000001", "m2000002", "mean"],
    (1000001, -2000001): ["m1000021", "m1000001", "m2000001", "mean"],
    (1000001, -2000003): ["m1000021", "m1000001", "m2000003", "mean"],
    (1000001, -2000004): ["m1000021", "m1000001", "m2000004", "mean"],
    (1000002, -1000002): ["m1000021", "m1000002", "mean"],
    (1000002, -2000002): ["m1000021", "m1000002", "m2000002", "mean"],
    (1000002, -2000001): ["m1000021", "m1000002", "m2000001", "mean"],
    (1000002, -2000003): ["m1000021", "m1000002", "m2000003", "mean"],
    (1000002, -2000004): ["m1000021", "m1000002", "m2000004", "mean"],
    (2000002, -2000002): ["m1000021", "m2000002", "mean"],
    (2000002, -2000001): ["m1000021", "m2000002", "m2000001", "mean"],
    (2000002, -2000003): ["m1000021", "m2000002", "m2000003", "mean"],
    (2000002, -2000004): ["m1000021", "m2000002", "m2000004", "mean"],
    (2000001, -2000001): ["m1000021", "m2000001", "mean"],
    (2000001, -2000003): ["m1000021", "m2000001", "m2000003", "mean"],
    (2000001, -2000004): ["m1000021", "m2000001", "m2000004", "mean"],
    (2000003, -2000003): ["m1000021", "m2000003", "mean"],
    (2000003, -2000004): ["m1000021", "m2000003", "m2000004", "mean"],
    (2000004, -2000004): ["m1000021", "m2000004", "mean"],
    # Sbottom--anti-sbottom
    (1000005, -1000005): ["m1000021", "m1000005", "sbotmix11", "mean"],
    (2000005, -2000005): ["m1000021", "m2000005", "sbotmix11", "mean"],
    # Stop--anti-stop
    (1000006, -1000006): ["m1000021", "m1000006", "stopmix11", "mean"],
    (2000006, -2000006): ["m1000021", "m2000006", "stopmix11", "mean"],
}


###############################################
# Get functions                               #
###############################################


def get_features(pid1, pid2):
    """
    Function that provides features for a PID pair. The order of the
    PIDs is irrelevant. The function will raise errors when keys are not
    found.
    """
    try:
        return FEATURES_LIST[(pid1, pid2)]
    except KeyError:
        try:
            return FEATURES_LIST[(pid2, pid1)]
        except KeyError:
            raise KeyError(
                "The entered process ({pid1}, {pid2}) is not in the list of "
                "allowed processes!".format(pid1=pid1, pid2=pid2)
            )


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
