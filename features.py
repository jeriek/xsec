# Module containing a dictionary of features for processes keyed by pid tuple pairs
import collections
from parameters import PARAMS, set_mean_mass

FEATURES_LIST = {
    # --- Gluino--gluino
    (1000021, 1000021): ['m1000021', 'm2000004', 'm2000003',
                         'm2000002', 'm2000001', 'm1000004',
                         'm1000003', 'm1000002', 'm1000001', 'mean'],

    # --- Gluino--squark
    (1000021, 1000001): ['m1000021', 'm1000001', 'mean'],
    (1000021, 1000002): ['m1000021', 'm1000002', 'mean'],
    (1000021, 1000003): ['m1000021', 'm1000003', 'mean'],
    (1000021, 1000004): ['m1000021', 'm1000004', 'mean'],
    (1000021, 2000001): ['m1000021', 'm2000001', 'mean'],
    (1000021, 2000002): ['m1000021', 'm2000002', 'mean'],
    (1000021, 2000003): ['m1000021', 'm2000003', 'mean'],
    (1000021, 2000004): ['m1000021', 'm2000004', 'mean'],

    # --- Squark--squark
    (1000004, 1000004): ['m1000021', 'm1000004', 'mean'],
    (1000004, 1000003): ['m1000021', 'm1000004', 'm1000003', 'mean'],
    (1000004, 1000001): ['m1000021', 'm1000004', 'm1000001', 'mean'],
    (1000004, 1000002): ['m1000021', 'm1000004', 'm1000002', 'mean'],
    (1000004, 2000002): ['m1000021', 'm1000004', 'm2000002', 'mean'],
    (1000004, 2000001): ['m1000021', 'm1000004', 'm2000001', 'mean'],
    (1000004, 2000003): ['m1000021', 'm1000004', 'm2000003', 'mean'],
    (1000004, 2000004): ['m1000021', 'm1000004', 'm2000004', 'mean'],
    (1000003, 1000003): ['m1000021', 'm1000003', 'mean'],
    (1000003, 1000001): ['m1000021', 'm1000003', 'm1000001', 'mean'],
    (1000003, 1000002): ['m1000021', 'm1000003', 'm1000002', 'mean'],
    (1000003, 2000002): ['m1000021', 'm1000003', 'm2000002', 'mean'],
    (1000003, 2000001): ['m1000021', 'm1000003', 'm2000001', 'mean'],
    (1000003, 2000003): ['m1000021', 'm1000003', 'm2000003', 'mean'],
    (1000003, 2000004): ['m1000021', 'm1000003', 'm2000004', 'mean'],
    (1000001, 1000001): ['m1000021', 'm1000001', 'mean'],
    (1000001, 1000002): ['m1000021', 'm1000001', 'm1000002', 'mean'],
    (1000001, 2000002): ['m1000021', 'm1000001', 'm2000002', 'mean'],
    (1000001, 2000001): ['m1000021', 'm1000001', 'm2000001', 'mean'],
    (1000001, 2000003): ['m1000021', 'm1000001', 'm2000003', 'mean'],
    (1000001, 2000004): ['m1000021', 'm1000001', 'm2000004', 'mean'],
    (1000002, 1000002): ['m1000021', 'm1000002', 'mean'],
    (1000002, 2000002): ['m1000021', 'm1000002', 'm2000002', 'mean'],
    (1000002, 2000001): ['m1000021', 'm1000002', 'm2000001', 'mean'],
    (1000002, 2000003): ['m1000021', 'm1000002', 'm2000003', 'mean'],
    (1000002, 2000004): ['m1000021', 'm1000002', 'm2000004', 'mean'],
    (2000002, 2000002): ['m1000021', 'm2000002', 'mean'],
    (2000002, 2000001): ['m1000021', 'm2000002', 'm2000001', 'mean'],
    (2000002, 2000003): ['m1000021', 'm2000002', 'm2000003', 'mean'],
    (2000002, 2000004): ['m1000021', 'm2000002', 'm2000004', 'mean'],
    (2000001, 2000001): ['m1000021', 'm2000001', 'mean'],
    (2000001, 2000003): ['m1000021', 'm2000001', 'm2000003', 'mean'],
    (2000001, 2000004): ['m1000021', 'm2000001', 'm2000004', 'mean'],
    (2000003, 2000003): ['m1000021', 'm2000003', 'mean'],
    (2000003, 2000004): ['m1000021', 'm2000003', 'm2000004', 'mean'],
    (2000004, 2000004): ['m1000021', 'm2000004', 'mean'],

    # --- Squark--anti-squark
    (1000004, -1000004): ['m1000021', 'm1000004', 'mean'],
    (1000004, -1000003): ['m1000021', 'm1000004', 'm1000003', 'mean'],
    (1000004, -1000001): ['m1000021', 'm1000004', 'm1000001', 'mean'],
    (1000004, -1000002): ['m1000021', 'm1000004', 'm1000002', 'mean'],
    (1000004, -2000002): ['m1000021', 'm1000004', 'm2000002', 'mean'],
    (1000004, -2000001): ['m1000021', 'm1000004', 'm2000001', 'mean'],
    (1000004, -2000003): ['m1000021', 'm1000004', 'm2000003', 'mean'],
    (1000004, -2000004): ['m1000021', 'm1000004', 'm2000004', 'mean'],
    (1000003, -1000003): ['m1000021', 'm1000003', 'mean'],
    (1000003, -1000001): ['m1000021', 'm1000003', 'm1000001', 'mean'],
    (1000003, -1000002): ['m1000021', 'm1000003', 'm1000002', 'mean'],
    (1000003, -2000002): ['m1000021', 'm1000003', 'm2000002', 'mean'],
    (1000003, -2000001): ['m1000021', 'm1000003', 'm2000001', 'mean'],
    (1000003, -2000003): ['m1000021', 'm1000003', 'm2000003', 'mean'],
    (1000003, -2000004): ['m1000021', 'm1000003', 'm2000004', 'mean'],
    (1000001, -1000001): ['m1000021', 'm1000001', 'mean'],
    (1000001, -1000002): ['m1000021', 'm1000001', 'm1000002', 'mean'],
    (1000001, -2000002): ['m1000021', 'm1000001', 'm2000002', 'mean'],
    (1000001, -2000001): ['m1000021', 'm1000001', 'm2000001', 'mean'],
    (1000001, -2000003): ['m1000021', 'm1000001', 'm2000003', 'mean'],
    (1000001, -2000004): ['m1000021', 'm1000001', 'm2000004', 'mean'],
    (1000002, -1000002): ['m1000021', 'm1000002', 'mean'],
    (1000002, -2000002): ['m1000021', 'm1000002', 'm2000002', 'mean'],
    (1000002, -2000001): ['m1000021', 'm1000002', 'm2000001', 'mean'],
    (1000002, -2000003): ['m1000021', 'm1000002', 'm2000003', 'mean'],
    (1000002, -2000004): ['m1000021', 'm1000002', 'm2000004', 'mean'],
    (2000002, -2000002): ['m1000021', 'm2000002', 'mean'],
    (2000002, -2000001): ['m1000021', 'm2000002', 'm2000001', 'mean'],
    (2000002, -2000003): ['m1000021', 'm2000002', 'm2000003', 'mean'],
    (2000002, -2000004): ['m1000021', 'm2000002', 'm2000004', 'mean'],
    (2000001, -2000001): ['m1000021', 'm2000001', 'mean'],
    (2000001, -2000003): ['m1000021', 'm2000001', 'm2000003', 'mean'],
    (2000001, -2000004): ['m1000021', 'm2000001', 'm2000004', 'mean'],
    (2000003, -2000003): ['m1000021', 'm2000003', 'mean'],
    (2000003, -2000004): ['m1000021', 'm2000003', 'm2000004', 'mean'],
    (2000004, -2000004): ['m1000021', 'm2000004', 'mean'],

    # Sbottom--anti-sbottom
    (1000005, -1000005): ['m1000021', 'm1000005', 'thetab', 'mean'],
    (2000005, -2000005): ['m1000021', 'm2000005', 'thetab', 'mean'],
    # Stop--anti-stop
    (1000006, -1000006): ['m1000021', 'm1000006', 'thetat', 'mean'],
    (2000006, -2000006): ['m1000021', 'm2000006', 'thetat', 'mean']
}


def get_features(pid1, pid2):
# Function that provides features for a tuple pid pair. Order of pids
# irrelevant. The function will raise errors when keys are not found.
    try:
        return FEATURES_LIST[(pid1, pid2)]
    except KeyError:
        try:
            return FEATURES_LIST[(pid2, pid1)]
        except KeyError as e:
            print('Feature requested for process not in dictionary!',
                  pid1, pid2)
            raise e


def get_features_dict(xsections_list):
# Produce a dictionary of features and their masses for each process in
# xsections_list

    # all_features = range(len(xsections_list)) # initialise dummy list
    # As dict
    all_features_dict = {}

    # TODO: why is mean mass set here? either the user specified it, so
    # we should set all other squark masses _here_, or it is not used at
    # all in the run ... unless we want to implement transforms using
    # the calculated mean squark mass! 
    set_mean_mass()

    # xsections_list has a list of proxesses we want features for so loop
    for i in range(len(xsections_list)):

        # Extract current process
        xsection = xsections_list[i]

        # Find features for this process
        features_index = get_features(xsection[0], xsection[1])

        # Make a feature dictionary
        #TODO: try/except KeyError!
        # features_dict = {key : PARAMS[key] for key in features_index}
        features_dict = collections.OrderedDict()
        for key in features_index:
            features_dict[key] = PARAMS[key]

        # ordered list! needed since X_train data from NIMBUS is ordered!
        features = features_dict.values()

        # Add features to feature array
        # all_features[i] = features
        all_features_dict.update({xsection: features})

    # Return feature array for all processes
    #return np.asarray(all_features)
    return all_features_dict
