
from pkg_resources import get_distribution, DistributionNotFound
import os.path

from gploader import (
    init, set_processes, load_processes, get_processes,
    finalize
    )
from features import(
    get_features, get_feature_list, get_features_dict
    )
from parameters import (
    set_parameter, set_parameters, set_common_squark_mass,
    set_gluino_mass, clear_parameter, clear_parameters,
    get_parameter, get_parameters, import_slha, write_slha
    )
from evaluation import (
    eval_xsection
    )

# Set the module's __version__ by getting the version set in setup.py
# (which in turn is set by the VERSION file)
try:
    _dist = get_distribution('xsec')
    # Normalize case for Windows systems
    dist_loc = os.path.normcase(_dist.location)
    here = os.path.normcase(__file__)
    if not here.startswith(os.path.join(dist_loc, 'xsec')):
        # if not installed, but there is another version that *is*
        raise DistributionNotFound
except DistributionNotFound:
    __version__ = 'Please install this project with pip.'
else:
    __version__ = _dist.version
