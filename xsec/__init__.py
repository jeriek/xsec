"""
Public interface for xsec.
"""

from utils import __version__
from gploader import (
    init,
    set_processes,
    load_processes,
    get_processes,
    finalise,
)
from parameters import (
    set_parameter,
    set_parameters,
    set_gluino_mass,
    set_12gen_squark_masses,
    set_all_squark_masses,
    clear_parameter,
    clear_parameters,
    get_parameter,
    get_parameters,
    import_slha,
    write_slha,
)
from features import get_features, get_unique_features, get_features_dict
from evaluation import eval_xsection
