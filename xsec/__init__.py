"""
Public interface for xsec.

All functions and global variables imported here are directly available
for usage in codes that import xsec.

Example:
    import xsec
    print(xsec.__version__)
    xsec.init()

Other functions and global variables can still be accessed by absolute
imports that specify the relevant module name, for example:
    from xsec.features import FEATURES_LIST
"""

from xsec.utils import __version__, list_all_xsec_processes
from xsec.gploader import (
    init,
    load_processes,
    unload_processes,
    get_processes,
    finalise,
)
from xsec.parameters import (
    set_parameter,
    set_parameters,
    set_energy,
    set_gluino_mass,
    set_12gen_squark_masses,
    set_all_squark_masses,
    clear_parameter,
    clear_parameters,
    get_parameter,
    get_parameters,
    get_energy,
    import_slha,
    import_slha_string,
    write_slha,
)
from xsec.features import get_features, get_unique_features, get_features_dict
from xsec.evaluation import eval_xsection
