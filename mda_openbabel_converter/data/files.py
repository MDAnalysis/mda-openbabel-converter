"""
Location of data files
======================

Use as ::

    from mda_openbabel_converter.data.files import *

"""

__all__ = [
    "MDANALYSIS_LOGO",  # example file of MDAnalysis logo
]

from importlib.resources import files
MDANALYSIS_LOGO = files("mda_openbabel_converter") / "data" / "mda.txt"
CRN = files("mda_openbabel_converter") / "data" / "1crn.pdb"
HEME = files("mda_openbabel_converter") / "data" / "ChEBI_26355.mol"
COMPLEX_SDF = files("mda_openbabel_converter") / "data" / "InChI_TestSet.sdf"
MET_ENKAPH_MOVIE = files("mda_openbabel_converter") / "data" / "met-enkaphalin_movie.xyz"