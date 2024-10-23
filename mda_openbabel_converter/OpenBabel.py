"""OpenBabel molecule I/O --- :mod:`mda_openbabel_converter.OpenBabel`
======================================================================

Read coordinates data from an
`OpenBabel <http://openbabel.org/api/3.0/classOpenBabel_1_1OBMol.shtml>`_
:class:`openbabel.openbabel.OBMol` with :class:`OpenBabelReader` into an
MDAnalysis Universe. Convert it back to a :class:`openbabel.openbabel.OBMol`
with :class:`OpenBabelConverter`.

Example
-------

To read an OpenBabel OBMol and then convert the AtomGroup back to an OpenBabel
OBMol::

    >>> from openbabel import openbabel as ob
    >>> import MDAnalysis as mda
    >>> obconversion = ob.OBConversion()
    >>> obconversion.SetInFormat("pdb")
    >>> mol = ob.OBMol()
    >>> obconversion.ReadFile(mol, "1crn.pdb")
    >>> u = mda.Universe(mol)
    >>> u
    <Universe with 327 atoms>
    >>> u.trajectory
    <OpenBabelReader with 1 frame of 327 atoms>
    >>> u.atoms.convert_to("OPENBABEL")
    <openbabel.openbabel.OBMol object at 0x7fcebb958148>


.. warning::
    The OpenBabel converter is currently *experimental* and may not work as
    expected for all molecules. 


Classes
-------

.. autoclass:: OpenBabelReader
   :members:

.. autoclass:: OpenBabelConverter
   :members:
"""

import MDAnalysis as mda
from MDAnalysis.converters.base import ConverterBase
from MDAnalysis.coordinates.memory import MemoryReader
from MDAnalysis.core.groups import AtomGroup
import numpy as np
import warnings

HAS_OBABEL = False

try:
    from openbabel import openbabel as ob
    from openbabel.openbabel import OBMol, OBAtom, OBMolAtomIter
    HAS_OBABEL = True
except ImportError:
    warnings.warn("Cannot find openbabel, install with `mamba install -c "
                  "conda-forge openbabel`")


class OpenBabelReader(MemoryReader):
    """
    Coordinate reader for OpenBabel.

    Inherits from MemoryReader and converts OpenBabel OBMol Coordinates to a
    MDAnalysis Trajectory which is used to build a Universe. This reader
    does NOT work in the reverse direction. 

    See :class:`mda_openbabel_converter.OpenBabel.OpenBabelConverter` for
    MDAnalysis Universe to OpenBabel OBMol conversion.
    """
    format = 'OPENBABEL'

    # Structure.coordinates always in Angstrom
    units = {'time': None, 'length': 'Angstrom'}

    @staticmethod
    def _format_hint(thing):
        """
        Base function to check if the reader can actually read this “thing”
        (i.e., is it a file that can be converted to an OpenBabel OBMol?)
        """
        if HAS_OBABEL is False:
            return False
        else:
            return isinstance(thing, OBMol)

    def __init__(self, filename: OBMol, **kwargs):
        """
        Converts file to OBMol to AtomGroup
        """
        n_atoms = filename.NumAtoms()
        # single position
        if filename.NumConformers() == 1:
            coordinates = np.array([
                [(coords := atom.GetVector()).GetX(),
                    coords.GetY(),
                    coords.GetZ()] for atom in OBMolAtomIter(filename)],
                dtype=np.float32)
        else:
            # multiple conformers, such as for a trajectory
            numConf = filename.NumConformers()
            coordinates = np.zeros((numConf, n_atoms, 3))
            for conf_id in range(numConf):
                filename.SetConformer(conf_id)
                for atom in OBMolAtomIter(filename):
                    coordinates_inner = np.array([
                        [(coords := atom.GetVector()).GetX(),
                            coords.GetY(),
                            coords.GetZ()] for atom in OBMolAtomIter(filename)],
                        dtype=np.float32)
                coordinates[conf_id] = coordinates_inner
        # no coordinates present
        if not np.any(coordinates):
            warnings.warn("No coordinates found in the OBMol")
            coordinates = np.empty((1, n_atoms, 3), dtype=np.float32)
            coordinates[:] = np.nan
        super(OpenBabelReader, self).__init__(coordinates, 
                order='fac', **kwargs)

class OpenBabelConverter(ConverterBase):
    """
    Inherits from ConverterBase and converts a MDAnalysis Universe to an
    OpenBabel OBMol. This converter does NOT work in the opposite direction.

    See :class:`mda_openbabel_converter.OpenBabelReader` for OpenBabel OBMol
    to MDAnalysis Universe conversion.
    """
    def __repr__(self, **kwargs):
        """
        String representation of the object (defined in Base Class)
        """
        pass

    def convert(self, obj, **kwargs):
        """
        Converts AtomGroup to OBMol
        """
        pass

    # add getter and setter methods as required

    # add helper methods