# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding:utf-8 -*-
# vim: tabstop=4 expandtab shiftwidth=4 softtabstop=4 
#
# MDAnalysis --- https://www.mdanalysis.org
# Copyright (c) 2006-2017 The MDAnalysis Development Team and contributors
# (see the file AUTHORS for the full list of names)
#
# Released under the GNU Public Licence, v2 or any higher version
#
# Please cite your use of MDAnalysis in published work:
#
# R. J. Gowers, M. Linke, J. Barnoud, T. J. E. Reddy, M. N. Melo, S. L. Seyler,
# D. L. Dotson, J. Domanski, S. Buchoux, I. M. Kenney, and O. Beckstein.
# MDAnalysis: A Python package for the rapid analysis of molecular dynamics
# simulations. In S. Benthall and S. Rostrup editors, Proceedings of the 15th
# Python in Science Conference, pages 102-109, Austin, TX, 2016. SciPy.
# doi: 10.25080/majora-629e541a-00e
#
# N. Michaud-Agrawal, E. J. Denning, T. B. Woolf, and O. Beckstein.
# MDAnalysis: A Toolkit for the Analysis of Molecular Dynamics Simulations.
# J. Comput. Chem. 32 (2011), 2319--2327, doi:10.1002/jcc.21787
#

"""
OpenBabel topology parser --- :mod:`MDAnalysis.converters.RDKitParser`
==================================================================

Converts an `OpenBabel <http://openbabel.org/api/3.0/classOpenBabel_1_1OBMol.shtml>`_ :class:`openbabel.openbabel.OBMol` into a :class:`MDAnalysis.core.Topology`.


See Also
--------
:mod:`MDAnalysis.MDAKits.mdakits.open-babel-converter`


Classes
-------

.. autoclass:: OpenBabelParser
   :members:
   :inherited-members:

"""

import MDAnalysis as mda
from MDAnalysis.topology.base import TopologyReaderBase, change_squash
from MDAnalysis.core.topology import Topology
from MDAnalysis.topology import guessers
from MDAnalysis.converters.base import ConverterBase
from MDAnalysis.core.topologyattrs import (
    Atomids,
    Atomnames,
    Atomtypes,
    Elements,
    Masses,
    Charges,
    Aromaticities,
    Bonds,
    Resids,
    Resnums,
    Resnames,
    RSChirality,
    Segids,
    AltLocs,
    ChainIDs,
    ICodes,
    Occupancies,
    Tempfactors,
)
import warnings
import numpy as np

HAS_OBABEL = False
NEUTRON_MASS = 1.008

try:
    import openbabel
    from openbabel import openbabel as ob
    from openbabel.openbabel import OBMol, OBResidue, GetSymbol
    from openbabel.openbabel import *
    HAS_OBABEL = True
except ImportError:
    warnings.warn("Cannot find openbabel, install with `mamba install -c "
                  "conda-forge openbabel`")


class OpenBabelParser(TopologyReaderBase):
    """
    For OpenBabel structure

    Inherits from TopologyReaderBase and converts an OpenBabel OBMol to a
    MDAnalysis Topology or adds it to a pre-existing Topology. This parser
    does not work in the reverse direction.

    For use examples, please see OpenBabel Class documentation

    Creates the following Attributes:
     - Atomids
     - Atomtypes
     - Aromaticities
     - Elements
     - Masses
     - Bonds
     - Resids
     - Resnums
     - Segids

    Depending on OpenBabel's input, the following Attributes might be present:
     - Charges
     - Resnames
     - ICodes
    
    Guesses the following:
     - Atomnames

    Missing Attributes unable to be retrieved from OpenBabel:
     - Chiralities
     - RSChirality
     - Occupancies
     - Tempfactors
     - ChainIDs
     - AltLocs

    Attributes table:

    +---------------------------------------------+-------------------------+
    | OpenBabel attribute                         | MDAnalysis equivalent   |
    +=============================================+=========================+
    |                                             | altLocs                 |
    +---------------------------------------------+-------------------------+
    | atom.IsAromatic()                           | aromaticities           |
    +---------------------------------------------+-------------------------+
    |                                             | chainIDs                |
    +---------------------------------------------+-------------------------+
    | atom.GetPartialCharge()                     | charges                 |
    +---------------------------------------------+-------------------------+
    | GetSymbol(atom.GetAtomicNum())              | elements                |
    +---------------------------------------------+-------------------------+
    | atom.GetResidue().GetInsertionCode()        | icodes                  |
    +---------------------------------------------+-------------------------+
    | atom.GetIdx()                               | indices                 |
    +---------------------------------------------+-------------------------+
    | atom.GetExactMass()                         | masses                  |
    +---------------------------------------------+-------------------------+
    | "%s%d" % (GetSymbol(atom.GetAtomicNum()),   | names                   |
    | atom.GetIdx())                              |                         |
    +---------------------------------------------+-------------------------+
    |                                             | chiralities             |
    +---------------------------------------------+-------------------------+
    |                                             | occupancies             |
    +---------------------------------------------+-------------------------+
    | atom.GetResidue().GetName()                 | resnames                |
    +---------------------------------------------+-------------------------+
    | atom.GetResidue().GetNum()                  | resnums                 |
    +---------------------------------------------+-------------------------+
    |                                             | tempfactors             |
    +---------------------------------------------+-------------------------+
    | atom.GetType()                              | types                   |
    +---------------------------------------------+-------------------------+

    Raises
    ------
    ValueError
        If only some of the atoms have ResidueInfo, from resid.GetNum(), 
        available

    """
    format = 'OPENBABEL'

    @staticmethod
    def _format_hint(thing):
        """
        Base function to check if the parser can actually parse this “thing”
        (i.e., is it a valid OpenBabel OBMol that can be converted to a
        MDAnalysis Topology?)
        """
        if HAS_OBABEL is False:
            return False
        else:
            return isinstance(thing, ob.OBMol)

    def parse(self, **kwargs):
        """
        Accepts an OpenBabel OBMol and returns a MDAnalysis Topology. Will need
        to extract the number of atoms, number of residues, number of segments,
        atom_residue index, residue_segment index and all of the atom's
        relevant attributes from the OBMol to initialise a new Topology.
        """
        mol = self.filename

        # Atoms
        names = []
        resnums = []
        resnames = []
        elements = []
        masses = []
        charges = []
        aromatics = []
        ids = []
        atomtypes = []
        segids = []
        icodes = []

        if mol.Empty():
            return Topology(n_atoms=0,
                            n_res=0,
                            n_seg=0,
                            attrs=None,
                            atom_resindex=None,
                            residue_segindex=None)

        for atom in ob.OBMolAtomIter(mol):
            # Name set with element and id, as name not stored by OpenBabel
            a_id = atom.GetIdx()
            name = "%s%d" % (GetSymbol(atom.GetAtomicNum()), a_id)
            names.append(name)
            atomtypes.append(atom.GetType())
            ids.append(a_id)
            masses.append(atom.GetExactMass())
            if abs(atom.GetExactMass()-atom.GetAtomicMass()) >= NEUTRON_MASS:
                warnings.warn(
                    f"Exact mass and atomic mass of atom ID: {a_id} are more"
                    " than 1.008 AMU different. Be aware of isotopes,"
                    " which are NOT flagged by MDAnalysis.")
            charges.append(atom.GetPartialCharge())

            # convert atomic number to element
            elements.append(GetSymbol(atom.GetAtomicNum()))

            # only for PBD and MOL2
            if atom.HasResidue():
                resid = atom.GetResidue()
                resnums.append(resid.GetNum())
                resnames.append(resid.GetName())
                icodes.append(resid.GetInsertionCode())

            aromatics.append(atom.IsAromatic())

        # make Topology attributes
        attrs = []
        n_atoms = len(ids)

        if resnums and (len(resnums) != len(ids)):
            raise ValueError(
                "ResidueInfo is only partially available in the molecule."
            )

        # * Attributes always present *

        # Atom attributes
        for vals, Attr, dtype in (
            (ids, Atomids, np.int32),
            (elements, Elements, object),
            (masses, Masses, np.float32),
            (aromatics, Aromaticities, bool),
        ):
            attrs.append(Attr(np.array(vals, dtype=dtype)))

        # Bonds
        bonds = []
        bond_orders = []
        for bond_idx in range(0, mol.NumBonds()):
            bond = mol.GetBond(bond_idx)
            bonds.append((bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()))
            bond_orders.append(float(bond.GetBondOrder()))
        attrs.append(Bonds(bonds, order=bond_orders))

        # * Optional attributes *
        attrs.append(Atomnames(np.array(names, dtype=object)))

        # Atom type
        if atomtypes:
            attrs.append(Atomtypes(np.array(atomtypes, dtype=object)))
        else:
            atomtypes = guessers.guess_types(names)
            attrs.append(Atomtypes(atomtypes, guessed=True))

        # Partial charges
        if charges:
            attrs.append(Charges(np.array(charges, dtype=np.float32)))
        else:
            pass  # no guesser yet

        # Residue
        if resnums:
            resnums = np.array(resnums, dtype=np.int32)
            resnames = np.array(resnames, dtype=object)
            icodes = np.array(icodes, dtype=object)
            residx, (resnums, resnames, icodes) = change_squash(
                (resnums, resnames, icodes),
                (resnums, resnames, icodes))
            n_residues = len(resnums)
            for vals, Attr, dtype in (
                (resnums, Resids, np.int32),
                (resnums.copy(), Resnums, np.int32),
                (resnames, Resnames, object),
                (icodes, ICodes, object),
            ):
                attrs.append(Attr(np.array(vals, dtype=dtype)))
        else:
            attrs.append(Resids(np.array([1])))
            attrs.append(Resnums(np.array([1])))
            residx = None
            n_residues = 1

        # Segment
        if len(segids) and not any(val is None for val in segids):
            segidx, (segids,) = change_squash((segids,), (segids,))
            n_segments = len(segids)
            attrs.append(Segids(segids))
        else:
            n_segments = 1
            attrs.append(Segids(np.array(['SYSTEM'], dtype=object)))
            segidx = None

        # create topology
        top = Topology(n_atoms, n_residues, n_segments,
                       attrs=attrs,
                       atom_resindex=residx,
                       residue_segindex=segidx)

        return top
