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
Test suite for the OpenBabel Reader that converts an OBMol's atom coordinates 
to an MDAnalysis topology, alongside the OpenBabel Parser, that can be used to 
construct an MDAnalysis Universe.
"""

import MDAnalysis as mda
import openbabel
from openbabel import openbabel as ob
from openbabel.openbabel import OBMol, OBConversion, GetSymbol, OBMolAtomIter
from mda_openbabel_converter.data.files import CRN, HEME, COMPLEX_SDF, MET_ENKAPH_MOVIE

import mda_openbabel_converter
import pytest  # version 8.2.2
import sys
import numpy as np
from numpy.testing import assert_equal, assert_allclose
from MDAnalysisTests.topology.base import ParserBase
from MDAnalysis.core.topology import Topology
# from MDAnalysis.core.universe import Merge
from MDAnalysis.coordinates.chain import ChainReader

import mda_openbabel_converter.OpenBabel

class TestOpenBabelReader(object):
    reader = mda_openbabel_converter.OpenBabel.OpenBabelReader

    @pytest.fixture()
    def n_frames(self):
        return 1

    def test_coordinates_crn(self, n_frames):
        obconversion = ob.OBConversion()
        obconversion.SetInFormat("pdb")
        obmol = ob.OBMol()
        obconversion.ReadFile(obmol, CRN.as_posix())
        universe = mda.Universe(obmol)
        assert universe.trajectory.n_frames == n_frames
        expected = np.array([
            [(coords := atom.GetVector()).GetX(),
            coords.GetY(),
            coords.GetZ()] for atom in OBMolAtomIter(obmol)],
            dtype=np.float32)
        assert_equal(expected, universe.trajectory.coordinate_array[0])

    def test_coordinates_heme(self, n_frames):
        obconversion = ob.OBConversion()
        obconversion.SetInFormat("mol")
        obmol = ob.OBMol()
        obconversion.ReadFile(obmol, HEME.as_posix())
        universe = mda.Universe(obmol)
        assert universe.trajectory.n_frames == n_frames
        expected = np.array([
            [(coords := atom.GetVector()).GetX(),
            coords.GetY(),
            coords.GetZ()] for atom in OBMolAtomIter(obmol)],
            dtype=np.float32)
        assert_equal(expected, universe.trajectory.coordinate_array[0])

    def test_coordinates_complex(self, n_frames):
        obconversion = ob.OBConversion()
        obconversion.SetInFormat("mol")
        obmol = ob.OBMol()
        obconversion.ReadFile(obmol, COMPLEX_SDF.as_posix())
        universe = mda.Universe(obmol)
        assert universe.trajectory.n_frames == n_frames
        expected = np.array([
            [(coords := atom.GetVector()).GetX(),
            coords.GetY(),
            coords.GetZ()] for atom in OBMolAtomIter(obmol)],
            dtype=np.float32)
        assert_equal(expected, universe.trajectory.coordinate_array[0])

    def test_multiple_conformer(self):
        ob_conversion = ob.OBConversion()
        ob_conversion.SetInFormat("smi")
        mol = ob.OBMol()
        ob_conversion.ReadString(mol, "CC(=O)OC1=CC=CC=C1C(=O)O")
        assert mol.AddHydrogens()

        # build basic 3D coordinates
        builder = ob.OBBuilder()
        assert builder.Build(mol)

        # generate multiconfs
        confsearch = ob.OBConformerSearch()
        num_conformers = 3
        confsearch.Setup(mol, num_conformers)
        confsearch.Search()
        confsearch.GetConformers(mol)
        assert mol.NumConformers() == 3

        # convert to Universe
        universe = mda.Universe(mol)
        n_atoms = mol.NumAtoms()
        expected_coordinates = np.empty((mol.NumConformers(), n_atoms, 3))
        expected_shape = expected_coordinates.shape
        assert_equal(universe.trajectory.coordinate_array.shape, expected_shape)

    def test_no_coordinates(self):
        obConversion = ob.OBConversion()
        obConversion.SetInFormat("smi")
        mol = OBMol()
        obConversion.ReadString(mol, "C1=CC=CS1")
        mol.AddHydrogens()
        universe = mda.Universe(mol)
        n_atoms = mol.NumAtoms()
        expected = coordinates = np.empty((1, n_atoms, 3), dtype=np.float32)
        coordinates[:] = np.nan
        assert_equal(universe.trajectory.coordinate_array, expected)
