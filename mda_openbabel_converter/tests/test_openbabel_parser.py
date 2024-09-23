# Testing OpenBabel and Pybel

import MDAnalysis as mda
import openbabel
from openbabel import openbabel as ob
from openbabel.openbabel import OBMol, OBConversion, GetSymbol
from mda_openbabel_converter.data.files import CRN, HEME, COMPLEX_SDF, MET_ENKAPH_MOVIE

import mda_openbabel_converter
import pytest  # version 8.2.2
import sys
import numpy as np
from numpy.testing import assert_equal, assert_allclose
from MDAnalysisTests.topology.base import ParserBase
from MDAnalysis.core.topology import Topology

import mda_openbabel_converter.OpenBabelParser  # version 2.7.0

# *** can run with "python -m pytest" but not "pytest" (can't find
# MDAnalysis) - need to fix this! ***


class OpenBabelParserBase(ParserBase):
    parser = mda_openbabel_converter.OpenBabelParser.OpenBabelParser

    expected_attrs = ['ids', 'names', 'elements', 'masses', 'aromaticities',
                      'resids', 'resnums', 'segids', 'bonds',
                      ]

    expected_n_atoms = 0
    expected_n_residues = 0
    expected_n_segments = 0
    expected_n_bonds = 0

    def test_creates_universe(self, top):
        u = mda.Universe(top, format='OPENBABEL')
        assert isinstance(u, mda.Universe)

    def test_bonds_total_counts(self, top):
        if hasattr(top, 'bonds'):
            assert len(top.bonds.values) == self.expected_n_bonds


class TestOpenBabelParserEmpty(OpenBabelParserBase):
    @pytest.fixture()
    def filename(self):
        return OBMol()

    expected_attrs = []
    mandatory_attrs = []  # as not instantiated during empty Topology creation

    def test_mandatory_attributes(self, top):
        for attr in self.mandatory_attrs:
            assert (hasattr(top, attr),
                    'Missing required attribute: {}'.format(attr))

    def test_attrs_total_counts(self, top):
        ag = mda.Universe(top).select_atoms("all")
        res = ag.residues
        seg = ag.segments
        assert len(ag) == self.expected_n_atoms
        assert len(res) == self.expected_n_residues
        assert len(seg) == self.expected_n_segments


class TestOpenBabelParserSMILES(OpenBabelParserBase):
    expected_attrs = OpenBabelParserBase.expected_attrs + ['charges']

    @pytest.fixture()
    def filename(self):
        obConversion = ob.OBConversion()
        obConversion.SetInFormat("smi")
        mol = OBMol()
        obConversion.ReadString(mol, "C1=CC=CS1")
        mol.AddHydrogens()
        return mol

    @pytest.fixture()
    def top(self, filename):
        yield self.parser(filename).parse()

    expected_n_atoms = 9
    expected_n_residues = 1
    expected_n_segments = 1
    expected_n_bonds = 9

    def test_attrs_total_counts(self, top):
        u = mda.Universe(top, format = 'OPENBABEL')
        ag = u.select_atoms("all")
        res = ag.residues
        seg = ag.segments
        assert len(ag) == self.expected_n_atoms
        assert len(res) == self.expected_n_residues
        assert len(seg) == self.expected_n_segments

    def test_aromaticities(self, top, filename):
        expected = np.array([
            atom.IsAromatic() for atom in ob.OBMolAtomIter(filename)])
        assert_equal(expected, top.aromaticities.values)

    def test_elements(self, top, filename):
        expected = np.array([
            GetSymbol(atom.GetAtomicNum()) for atom in
            ob.OBMolAtomIter(filename)])
        assert_equal(expected, top.elements.values)

    def test_charges(self, top, filename):
        expected = np.array([
            atom.GetPartialCharge() for atom in ob.OBMolAtomIter(filename)])
        assert_allclose(expected, top.charges.values)

    def test_mass_check(self, top, filename):
        expected = np.array([
            atom.GetExactMass() for atom in ob.OBMolAtomIter(filename)])
        assert_allclose(expected, top.masses.values)

class TestOpenBabelParserPDB(OpenBabelParserBase):
    # expected_attrs = OpenBabelParserBase.expected_attrs + ['charges', 'resnames', 'icodes', 'chainIDs']
    expected_attrs = OpenBabelParserBase.expected_attrs + ['charges', 'resnames', 'icodes']

    @pytest.fixture()
    def filename(self):
        obconversion = ob.OBConversion()
        obconversion.SetInFormat("pdb")
        mol = ob.OBMol()
        obconversion.ReadFile(mol, CRN.as_posix())
        return mol

    @pytest.fixture()
    def top(self, filename):
        yield self.parser(filename).parse()

    # to do: check these values!
    expected_n_atoms = 327
    expected_n_residues = 46
    expected_n_segments = 1
    expected_n_bonds = 337

    def test_attrs_total_counts(self, top):
        u = mda.Universe(top, format = 'OPENBABEL')
        # expected = np.array([
        #     [(coords := atom.GetVector()).GetX(),
        #     coords.GetY(),
        #     coords.GetZ()] for atom in OBMolAtomIter(obmol)],
        #     dtype=np.float32)
        # assert_equal(expected, universe.trajectory.coordinate_array[0])
        ag = u.select_atoms("all")
        res = ag.residues
        seg = ag.segments
        assert len(ag) == self.expected_n_atoms
        assert len(res) == self.expected_n_residues
        assert len(seg) == self.expected_n_segments
        
    # def test_icodes(self, top, filename):
    #     expected = np.array([
    #         atom.GetResidue().GetInsertionCode() for atom in ob.OBMolAtomIter(filename)])
    #     for atom in ob.OBMolAtomIter(filename):
    #         print(f"ic ={atom.GetResidue().GetInsertionCode()}gotit?")
    #     self.parser(filename).parse()
    #     print(top.icodes.values)
    #     print(expected)
    #     assert_equal(expected, top.icodes.values)
        
    def test_elements(self, top, filename):
        expected = np.array([
            GetSymbol(atom.GetAtomicNum()) for atom in
            ob.OBMolAtomIter(filename)])
        assert_equal(expected, top.elements.values)

    def test_charges(self, top, filename):
        expected = np.array([
            atom.GetPartialCharge() for atom in ob.OBMolAtomIter(filename)])
        assert_allclose(expected, top.charges.values)

    def test_mass_check(self, top, filename):
        expected = np.array([
            atom.GetExactMass() for atom in ob.OBMolAtomIter(filename)])
        assert_allclose(expected, top.masses.values)

    def test_residues(self, top, filename):
        expected_names = np.array([
            residue.GetName() for residue in ob.OBResidueIter(filename)])
        assert_equal(expected_names, top.resnames.values)
        
        expected_ids = np.array([
            residue.GetNum() for residue in ob.OBResidueIter(filename)])
        assert_equal(expected_ids, top.resids.values)

        # self.parser(filename).parse()
        # expected_chains = np.array([
        #     residue.GetChain() for residue in ob.OBResidueIter(filename)])
        # print(expected_chains)
        # print(top.chainIDs.values)
        # # assert_equal(expected_chains, top.chainIDs.values)
        # assert_equal(0, 1)

class TestOpenBabelParserSDF(OpenBabelParserBase):
    expected_attrs = OpenBabelParserBase.expected_attrs + ['charges']

    @pytest.fixture()
    def filename(self):
        obconversion = ob.OBConversion()
        obconversion.SetInFormat("sdf")
        mol = ob.OBMol()
        obconversion.ReadFile(mol, COMPLEX_SDF.as_posix())
        print(mol.NumBonds())
        return mol

    @pytest.fixture()
    def top(self, filename):
        yield self.parser(filename).parse()

    # to do: check these values!
    expected_n_atoms = 212
    expected_n_residues = 1
    expected_n_segments = 1
    expected_n_bonds = 0  # OpenBabel unable to read bonds

    def test_attrs_total_counts(self, top):
        u = mda.Universe(top, format = 'OPENBABEL')
        ag = u.select_atoms("all")
        res = ag.residues
        seg = ag.segments
        assert len(ag) == self.expected_n_atoms
        assert len(res) == self.expected_n_residues
        assert len(seg) == self.expected_n_segments

class TestOpenBabelParserMOL(OpenBabelParserBase):
    expected_attrs = OpenBabelParserBase.expected_attrs + ['charges']

    @pytest.fixture()
    def filename(self):
        obconversion = ob.OBConversion()
        obconversion.SetInFormat("mol")
        mol = ob.OBMol()
        obconversion.ReadFile(mol, HEME.as_posix())
        return mol

    @pytest.fixture()
    def top(self, filename):
        yield self.parser(filename).parse()

    # to do: check these values!
    expected_n_atoms = 43
    expected_n_residues = 1
    expected_n_segments = 1
    expected_n_bonds = 50

    def test_attrs_total_counts(self, top):
        u = mda.Universe(top, format = 'OPENBABEL')
        ag = u.select_atoms("all")
        res = ag.residues
        seg = ag.segments
        print(len(ag))
        assert len(ag) == self.expected_n_atoms
        assert len(res) == self.expected_n_residues
        assert len(seg) == self.expected_n_segments