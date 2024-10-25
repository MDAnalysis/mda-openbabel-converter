"""
Base tests for the mda_openbabel_converter package.
"""

import mda_openbabel_converter
import pytest
import sys


def test_mda_openbabel_converter_imported():
    """Sample test, will always pass so long as import statement worked"""
    assert "mda_openbabel_converter" in sys.modules


def test_mdanalysis_logo_length(mdanalysis_logo_text):
    """Example test using a fixture defined in conftest.py"""
    logo_lines = mdanalysis_logo_text.split("\n")
    assert len(logo_lines) == 46, "Logo file does not have 46 lines!"
