"""
Global pytest fixtures
"""

import pytest
from mda_openbabel_converter.data.files import MDANALYSIS_LOGO


@pytest.fixture
def mdanalysis_logo_text() -> str:
    """Example fixture demonstrating how data files can be accessed"""
    with open(MDANALYSIS_LOGO, "r", encoding="utf8") as f:
        logo_text = f.read()
    return logo_text
