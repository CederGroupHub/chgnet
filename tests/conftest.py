from __future__ import annotations

import pytest
from pymatgen.core import Structure

from chgnet import ROOT


@pytest.fixture
def li_mn_o2() -> Structure:
    return Structure.from_file(f"{ROOT}/examples/mp-18767-LiMnO2.cif")
