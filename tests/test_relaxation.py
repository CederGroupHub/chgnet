from __future__ import annotations

from pymatgen.core import Structure
from pytest import approx

from chgnet.model import StructOptimizer

relaxer = StructOptimizer()
structure = Structure.from_file("examples/o-LiMnO2_unit.cif")


def test_relaxation():
    result = relaxer.relax(structure, verbose=True)

    assert list(result) == ["final_structure", "trajectory"]

    traj = result["trajectory"]
    # make sure trajectory has expected attributes
    assert list(traj.__dict__) == [
        "atoms",
        "energies",
        "forces",
        "stresses",
        "magmoms",
        "atom_positions",
        "cells",
    ]
    assert len(traj) == 4

    # make sure final structure is more relaxed than initial one
    assert traj.energies[0] > traj.energies[-1]

    assert traj.energies[-1] == approx(-58.972927)
