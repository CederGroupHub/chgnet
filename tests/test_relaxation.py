from __future__ import annotations

from pymatgen.core import Structure
from pytest import approx, mark

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


@mark.parametrize("use_device", ["cpu", "mps", "cuda"])
def test_structure_optimizer_passes_kwargs_to_model(use_device) -> None:
    try:
        relaxer = StructOptimizer(use_device=use_device)
        assert relaxer.calculator.device == use_device
    except (RuntimeError, AssertionError) as exc:
        assert "Torch not compiled with CUDA enabled" in str(
            exc
        ) or 'NotImplementedError("mps is not supported yet")' in str(exc)
        # TODO: remove mps case once mps is supported
