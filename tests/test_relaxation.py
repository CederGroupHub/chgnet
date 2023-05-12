from __future__ import annotations

import torch
from pymatgen.core import Structure
from pytest import approx, mark, param

from chgnet import ROOT
from chgnet.model import StructOptimizer

relaxer = StructOptimizer()
structure = Structure.from_file(f"{ROOT}/examples/o-LiMnO2_unit.cif")


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


no_cuda = mark.skipif(not torch.cuda.is_available(), reason="No CUDA device")
no_mps = mark.skipif(not hasattr(torch.backends, "mps"), reason="No MPS device")


@mark.parametrize(
    "use_device", ["cpu", param("cuda", marks=no_cuda), param("mps", marks=no_mps)]
)
def test_structure_optimizer_passes_kwargs_to_model(use_device) -> None:
    try:
        relaxer = StructOptimizer(use_device=use_device)
        assert relaxer.calculator.device == use_device
    except NotImplementedError as exc:
        # TODO: remove try/except once mps is supported
        assert str(exc) == "mps is not supported yet"  # noqa: PT017
