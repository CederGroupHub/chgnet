from __future__ import annotations

import numpy as np
import pytest
from pymatgen.core import Structure
from pytest import mark

from chgnet import ROOT
from chgnet.graph import CrystalGraphConverter
from chgnet.model.model import CHGNet

structure = Structure.from_file(f"{ROOT}/examples/o-LiMnO2_unit.cif")
graph = CrystalGraphConverter()(structure, graph_id="test-model")


@mark.parametrize("atom_fea_dim", [1, 64])
@mark.parametrize("bond_fea_dim", [1, 64])
@mark.parametrize("angle_fea_dim", [1, 64])
@mark.parametrize("num_radial", [1, 9])
@mark.parametrize("num_angular", [1, 9])
@mark.parametrize("n_conv", [1, 4])
@mark.parametrize("composition_model", ["MPtrj", "MPtrj_e", "MPF"])
def test_model(
    atom_fea_dim: int,
    bond_fea_dim: int,
    angle_fea_dim: int,
    num_radial: int,
    num_angular: int,
    n_conv: int,
    composition_model: str,
) -> None:
    model = CHGNet(
        atom_fea_dim=atom_fea_dim,
        bond_fea_dim=bond_fea_dim,
        angle_fea_dim=angle_fea_dim,
        num_radial=num_radial,
        num_angular=num_angular,
        n_conv=n_conv,
        composition_model=composition_model,
    )
    out = model([graph])
    assert list(out) == ["atoms_per_graph", "e"]
    assert out["atoms_per_graph"].shape == (1,)
    assert out["e"] < 0


def test_predict_structure() -> None:
    model = CHGNet.load()
    out = model.predict_structure(structure)

    assert sorted(out) == ["e", "f", "m", "s"]
    assert out["e"] == pytest.approx(-7.37159, abs=1e-4)
    assert len(out["f"]) == len(structure)
    assert len(out["m"]) == len(structure)
    assert out["m"] == pytest.approx(
        [0.00521, 0.00521, 3.85728, 3.85729, 0.02538, 0.03706, 0.03706, 0.02538],
        abs=1e-4,
    )
    assert len(out["s"]) == 3
    stress = np.array(
        [
            [3.3677614e-01, -1.9665707e-07, -5.6416429e-06],
            [4.9939729e-07, 2.4675032e-01, 1.8549043e-05],
            [-4.0414070e-06, 1.9096897e-05, 4.0323928e-02],
        ]
    )
    assert out["s"] == pytest.approx(stress, abs=1e-4)
