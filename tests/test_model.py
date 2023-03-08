from __future__ import annotations

from pymatgen.core import Structure
from pytest import mark

from chgnet.graph import CrystalGraphConverter
from chgnet.model.model import CHGNet

structure = Structure.from_file("examples/o-LiMnO2_unit.cif")
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
):
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
