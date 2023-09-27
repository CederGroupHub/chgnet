from __future__ import annotations

import numpy as np
import pytest
from pymatgen.core import Structure
from pytest import mark

from chgnet import ROOT
from chgnet.graph import CrystalGraphConverter
from chgnet.model.model import CHGNet

structure = Structure.from_file(f"{ROOT}/examples/mp-18767-LiMnO2.cif")
graph = CrystalGraphConverter()(structure, graph_id="test-model")
model = CHGNet.load()


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
    capsys: pytest.CaptureFixture[str],
) -> None:
    converter_verbose = False
    model = CHGNet(
        atom_fea_dim=atom_fea_dim,
        bond_fea_dim=bond_fea_dim,
        angle_fea_dim=angle_fea_dim,
        num_radial=num_radial,
        num_angular=num_angular,
        n_conv=n_conv,
        composition_model=composition_model,
        converter_verbose=converter_verbose,
    )
    out = model([graph])
    assert list(out) == ["atoms_per_graph", "e"]
    assert out["atoms_per_graph"].shape == (1,)
    assert out["e"] < 0

    stdout, stderr = capsys.readouterr()
    if converter_verbose:
        assert repr(model.graph_converter) in stdout
    else:
        assert "CHGNet initialized with" in stdout

    assert stderr == ""


def test_predict_structure() -> None:
    out = model.predict_structure(
        structure,
        return_site_energies=True,
        return_atom_feas=True,
        return_crystal_feas=True,
    )

    assert sorted(out) == [
        "atom_fea",
        "crystal_fea",
        "e",
        "f",
        "m",
        "s",
        "site_energies",
    ]
    assert out["e"] == pytest.approx(-7.37159, rel=1e-4, abs=1e-4)

    forces = [
        [4.4703484e-08, -4.2840838e-08, 2.4071064e-02],
        [-4.4703484e-08, -1.4551915e-08, -2.4071217e-02],
        [-1.7881393e-07, 1.0244548e-08, 2.5402933e-02],
        [5.9604645e-08, -2.3283064e-08, -2.5402665e-02],
        [-1.1920929e-07, 6.6356733e-08, -2.1660209e-02],
        [2.3543835e-06, -8.0077443e-06, 9.5508099e-03],
        [-2.2947788e-06, 7.9898164e-06, -9.5513463e-03],
        [-5.9604645e-08, -0.0000000e00, 2.1660626e-02],
    ]
    assert out["f"] == pytest.approx(np.array(forces), rel=1e-4, abs=1e-4)

    stress = [
        [3.3677614e-01, -1.9665707e-07, -5.6416429e-06],
        [4.9939729e-07, 2.4675032e-01, 1.8549043e-05],
        [-4.0414070e-06, 1.9096897e-05, 4.0323928e-02],
    ]
    assert out["s"] == pytest.approx(np.array(stress), rel=1e-4, abs=1e-4)

    magmom = [0.00521, 0.00521, 3.85728, 3.85729, 0.02538, 0.03706, 0.03706, 0.02538]
    assert out["m"] == pytest.approx(magmom, rel=1e-4, abs=1e-4)

    site_energies = [
        -3.8090043,
        -3.8090036,
        -10.2737875,
        -10.2737875,
        -7.659066,
        -7.744509,
        -7.744509,
        -7.659066,
    ]
    assert out["site_energies"] == pytest.approx(site_energies, rel=1e-4, abs=1e-4)
    assert out["site_energies"].shape == (8,)
    assert np.sum(out["site_energies"]) / len(structure) == pytest.approx(
        out["e"], rel=1e-4, abs=1e-6
    )
    assert out["crystal_fea"].mean() == pytest.approx(0.27905, rel=1e-4, abs=1e-4)
    assert out["crystal_fea"].shape == (64,)
    assert out["atom_fea"].mean() == pytest.approx(0.01606, rel=1e-4, abs=1e-4)
    assert out["atom_fea"].shape == (8, 64)


@mark.parametrize("axis", [[0, 0, 1], [1, 1, 0], [-2, 3, 1]])
@mark.parametrize("rotation_angle", [5, 30, 45, 120])
def test_predict_structure_rotated(rotation_angle: float, axis: list) -> None:
    from pymatgen.transformations.standard_transformations import RotationTransformation

    pristine_structure = structure.copy()
    pristine_structure.perturb(0.1)
    pristine_prediction = model.predict_structure(
        pristine_structure, return_site_energies=True
    )

    # Rotation
    rotation_transformation = RotationTransformation(axis=axis, angle=rotation_angle)
    rotated_structure = rotation_transformation.apply_transformation(pristine_structure)
    out = model.predict_structure(rotated_structure, return_site_energies=True)

    assert sorted(out) == ["e", "f", "m", "s", "site_energies"]
    assert out["e"] == pytest.approx(pristine_prediction["e"], rel=1e-4, abs=1e-4)

    # Convert angle to radians
    theta = np.radians(rotation_angle)

    # Normalize the axis
    axis_normalized = axis / np.linalg.norm(axis)
    a, b, c = axis_normalized

    # Compute the skew-symmetric matrix K
    K = np.array([[0, -c, b], [c, 0, -a], [-b, a, 0]])

    # Compute the rotation matrix using Rodrigues' formula
    R = np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * np.dot(K, K)

    rotated_force = pristine_prediction["f"] @ R.transpose()
    assert out["f"] == pytest.approx(rotated_force, rel=1e-4, abs=1e-3)

    rotated_stress = R @ pristine_prediction["s"] @ R.transpose()
    assert out["s"] == pytest.approx(rotated_stress, rel=1e-4, abs=1e-3)

    assert out["m"] == pytest.approx(pristine_prediction["m"], rel=1e-4, abs=1e-4)

    assert out["site_energies"] == pytest.approx(
        pristine_prediction["site_energies"], rel=1e-4, abs=1e-4
    )


def test_predict_supercell() -> None:
    pristine_structure = structure.copy()
    pristine_structure.perturb(0.1)
    pristine_prediction = model.predict_structure(
        pristine_structure, return_site_energies=True
    )
    supercell = pristine_structure.make_supercell([2, 2, 1], in_place=False)
    out = model.predict_structure(supercell, return_site_energies=True)

    assert sorted(out) == ["e", "f", "m", "s", "site_energies"]
    assert out["e"] == pytest.approx(pristine_prediction["e"], rel=1e-4, abs=1e-4)

    assert out["f"] == pytest.approx(
        np.repeat(pristine_prediction["f"], 4, axis=0), rel=1e-4, abs=1e-4
    )

    assert out["s"] == pytest.approx(pristine_prediction["s"], rel=1e-4, abs=1e-4)

    assert out["site_energies"] == pytest.approx(
        np.repeat(pristine_prediction["site_energies"], 4), rel=1e-4, abs=1e-4
    )


def test_predict_batched_structures() -> None:
    pristine_structure = structure.copy()
    pristine_structure.perturb(0.1)
    pristine_prediction = model.predict_structure(
        pristine_structure, return_site_energies=True
    )
    structs = [pristine_structure, pristine_structure, pristine_structure]
    out = model.predict_structure(structs, return_site_energies=True)
    assert len(out) == len(structs)
    for preds in out:
        for property in ["e", "f", "s", "m", "site_energies"]:
            assert preds[property] == pytest.approx(
                pristine_prediction[property], rel=1e-4, abs=1e-4
            )


model_arg_keys = frozenset(
    "atom_fea_dim bond_fea_dim angle_fea_dim composition_model num_radial num_angular n_conv "
    "atom_conv_hidden_dim update_bond bond_conv_hidden_dim update_angle angle_layer_hidden_dim"
    " conv_dropout read_out mlp_hidden_dims mlp_dropout mlp_first is_intensive non_linearity "
    "atom_graph_cutoff bond_graph_cutoff graph_converter_algorithm cutoff_coeff learnable_rbf "
    "skip_connection conv_norm gMLP_norm readout_norm".split()
)


def test_as_to_from_dict() -> None:
    dct = model.as_dict()
    assert {*dct} == {"model_args", "state_dict"}
    assert {*dct["model_args"]} >= model_arg_keys

    model_2 = CHGNet.from_dict(dct)
    assert model_2.as_dict()["model_args"] == dct["model_args"]

    to_dict = model.todict()
    assert {*to_dict} == {"model_name", "model_args"}
    assert {*to_dict["model_args"]} >= model_arg_keys

    model_3 = CHGNet(**to_dict["model_args"])
    assert model_3.todict() == to_dict
