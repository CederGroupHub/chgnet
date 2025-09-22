from __future__ import annotations

import inspect

import numpy as np
import pytest
from pymatgen.core import Lattice, Structure

from chgnet import ROOT
from chgnet.graph import CrystalGraphConverter
from chgnet.model.model import CHGNet

structure = Structure.from_file(f"{ROOT}/examples/mp-18767-LiMnO2.cif")
graph = CrystalGraphConverter()(structure, graph_id="test-model")
model = CHGNet.load()


@pytest.mark.parametrize("atom_fea_dim", [1, 64])
@pytest.mark.parametrize("bond_fea_dim", [1, 64])
@pytest.mark.parametrize("angle_fea_dim", [1, 64])
@pytest.mark.parametrize("num_radial", [1, 9])
@pytest.mark.parametrize("num_angular", [1, 9])
@pytest.mark.parametrize("n_conv", [1, 4])
@pytest.mark.parametrize("composition_model", ["MPtrj", "MPtrj_e", "MPF"])
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
    assert sorted(out) == ["atom_fea", "crystal_fea", *"efms", "site_energies"]
    assert out["e"] == pytest.approx(-7.36769, rel=1e-4, abs=1e-4)

    forces = [
        [1.34110451e-07, -2.92202458e-08, 2.38135569e-02],
        [5.96046448e-08, 4.63332981e-08, -2.38130391e-02],
        [8.94069672e-08, -2.06753612e-07, 9.25870836e-02],
        [-1.49011612e-07, -1.06170774e-07, -9.25877392e-02],
        [5.96046448e-08, 2.00234354e-08, -2.43449211e-03],
        [-1.19209290e-06, -4.74974513e-08, -1.30698681e-02],
        [1.40070915e-06, 1.64378434e-07, 1.30702555e-02],
        [-5.96046448e-08, 1.66241080e-07, 2.43446976e-03],
    ]
    assert out["f"] == pytest.approx(np.array(forces), rel=1e-3, abs=1e-4)

    stress = [
        [-3.0366361e-01, -3.7709856e-07, 2.2964025e-06],
        [-1.2128221e-06, 2.2305478e-01, -3.2104114e-07],
        [1.3322200e-06, -8.3219516e-07, -1.0736181e-01],
    ]
    assert out["s"] == pytest.approx(np.array(stress), rel=5e-3, abs=1e-4)

    magmom = [
        3.0495524e-03,
        3.0494630e-03,
        3.8694179e00,
        3.8694181e00,
        4.4136152e-02,
        3.8622141e-02,
        3.8622111e-02,
        4.4136211e-02,
    ]
    assert out["m"] == pytest.approx(magmom, rel=1e-3, abs=1e-4)

    site_energies = [
        -3.6264274,
        -3.6264274,
        -9.634681,
        -9.634682,
        -8.024935,
        -8.184724,
        -8.184724,
        -8.024935,
    ]
    assert out["site_energies"] == pytest.approx(site_energies, rel=1e-4, abs=1e-4)
    assert out["site_energies"].shape == (8,)
    assert np.sum(out["site_energies"]) / len(structure) == pytest.approx(
        out["e"], rel=1e-4, abs=1e-6
    )
    assert out["crystal_fea"].mean() == pytest.approx(0.26999, rel=1e-4, abs=1e-4)
    assert out["crystal_fea"].shape == (64,)
    assert out["atom_fea"].mean() == pytest.approx(-0.09668, rel=1e-4, abs=1e-4)
    assert out["atom_fea"].shape == (8, 64)


@pytest.mark.parametrize("axis", [[0, 0, 1], [1, 1, 0], [-2, 3, 1]])
@pytest.mark.parametrize("rotation_angle", [5, 30, 45, 120])
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
    skew_mat = np.array([[0, -c, b], [c, 0, -a], [-b, a, 0]])

    # Compute the rotation matrix using Rodrigues' formula
    rot_mat = (
        np.eye(3)
        + np.sin(theta) * skew_mat
        + (1 - np.cos(theta)) * np.dot(skew_mat, skew_mat)
    )

    rotated_force = pristine_prediction["f"] @ rot_mat.transpose()
    assert out["f"] == pytest.approx(rotated_force, rel=1e-3, abs=1e-3)

    rotated_stress = rot_mat @ pristine_prediction["s"] @ rot_mat.transpose()
    assert out["s"] == pytest.approx(rotated_stress, rel=1e-3, abs=1e-3)

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

    assert out["s"] == pytest.approx(pristine_prediction["s"], rel=1e-3, abs=1e-3)

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
        for prop in ["e", "f", "s", "m", "site_energies"]:
            assert preds[prop] == pytest.approx(
                pristine_prediction[prop], rel=1e-3, abs=1e-3
            )


def test_predict_isolated_structures() -> None:
    lattice10 = Lattice.cubic(10)
    lattice20 = Lattice.cubic(20)
    positions = [[0, 0, 0], [0.5, 0.5, 0.5]]

    # Create the structure
    model.graph_converter.set_isolated_atom_response("ignore")
    prediction10 = model.predict_structure(Structure(lattice10, ["H", "H"], positions))
    prediction20 = model.predict_structure(Structure(lattice20, ["H", "H"], positions))
    assert prediction10["e"] == pytest.approx(prediction20["e"], rel=1e-5, abs=1e-5)


def test_as_to_from_dict() -> None:
    dct = model.as_dict()
    assert {*dct} == {"model_args", "state_dict"}

    model_2 = CHGNet.from_dict(dct)
    assert model_2.as_dict()["model_args"] == dct["model_args"]

    to_dict = model.todict()
    assert {*to_dict} == {"model_name", "model_args"}

    model_3 = CHGNet(**to_dict["model_args"])
    assert model_3.todict() == to_dict


def test_model_load_version_params(
    capsys: pytest.CaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    model = CHGNet.load(use_device="cpu")
    v030_key, v030_params = "0.3.0", 412_525
    assert model.version == v030_key
    assert model.n_params == v030_params
    stdout, stderr = capsys.readouterr()

    assert stdout == (
        f"CHGNet v{v030_key} initialized with {v030_params:,} parameters\n"
        "CHGNet will run on cpu\n"
    )
    assert stderr == ""

    v020_key, v020_params = "0.2.0", 400_438
    model = CHGNet.load(model_name=v020_key, use_device="cpu")
    assert model.version == v020_key
    assert model.n_params == v020_params
    stdout, stderr = capsys.readouterr()
    assert stdout == (
        f"CHGNet v{v020_key} initialized with {v020_params:,} parameters\n"
        "CHGNet will run on cpu\n"
    )
    assert stderr == ""

    model_name = "0.1.0"  # invalid
    with pytest.raises(ValueError, match=f"Unknown {model_name=}"):
        CHGNet.load(model_name=model_name)

    bad_env_device = "foobar"
    err_msg = f"Expected one of cpu, .+type at start of device string: {bad_env_device}"
    with (  # noqa: PT012
        monkeypatch.context() as ctx,
        pytest.raises(RuntimeError, match=err_msg),
    ):
        ctx.setenv("CHGNET_DEVICE", bad_env_device)
        CHGNet.load()

    # check check_cuda_mem defaults to False
    inspect_signature = inspect.signature(CHGNet.load)
    assert inspect_signature.parameters["check_cuda_mem"].default is False


def test_model_load_r2scan(capsys: pytest.CaptureFixture) -> None:
    """Test loading the r2scan pretrained model."""
    model = CHGNet.load(model_name="r2scan", use_device="cpu")
    r2scan_key, r2scan_params = "r2scan", 412_525
    assert model.version == r2scan_key
    assert model.n_params == r2scan_params
    stdout, stderr = capsys.readouterr()

    assert stdout == (
        f"CHGNet v{r2scan_key} initialized with {r2scan_params:,} parameters\n"
        "CHGNet will run on cpu\n"
    )
    assert stderr == ""


def test_model_load_all_pretrained_models() -> None:
    """Test loading all three pretrained models."""
    # Test default model (0.3.0)
    model_030 = CHGNet.load(use_device="cpu")
    assert model_030.version == "0.3.0"
    assert model_030.n_params == 412_525

    # Test 0.2.0 model
    model_020 = CHGNet.load(model_name="0.2.0", use_device="cpu")
    assert model_020.version == "0.2.0"
    assert model_020.n_params == 400_438

    # Test r2scan model
    model_r2scan = CHGNet.load(model_name="r2scan", use_device="cpu")
    assert model_r2scan.version == "r2scan"
    assert model_r2scan.n_params == 412_525

    # Test that all models can make predictions
    from pymatgen.core import Structure

    from chgnet import ROOT

    structure = Structure.from_file(f"{ROOT}/examples/mp-18767-LiMnO2.cif")
    converter = CrystalGraphConverter()
    graph = converter(structure, graph_id="test-all-models")

    # Test prediction with all models
    for model in [model_030, model_020, model_r2scan]:
        prediction = model.predict_graph(graph, task="e")
        assert "e" in prediction
        # prediction["e"] is a numpy array, convert to float for assertion
        energy = float(prediction["e"])
        assert isinstance(energy, float)
        assert energy < 0  # Energy should be negative
