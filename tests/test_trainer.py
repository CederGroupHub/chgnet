from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import patch

import numpy as np
import pytest
import torch
import wandb
from pymatgen.core import Lattice, Structure

from chgnet.data.dataset import StructureData, get_train_val_test_loader
from chgnet.model import CHGNet
from chgnet.trainer.trainer import CombinedLoss, Trainer

if TYPE_CHECKING:
    from pathlib import Path

lattice = Lattice.cubic(4)
species = ["Na", "Cl"]
coords = [[0, 0, 0], [0.5, 0.5, 0.5]]
NaCl = Structure(lattice, species, coords)
structures, energies, forces, stresses, magmoms = [], [], [], [], []
for _ in range(20):
    struct = NaCl.copy()
    struct.perturb(0.1)
    structures.append(struct)
    energies.append(np.random.random(1))
    forces.append(np.random.random([2, 3]))
    stresses.append(np.random.random([3, 3]))
    magmoms.append(np.random.random(2))

# Create some missing labels
energies[10] = np.nan
forces[4] = (np.nan * np.ones((len(structures[4]), 3))).tolist()
stresses[6] = (np.nan * np.ones((3, 3))).tolist()
magmoms[8] = (np.nan * np.ones((len(structures[8]), 1))).tolist()

data = StructureData(
    structures=structures,
    energies=energies,
    forces=forces,
    stresses=stresses,
    magmoms=magmoms,
    shuffle=False,
)
train_loader, val_loader, _test_loader = get_train_val_test_loader(
    data, batch_size=4, train_ratio=0.9, val_ratio=0.05
)
chgnet = CHGNet.load()


def test_combined_loss() -> None:
    criterion = CombinedLoss(
        target_str="ef",
        criterion="MSE",
        energy_loss_ratio=1,
        force_loss_ratio=1,
        stress_loss_ratio=0.1,
        mag_loss_ratio=0.1,
        allow_missing_labels=False,
    )
    target1 = {"e": torch.Tensor([1]), "f": [torch.Tensor([[[1, 1, 1], [2, 2, 2]]])]}
    prediction1 = chgnet.predict_structure(NaCl)
    prediction1 = {
        "e": torch.from_numpy(prediction1["e"]).unsqueeze(0),
        "f": [torch.from_numpy(prediction1["f"])],
        "atoms_per_graph": torch.tensor([2]),
    }
    out1 = criterion(
        targets=target1,
        prediction=prediction1,
    )
    target2 = {
        "e": torch.Tensor([1]),
        "f": [
            torch.Tensor(
                [
                    [
                        [1, 1, 1],
                        [1, 1, 1],
                        [1, 1, 1],
                        [1, 1, 1],
                        [2, 2, 2],
                        [2, 2, 2],
                        [2, 2, 2],
                        [2, 2, 2],
                    ]
                ]
            )
        ],
    }
    supercell = NaCl.make_supercell([2, 2, 1], in_place=False)
    prediction2 = chgnet.predict_structure(supercell)
    prediction2 = {
        "e": torch.from_numpy(prediction2["e"]).unsqueeze(0),
        "f": [torch.from_numpy(prediction2["f"])],
        "atoms_per_graph": torch.tensor([8]),
    }
    out2 = criterion(
        targets=target2,
        prediction=prediction2,
    )
    assert np.isclose(out1["loss"], out2["loss"], rtol=1e-04, atol=1e-05)


def test_trainer(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    extra_run_config = dict(some_other_hyperparam=42)
    trainer = Trainer(
        model=chgnet,
        targets="efsm",
        optimizer="Adam",
        criterion="MSE",
        learning_rate=1e-2,
        epochs=5,
        wandb_path="test/run",
        wandb_init_kwargs=dict(anonymous="must"),
        extra_run_config=extra_run_config,
        allow_missing_labels=True,
    )
    trainer.train(
        train_loader,
        val_loader,
        save_dir=tmp_path,
        save_test_result=tmp_path / "test-preds.json",
    )
    assert dict(wandb.config).items() >= extra_run_config.items()
    for param in chgnet.composition_model.parameters():
        assert param.requires_grad is False
    assert tmp_path.is_dir(), "Training dir was not created"
    for prop in "efsm":
        assert ~np.isnan(trainer.training_history[prop]["train"]).any()
        assert ~np.isnan(trainer.training_history[prop]["val"]).any()
    output_files = [file.name for file in tmp_path.iterdir()]
    for prefix in ("epoch", "bestE_", "bestF_"):
        n_matches = sum(file.startswith(prefix) for file in output_files)
        assert (
            n_matches == 1
        ), f"Expected 1 {prefix} file, found {n_matches} in {output_files}"

    # expect ImportError when passing wandb_path without wandb installed
    err_msg = "Weights and Biases not installed. pip install wandb to use wandb logging"
    with monkeypatch.context() as ctx, pytest.raises(ImportError, match=err_msg):  # noqa: PT012
        ctx.setattr("chgnet.trainer.trainer.wandb", None)
        _ = Trainer(model=chgnet, wandb_path="some-org/some-project")


def test_trainer_composition_model(tmp_path: Path) -> None:
    for param in chgnet.composition_model.parameters():
        assert param.requires_grad is False
    trainer = Trainer(
        model=chgnet,
        targets="efsm",
        optimizer="Adam",
        criterion="MSE",
        learning_rate=1e-2,
        epochs=5,
    )
    initial_weights = chgnet.composition_model.state_dict()["fc.weight"].clone()
    trainer.train(
        train_loader, val_loader, save_dir=tmp_path, train_composition_model=True
    )
    for param in chgnet.composition_model.parameters():
        assert param.requires_grad is True

    output_files = list(tmp_path.iterdir())
    weights_path = next(file for file in output_files if file.name.startswith("epoch"))
    new_chgnet = CHGNet.from_file(weights_path)
    for param in new_chgnet.composition_model.parameters():
        assert param.requires_grad is False
    comparison = new_chgnet.composition_model.state_dict()["fc.weight"].to(
        "cpu"
    ) == initial_weights.to("cpu")
    expect = torch.ones_like(comparison)
    # Only Na and Cl should have updated
    expect[0][10] = 0
    expect[0][16] = 0
    assert torch.all(comparison == expect)


@pytest.fixture
def mock_wandb():
    with patch("chgnet.trainer.trainer.wandb") as mock:
        yield mock


def test_wandb_init(mock_wandb):
    chgnet = CHGNet.load()
    _trainer = Trainer(
        model=chgnet,
        wandb_path="test-project/test-run",
        wandb_init_kwargs={"tags": ["test"]},
    )
    expected_config = {
        "targets": "ef",
        "energy_loss_ratio": 1,
        "force_loss_ratio": 1,
        "stress_loss_ratio": 0.1,
        "mag_loss_ratio": 0.1,
        "optimizer": "Adam",
        "scheduler": "CosLR",
        "criterion": "MSE",
        "epochs": 50,
        "starting_epoch": 0,
        "learning_rate": 0.001,
        "print_freq": 100,
        "torch_seed": None,
        "data_seed": None,
        "use_device": None,
        "check_cuda_mem": False,
        "wandb_path": "test-project/test-run",
        "wandb_init_kwargs": {"tags": ["test"]},
        "extra_run_config": None,
        "allow_missing_labels": True,
    }
    mock_wandb.init.assert_called_once_with(
        project="test-project", name="test-run", config=expected_config, tags=["test"]
    )


def test_wandb_log_frequency(tmp_path, mock_wandb):
    trainer = Trainer(model=chgnet, wandb_path="test-project/test-run", epochs=1)

    # Test epoch logging
    trainer.train(train_loader, val_loader, wandb_log_freq="epoch", save_dir=tmp_path)
    assert (
        mock_wandb.log.call_count == 2 * trainer.epochs
    ), "Expected one train and one val log per epoch"

    mock_wandb.log.reset_mock()

    # Test batch logging
    trainer.train(train_loader, val_loader, wandb_log_freq="batch", save_dir=tmp_path)
    expected_batch_calls = trainer.epochs * len(train_loader)
    assert (
        mock_wandb.log.call_count > expected_batch_calls
    ), "Expected more calls for batch logging"

    # Test log content (for both epoch and batch logging)
    for call_args in mock_wandb.log.call_args_list:
        logged_data = call_args[0][0]
        assert isinstance(logged_data, dict), "Logged data should be a dictionary"
        assert any(
            key.endswith("_mae") for key in logged_data
        ), "Logged data should contain MAE metrics"

    mock_wandb.log.reset_mock()

    # Test no logging when wandb_path is not provided
    trainer_no_wandb = Trainer(model=chgnet, epochs=1)
    trainer_no_wandb.train(train_loader, val_loader, save_dir=tmp_path)
    mock_wandb.log.assert_not_called()
