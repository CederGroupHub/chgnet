from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import patch

import numpy as np
import pytest
import torch
import torch.nn as nn
import wandb
from pymatgen.core import Lattice, Structure

from chgnet.data.dataset import StructureData, get_train_val_test_loader
from chgnet.model import CHGNet
from chgnet.trainer.trainer import CombinedLoss, Trainer
from chgnet.utils import AverageMeter

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


@pytest.mark.parametrize("target", ["e", "f", "s"])
@pytest.mark.parametrize("criterion", ["MSE", "MAE", "Huber"])
def test_combined_loss_empty_labels(target: str, criterion: str) -> None:
    """An unsupervised target contributes zero, including during backpropagation."""
    shape = {"e": (2,), "f": (2, 3), "s": (3, 3)}[target]
    labels = torch.full(shape, torch.nan, dtype=torch.float64)
    pred = torch.full(shape, torch.nan, dtype=torch.float64, requires_grad=True)
    out = CombinedLoss(target_str=target, criterion=criterion)(
        {target: labels if target == "e" else [labels]},
        {target: pred if target == "e" else [pred]},
    )
    assert out["loss"].item() == 0
    assert out["loss"].dtype == pred.dtype
    assert out[f"{target}_MAE"].item() == 0
    assert out[f"{target}_MAE_size"] == 0
    out["loss"].backward()
    torch.testing.assert_close(pred.grad, torch.zeros_like(pred))


@pytest.mark.parametrize(
    ("target", "shape", "ratio"),
    [("e", (2,), 1), ("f", (2, 3), 1), ("s", (3, 3), 0.1)],
)
def test_combined_loss_partial_labels(
    target: str, shape: tuple[int, ...], ratio: float
) -> None:
    """Only supervised entries affect the loss, gradient and metric denominator."""
    labels = torch.full(shape, torch.nan)
    labels.view(-1)[0] = 0
    pred = torch.full(shape, 100.0)
    pred.view(-1)[0] = 2
    pred.requires_grad_()
    out = CombinedLoss(target_str=target)(
        {target: labels if target == "e" else [labels]},
        {target: pred if target == "e" else [pred]},
    )
    assert out["loss"].item() == pytest.approx(4 * ratio)
    assert out[f"{target}_MAE"].item() == 2
    assert out[f"{target}_MAE_size"] == 1
    out["loss"].backward()
    expected_grad = torch.zeros_like(pred)
    expected_grad.view(-1)[0] = 4 * ratio
    torch.testing.assert_close(pred.grad, expected_grad)


def test_energy_mae_uses_valid_labels_across_batches() -> None:
    """Epoch MAE is independent of batching, including an entirely unlabelled batch."""
    labels = torch.tensor([0.0, torch.nan, 0.0, 0.0, torch.nan])
    pred = torch.tensor([2.0, torch.nan, 4.0, 6.0, torch.nan])
    criterion = CombinedLoss(target_str="e")
    meter = AverageMeter()
    for batch in (slice(0, 2), slice(2, 4), slice(4, 5)):
        out = criterion({"e": labels[batch]}, {"e": pred[batch]})
        meter.update(out["e_MAE"].item(), out["e_MAE_size"])
    assert meter.count == 3
    assert meter.avg == 4
    whole_batch = criterion({"e": labels}, {"e": pred})
    assert whole_batch["e_MAE_size"] == 3
    assert whole_batch["e_MAE"].item() == meter.avg


@pytest.mark.parametrize("criterion", ["MSE", "MAE", "Huber"])
def test_combined_loss_fully_labeled_matches_direct_criterion(criterion: str) -> None:
    """Fully labeled efs loss and gradients match the weighted direct PyTorch criteria."""
    prediction = {
        "e": torch.tensor([1.0, -2.0, 3.0, -4.0], requires_grad=True),
        "f": [torch.tensor([[0.5, 1.5, -2.5]], requires_grad=True)],
        "s": [torch.diag(torch.tensor([1.0, 2.0, 3.0])).requires_grad_(True)],
    }
    targets = {
        "e": torch.tensor([0.5, 1.0, -1.0, 2.0]),
        "f": [torch.tensor([[0.0, 1.0, -2.0]])],
        "s": [torch.diag(torch.tensor([2.0, 4.0, 6.0]))],
    }
    loss_fn = CombinedLoss(
        target_str="efs",
        criterion=criterion,
        energy_loss_ratio=2.0,
        force_loss_ratio=3.0,
        stress_loss_ratio=0.5,
    )
    out = loss_fn(targets, prediction)

    crit = {
        "MSE": nn.MSELoss(),
        "MAE": nn.L1Loss(),
        "Huber": nn.HuberLoss(delta=0.1),  # CombinedLoss default delta
    }[criterion]
    direct_pred = {
        "e": prediction["e"].detach().clone().requires_grad_(True),
        "f": [prediction["f"][0].detach().clone().requires_grad_(True)],
        "s": [prediction["s"][0].detach().requires_grad_(True)],
    }
    direct = (
        2.0 * crit(direct_pred["e"], targets["e"])
        + 3.0
        * crit(
            torch.cat(direct_pred["f"], dim=0),
            torch.cat(targets["f"], dim=0),
        )
        + 0.5
        * crit(
            torch.cat(direct_pred["s"], dim=0),
            torch.cat(targets["s"], dim=0),
        )
    )
    assert torch.allclose(out["loss"], direct)

    out["loss"].backward()
    direct.backward()
    torch.testing.assert_close(prediction["e"].grad, direct_pred["e"].grad)
    torch.testing.assert_close(prediction["f"][0].grad, direct_pred["f"][0].grad)
    torch.testing.assert_close(prediction["s"][0].grad, direct_pred["s"][0].grad)


def test_combined_loss_all_missing_combined_batch_backpropagates_zeros() -> None:
    """An efs batch without any labels still backpropagates zero gradients."""
    nan = float("nan")
    targets = {
        "e": torch.tensor([nan, nan]),
        "f": [torch.full((2, 3), nan)],
        "s": [torch.full((3, 3), nan)],
    }
    prediction = {
        "e": torch.tensor([1.0, 2.0], requires_grad=True),
        "f": [torch.full((2, 3), nan, requires_grad=True)],
        "s": [torch.full((3, 3), nan, requires_grad=True)],
    }
    out = CombinedLoss(target_str="efs")(targets, prediction)
    assert isinstance(out["loss"], torch.Tensor)
    assert out["loss"].item() == 0
    out["loss"].backward()
    for tensor in (prediction["e"], prediction["f"][0], prediction["s"][0]):
        assert torch.all(tensor.grad == 0)


def test_combined_loss_no_missing_labels_keeps_nan() -> None:
    """With allow_missing_labels=False an unlabeled target still yields a NaN loss."""
    out = CombinedLoss(target_str="e", allow_missing_labels=False)(
        {"e": torch.tensor([0.0, torch.nan])},
        {"e": torch.tensor([2.0, 3.0], requires_grad=True)},
    )
    assert torch.isnan(out["loss"])
    assert torch.isnan(out["e_MAE"])
    assert out["e_MAE_size"] == 2


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
