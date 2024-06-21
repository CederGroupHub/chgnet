from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest
import torch
import wandb
from pymatgen.core import Lattice, Structure

from chgnet.data.dataset import StructureData, get_train_val_test_loader
from chgnet.model import CHGNet
from chgnet.trainer import Trainer

if TYPE_CHECKING:
    from pathlib import Path

lattice = Lattice.cubic(4)
species = ["Na", "Cl"]
coords = [[0, 0, 0], [0.5, 0.5, 0.5]]
NaCl = Structure(lattice, species, coords)
structures, energies, forces, stresses, magmoms = [], [], [], [], []
for _ in range(100):
    struct = NaCl.copy()
    struct.perturb(0.1)
    structures.append(struct)
    energies.append(np.random.random(1))
    forces.append(np.random.random([2, 3]))
    stresses.append(np.random.random([3, 3]))
    magmoms.append(np.random.random(2))

data = StructureData(
    structures=structures,
    energies=energies,
    forces=forces,
    stresses=stresses,
    magmoms=magmoms,
)


def test_trainer(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    chgnet = CHGNet.load()
    train_loader, val_loader, _test_loader = get_train_val_test_loader(
        data, batch_size=16, train_ratio=0.9, val_ratio=0.05
    )
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
    chgnet = CHGNet.load()
    for param in chgnet.composition_model.parameters():
        assert param.requires_grad is False
    train_loader, val_loader, _test_loader = get_train_val_test_loader(
        data, batch_size=16, train_ratio=0.9, val_ratio=0.05
    )
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
