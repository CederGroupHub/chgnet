from __future__ import annotations

import copy
import pickle
from typing import TYPE_CHECKING

import numpy as np
import pytest
import torch
from ase.build import bulk
from ase.units import GPa
from pymatgen.io.ase import AseAtomsAdaptor
from torch.utils.data import DataLoader

from chgnet.data.dataset import StructureData, collate_graphs
from chgnet.model import CHGNet, StructOptimizer
from chgnet.model.dynamics import CHGNetCalculator
from chgnet.trainer import Trainer

if TYPE_CHECKING:
    from pathlib import Path

    from ase import Atoms


@pytest.fixture
def silicon() -> Atoms:
    atoms = bulk("Si", "diamond", a=5.43)
    atoms.positions[1] += [0.02, -0.01, 0.03]
    return atoms


@pytest.mark.parametrize("transform", ["identity", "swap", "negate", "reflect"])
def test_stress_matches_energy_derivative(silicon: Atoms, transform: str) -> None:
    model = CHGNet.load()
    changed = silicon.copy()
    reflection = np.eye(3)
    if transform == "swap":
        changed.set_cell(silicon.cell[[1, 0, 2]], scale_atoms=False)
    elif transform == "negate":
        changed.set_cell(np.diag([-1, 1, 1]) @ silicon.cell, scale_atoms=False)
    elif transform == "reflect":
        reflection[0, 0] = -1
        changed.set_cell(silicon.cell @ reflection, scale_atoms=False)
        changed.positions[:] = silicon.positions @ reflection

    structures = [AseAtomsAdaptor.get_structure(frame) for frame in (silicon, changed)]
    original, transformed = model.predict_structure(structures, task="efs")
    for key, expected in {
        "e": original["e"],
        "f": original["f"] @ reflection,
        "s": reflection @ original["s"] @ reflection,
    }.items():
        if not np.allclose(transformed[key], expected, rtol=1e-3, atol=1e-4):
            pytest.fail(f"{transform} changed {key}: {transformed[key]} != {expected}")

    changed.calc = CHGNetCalculator(
        model=model, use_device=str(next(model.parameters()).device)
    )
    stress = changed.get_stress(voigt=False)
    if not np.allclose(stress, transformed["s"] * GPa, rtol=1e-4, atol=1e-6):
        pytest.fail("ASE stress differs from the batched model prediction")

    # dE/d(strain) / V uses the physical volume for either lattice handedness.
    epsilon = 0.002
    for row, col in ((0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2)):
        strain = np.zeros((3, 3))
        strain[row, col] += 0.5
        strain[col, row] += 0.5
        energies = []
        for sign in (-1, 1):
            displaced = changed.copy()
            displaced.set_cell(
                changed.cell @ (np.eye(3) + sign * epsilon * strain), scale_atoms=True
            )
            displaced.calc = changed.calc
            energies.append(displaced.get_potential_energy())
        derivative = (energies[1] - energies[0]) / (2 * epsilon * changed.get_volume())
        if not np.isclose(stress[row, col], derivative, rtol=0.01, atol=1e-4):
            pytest.fail(f"stress[{row}, {col}]={stress[row, col]} != dE/V={derivative}")


def test_cell_relaxation_preserves_basis_equivalence(tmp_path: Path) -> None:
    model = CHGNet.load()
    results = []
    for order in ([0, 1, 2], [1, 0, 2]):
        atoms = bulk("Si", "diamond", a=5.7)
        atoms.set_cell(atoms.cell[order], scale_atoms=False)
        initial_determinant = np.linalg.det(atoms.cell)
        path = tmp_path / f"relax-{order[0]}.pkl"
        result = StructOptimizer(
            model=model, use_device=str(next(model.parameters()).device)
        ).relax(atoms, steps=5, fmax=0.001, save_path=path, verbose=False)
        with path.open("rb") as file:
            saved = pickle.load(file)
        if saved["energy"][-1] >= saved["energy"][0]:
            pytest.fail(f"Cell relaxation increased energy for lattice order {order}")
        final = AseAtomsAdaptor.get_atoms(result["final_structure"])
        final.calc = CHGNetCalculator(
            model=model, use_device=str(next(model.parameters()).device)
        )
        if not np.isclose(final.get_potential_energy(), saved["energy"][-1], atol=1e-5):
            pytest.fail("Saved relaxation energy differs from the returned structure")
        if not np.allclose(final.cell, saved["cell"][-1], atol=1e-6):
            pytest.fail("Saved relaxation cell differs from the returned structure")
        if np.linalg.det(final.cell) * initial_determinant <= 0:
            pytest.fail("Cell relaxation changed lattice handedness")
        results.append(saved)
    if not np.allclose(results[0]["energy"], results[1]["energy"], atol=1e-4):
        pytest.fail("Equivalent lattice bases followed different relaxation energies")


def test_stress_training_and_checkpoint_preserve_basis_equivalence(
    silicon: Atoms, tmp_path: Path
) -> None:
    model = CHGNet.load()
    device = str(next(model.parameters()).device)
    original = AseAtomsAdaptor.get_structure(silicon)
    swapped = silicon.copy()
    swapped.set_cell(silicon.cell[[1, 0, 2]], scale_atoms=False)
    left_handed = AseAtomsAdaptor.get_structure(swapped)
    gradients = []
    predictions = []
    for name, second in (("right", original), ("mixed", left_handed)):
        dataset = StructureData(
            structures=[original, second],
            energies=[0.0, 0.0],
            forces=[np.zeros((2, 3)), np.zeros((2, 3))],
            # StructureData converts VASP kbar to GPa, giving a +1 GPa label.
            stresses=[-10 * np.eye(3), -10 * np.eye(3)],
            shuffle=False,
        )
        loader = DataLoader(dataset, batch_size=2, collate_fn=collate_graphs)
        trainer = Trainer(
            model=copy.deepcopy(model),
            targets="efs",
            energy_loss_ratio=0,
            force_loss_ratio=0,
            stress_loss_ratio=1,
            optimizer="SGD",
            momentum=0,
            epochs=1,
            learning_rate=1e-4,
            use_device=device,
        )
        save_dir = tmp_path / name
        trainer.train(loader, loader, save_dir=save_dir)
        gradient = torch.cat(
            [p.grad.flatten() for p in trainer.model.parameters() if p.grad is not None]
        )
        if not torch.isfinite(gradient).all() or gradient.norm() == 0:
            pytest.fail("Stress training did not produce finite, nonzero gradients")
        gradients.append(gradient)
        checkpoint = next(save_dir.glob("epoch*.pth.tar"))
        reloaded = CHGNet.from_file(checkpoint).to(device)
        if all(
            torch.equal(before, after)
            for before, after in zip(
                model.parameters(), reloaded.parameters(), strict=True
            )
        ):
            pytest.fail("Stress training did not update the saved model weights")
        expected = trainer.model.predict_structure(original, task="efs")
        actual = reloaded.predict_structure(original, task="efs")
        for key in "efs":
            if not np.allclose(actual[key], expected[key], rtol=1e-4, atol=1e-5):
                pytest.fail(f"Checkpoint reload changed {key}")
        predictions.append(actual)
    if (gradients[0] - gradients[1]).norm() > 1e-3 * gradients[0].norm():
        pytest.fail("Equivalent lattice bases produced different training gradients")
    for key in "efs":
        if not np.allclose(
            predictions[0][key], predictions[1][key], rtol=1e-3, atol=1e-4
        ):
            pytest.fail(f"Equivalent lattice bases trained different {key} predictions")
