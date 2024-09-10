from __future__ import annotations

import random

import numpy as np
import pytest
import torch
from pymatgen.core import Lattice, Structure

from chgnet.data.dataset import StructureData, get_train_val_test_loader
from chgnet.graph import CrystalGraph

lattice = Lattice.cubic(4)
species = ["Na", "Cl"]
coords = [[0, 0, 0], [0.5, 0.5, 0.5]]
NaCl = Structure(lattice, species, coords)


@pytest.fixture
def structure_data() -> StructureData:
    """Create a graph with 3 nodes and 3 directed edges."""
    random.seed(42)
    structures, energies, forces = [], [], []
    stresses, magmoms, structure_ids = [], [], []

    for index in range(100):
        structures += [NaCl.copy().perturb(0.1)]
        energies += [np.random.random(1)]
        forces += [np.random.random([2, 3])]
        stresses += [np.random.random([3, 3])]
        magmoms += [np.random.random([2, 1])]
        structure_ids += [index]

    return StructureData(
        structures=structures,
        energies=energies,
        forces=forces,
        stresses=stresses,
        magmoms=magmoms,
        structure_ids=structure_ids,
    )


def test_structure_data(structure_data: StructureData) -> None:
    get_one = structure_data[0]
    assert isinstance(get_one[0], CrystalGraph)
    assert isinstance(get_one[0].mp_id, int)
    assert get_one[0].mp_id == 42
    assert isinstance(get_one[1], dict)
    assert isinstance(get_one[1]["e"], torch.Tensor)
    assert isinstance(get_one[1]["f"], torch.Tensor)
    assert isinstance(get_one[1]["s"], torch.Tensor)
    assert isinstance(get_one[1]["m"], torch.Tensor)


def test_data_loader(structure_data: StructureData) -> None:
    train_loader, _val_loader, _test_loader = get_train_val_test_loader(
        structure_data, batch_size=16, train_ratio=0.9, val_ratio=0.05
    )
    graphs, targets = next(iter(train_loader))
    assert isinstance(train_loader, torch.utils.data.DataLoader)
    assert len(graphs) == 16
    assert isinstance(graphs[0], CrystalGraph)
    assert targets["e"].shape == (16,)
    assert len(targets["f"]) == 16
    assert targets["f"][0].shape == (2, 3)
    assert len(targets["s"]) == 16
    assert targets["s"][0].shape == (3, 3)
    assert len(targets["m"]) == 16
    assert targets["m"][0].shape == (2, 1)


def test_structure_data_inconsistent_length():
    # https://github.com/CederGroupHub/chgnet/pull/69
    structures = [NaCl.copy() for _ in range(5)]
    energies = [np.random.random(1) for _ in range(5)]
    forces = [np.random.random([2, 3]) for _ in range(4)]
    with pytest.raises(RuntimeError) as exc:
        StructureData(structures=structures, energies=energies, forces=forces)

    assert (
        str(exc.value)
        == f"Inconsistent number of structures and labels: {len(structures)=}, "
        f"{len(forces)=}"
    )


def test_dataset_no_shuffling():
    n_samples = 100
    structure_ids = list(range(n_samples))

    structure_data = StructureData(
        structures=[NaCl.copy() for _ in range(n_samples)],
        energies=np.random.random(n_samples),
        forces=np.random.random([n_samples, 2, 3]),
        stresses=np.random.random([n_samples, 3, 3]),
        magmoms=np.random.random([n_samples, 2, 1]),
        structure_ids=structure_ids,
        shuffle=False,
    )
    sample_ids = [data[0].mp_id for data in structure_data]
    # shuffle=False means structure_ids should be in order
    assert sample_ids == structure_ids
