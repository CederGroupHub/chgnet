from __future__ import annotations

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


@pytest.fixture()
def structure_data() -> StructureData:
    """Create a graph with 3 nodes and 3 directed edges."""
    structures, energies, forces, stresses, magmoms = [], [], [], [], []
    for _ in range(100):
        struct = NaCl.copy()
        struct.perturb(0.1)
        structures.append(struct)
        energies.append(np.random.random(1))
        forces.append(np.random.random([2, 3]))
        stresses.append(np.random.random([3, 3]))
        magmoms.append(np.random.random([2, 1]))
    return StructureData(
        structures=structures,
        energies=energies,
        forces=forces,
        stresses=stresses,
        magmoms=magmoms,
    )


def test_structure_data(structure_data: StructureData) -> None:
    get_one = structure_data[0]
    assert isinstance(get_one[0], CrystalGraph)
    assert isinstance(get_one[1], dict)
    assert isinstance(get_one[1]["e"], torch.Tensor)
    assert isinstance(get_one[1]["f"], torch.Tensor)
    assert isinstance(get_one[1]["s"], torch.Tensor)
    assert isinstance(get_one[1]["m"], torch.Tensor)


def test_data_loader(structure_data: StructureData) -> None:
    train_loader, val_loader, test_loader = get_train_val_test_loader(
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
