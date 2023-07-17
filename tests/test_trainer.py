from __future__ import annotations

import numpy as np
from pymatgen.core import Lattice, Structure

from chgnet.data.dataset import StructureData, get_train_val_test_loader
from chgnet.model import CHGNet
from chgnet.trainer import Trainer

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


def test_trainer(tmp_path) -> None:
    chgnet = CHGNet.load()
    train_loader, val_loader, test_loader = get_train_val_test_loader(
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
    dir_name = "test_tmp_dir"
    test_dir = tmp_path / dir_name
    trainer.train(train_loader, val_loader, save_dir=test_dir)
    assert test_dir.is_dir(), "Training dir was not created"

    output_files = list(test_dir.iterdir())
    for prefix in ("epoch", "bestE", "bestF"):
        n_matches = sum(file.name.startswith(prefix) for file in output_files)
        assert n_matches == 1
