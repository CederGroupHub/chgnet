import os
from pathlib import Path

from ase import Atoms
from ase.md.nvtberendsen import NVTBerendsen
from pymatgen.core import Structure
from pytest import MonkeyPatch

from chgnet.model import StructOptimizer
from chgnet.model.dynamics import CHGNetCalculator, MolecularDynamics
from chgnet.model.model import CHGNet

relaxer = StructOptimizer()
structure = Structure.from_file("examples/o-LiMnO2_unit.cif")
chgnet = CHGNet.load()


def test_relaxation(tmp_path: Path, monkeypatch: MonkeyPatch):
    # cd into the temporary directory
    monkeypatch.chdir(tmp_path)

    md = MolecularDynamics(
        atoms=structure,
        model=chgnet,
        ensemble="nvt",
        compressibility_au=1.6,
        temperature=1000,  # in k
        timestep=2,  # in fs
        trajectory="md_out.traj",
        logfile="md_out.log",
        loginterval=100,
        use_device="cpu",
    )
    md.run(50)  # run a 0.1 ps MD simulation

    assert isinstance(md.atoms, Atoms)
    assert isinstance(md.atoms.calc, CHGNetCalculator)
    assert isinstance(md.dyn, NVTBerendsen)
    assert os.path.isfile("md_out.traj")
    assert os.path.isfile("md_out.log")
    with open("md_out.log") as log_file:
        logs = log_file.read()
    assert logs == (
        "Time[ps]      Etot[eV]     Epot[eV]     Ekin[eV]    T[K]\n"
        "0.0000         -58.9727     -58.9727       0.0000     0.0\n"
    )
