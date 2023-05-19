from __future__ import annotations

import os
from typing import TYPE_CHECKING

from ase import Atoms
from ase.md.nptberendsen import Inhomogeneous_NPTBerendsen
from ase.md.nvtberendsen import NVTBerendsen
from pymatgen.core import Structure
from pytest import MonkeyPatch, approx

from chgnet import ROOT
from chgnet.model import StructOptimizer
from chgnet.model.dynamics import CHGNetCalculator, EquationOfState, MolecularDynamics
from chgnet.model.model import CHGNet

if TYPE_CHECKING:
    from pathlib import Path

relaxer = StructOptimizer()
structure = Structure.from_file(f"{ROOT}/examples/o-LiMnO2_unit.cif")
chgnet = CHGNet.load()


def test_eos():
    eos = EquationOfState()
    eos.fit(atoms=structure)
    print(eos.get_bulk_mudulus())
    print(eos.get_bulk_mudulus(unit="GPa"))
    print(eos.get_compressibility())
    print(eos.get_compressibility(unit="GPa^-1"))
    assert eos.get_bulk_mudulus() == approx(0.6621170816, rel=1e-5)
    assert eos.get_bulk_mudulus(unit="GPa") == approx(106.08285172, rel=1e-5)
    assert eos.get_compressibility() == approx(1.510306904, rel=1e-5)
    assert eos.get_compressibility(unit="GPa^-1") == approx(0.009426594, rel=1e-5)


def test_md_nvt(tmp_path: Path, monkeypatch: MonkeyPatch):
    # cd into the temporary directory
    monkeypatch.chdir(tmp_path)

    md = MolecularDynamics(
        atoms=structure,
        model=chgnet,
        ensemble="nvt",
        temperature=1000,  # in k
        timestep=2,  # in fs
        trajectory="md_out.traj",
        logfile="md_out.log",
        loginterval=100,
        use_device="cpu",
    )
    md.run(10)

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


def test_md_npt_inhomogeneous_berendsen(tmp_path: Path, monkeypatch: MonkeyPatch):
    # cd into the temporary directory
    monkeypatch.chdir(tmp_path)

    md = MolecularDynamics(
        atoms=structure,
        model=chgnet,
        ensemble="npt",
        temperature=1000,  # in k
        timestep=2,  # in fs
        trajectory="md_out.traj",
        logfile="md_out.log",
        loginterval=100,
    )
    md.run(10)

    assert isinstance(md.atoms, Atoms)
    assert isinstance(md.atoms.calc, CHGNetCalculator)
    assert isinstance(md.dyn, Inhomogeneous_NPTBerendsen)
    assert md.dyn.pressure == approx(6.324209e-07, rel=1e-5)
    assert md.dyn.compressibility == approx(1.5103069, rel=1e-5)
    assert os.path.isfile("md_out.traj")
    assert os.path.isfile("md_out.log")
    with open("md_out.log") as log_file:
        logs = log_file.read()
    assert logs == (
        "Time[ps]      Etot[eV]     Epot[eV]     Ekin[eV]    T[K]\n"
        "0.0000         -58.9727     -58.9727       0.0000     0.0\n"
    )
