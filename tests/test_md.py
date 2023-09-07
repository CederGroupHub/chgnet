from __future__ import annotations

import os
import pickle
from typing import TYPE_CHECKING, Literal

import numpy as np
import pytest
from ase import Atoms
from ase.md.nptberendsen import Inhomogeneous_NPTBerendsen
from ase.md.nvtberendsen import NVTBerendsen
from ase.md.verlet import VelocityVerlet
from pymatgen.core import Structure
from pytest import MonkeyPatch, approx

from chgnet import ROOT
from chgnet.graph import CrystalGraphConverter
from chgnet.model import StructOptimizer
from chgnet.model.dynamics import CHGNetCalculator, EquationOfState, MolecularDynamics
from chgnet.model.model import CHGNet

if TYPE_CHECKING:
    from pathlib import Path

relaxer = StructOptimizer()
structure = Structure.from_file(f"{ROOT}/examples/mp-18767-LiMnO2.cif")
chgnet = CHGNet.load()


def test_eos():
    eos = EquationOfState()
    eos.fit(atoms=structure)
    assert eos.get_bulk_modulus() == approx(0.66012829210838, rel=1e-4)
    assert eos.get_bulk_modulus(unit="GPa") == approx(105.76421250583728, rel=1e-4)
    assert eos.get_compressibility() == approx(1.51485, rel=1e-4)
    assert eos.get_compressibility(unit="GPa^-1") == approx(0.0094549940505, rel=1e-4)


@pytest.mark.parametrize("algorithm", ["legacy", "fast"])
def test_md_nvt(
    tmp_path: Path, monkeypatch: MonkeyPatch, algorithm: Literal["legacy", "fast"]
):
    monkeypatch.chdir(tmp_path)  # run MD in temporary directory

    chgnet_legacy = CHGNet.load()
    converter_legacy = CrystalGraphConverter(
        atom_graph_cutoff=5, bond_graph_cutoff=3, algorithm=algorithm
    )
    assert converter_legacy.algorithm == algorithm

    chgnet_legacy.graph_converter = converter_legacy

    md = MolecularDynamics(
        atoms=structure,
        model=chgnet_legacy,
        ensemble="nvt",
        thermostat="Berendsen_inhomogeneous",
        temperature=1000,  # in k
        timestep=2,  # in fs
        trajectory="md_out.traj",
        logfile="md_out.log",
        loginterval=10,
    )
    md.run(100)

    assert isinstance(md.atoms, Atoms)
    assert isinstance(md.atoms.calc, CHGNetCalculator)
    assert isinstance(md.dyn, NVTBerendsen)
    assert os.path.isfile("md_out.traj")
    assert os.path.isfile("md_out.log")
    with open("md_out.log") as log_file:
        next(log_file)
        logs = log_file.read()
        logs = np.fromstring(logs, dtype=float, sep=" ")
    ref = np.fromstring(
        "0.0000         -58.9727     -58.9727       0.0000     0.0\n"
        "0.0200         -58.9723     -58.9731       0.0009     0.8\n"
        "0.0400         -58.9672     -58.9727       0.0055     5.4\n"
        "0.0600         -58.9427     -58.9663       0.0235    22.8\n"
        "0.0800         -58.8605     -58.9352       0.0747    72.2\n"
        "0.1000         -58.7651     -58.8438       0.0786    76.0\n"
        "0.1200         -58.6684     -58.7268       0.0584    56.4\n"
        "0.1400         -58.5703     -58.6202       0.0499    48.2\n"
        "0.1600         -58.4724     -58.5531       0.0807    78.1\n"
        "0.1800         -58.3891     -58.8077       0.4186   404.8\n"
        "0.2000         -58.3398     -58.9244       0.5846   565.4\n",
        dtype=float,
        sep=" ",
    )
    assert np.isclose(logs, ref, rtol=2.1e-3, atol=1e-8).all()


def test_md_nve(tmp_path: Path, monkeypatch: MonkeyPatch):
    monkeypatch.chdir(tmp_path)  # run MD in temporary directory

    md = MolecularDynamics(
        atoms=structure,
        model=chgnet,
        ensemble="nve",
        timestep=1,  # in fs
        trajectory="md_out.traj",
        logfile="md_out.log",
        loginterval=10,
    )
    md.run(10)

    assert isinstance(md.atoms, Atoms)
    assert isinstance(md.atoms.calc, CHGNetCalculator)
    assert isinstance(md.dyn, VelocityVerlet)
    assert os.path.isfile("md_out.traj")
    assert os.path.isfile("md_out.log")
    with open("md_out.log") as log_file:
        logs = log_file.read()
    assert logs == (
        "Time[ps]      Etot[eV]     Epot[eV]     Ekin[eV]    T[K]\n"
        "0.0000         -58.9727     -58.9727       0.0000     0.0\n"
        "0.0100         -58.9727     -58.9728       0.0001     0.1\n"
    )


def test_md_npt_inhomogeneous_berendsen(tmp_path: Path, monkeypatch: MonkeyPatch):
    monkeypatch.chdir(tmp_path)  # run MD in temporary directory

    md = MolecularDynamics(
        atoms=structure,
        model=chgnet,
        ensemble="npt",
        thermostat="Berendsen_inhomogeneous",
        temperature=1000,  # in k
        timestep=2,  # in fs
        trajectory="md_out.traj",
        logfile="md_out.log",
        loginterval=10,
    )
    md.run(100)

    assert isinstance(md.atoms, Atoms)
    assert isinstance(md.atoms.calc, CHGNetCalculator)
    assert isinstance(md.dyn, Inhomogeneous_NPTBerendsen)
    assert md.bulk_modulus == approx(105.764, rel=1e-2)
    assert md.dyn.pressure == approx(6.324e-07, rel=1e-4)
    assert os.path.isfile("md_out.traj")
    assert os.path.isfile("md_out.log")
    with open("md_out.log") as log_file:
        next(log_file)
        logs = log_file.read()
        logs = np.fromstring(logs, dtype=float, sep=" ")
    ref = np.fromstring(
        "0.0000         -58.9727     -58.9727       0.0000     0.0\n"
        "0.0200         -58.9723     -58.9731       0.0009     0.8\n"
        "0.0400         -58.9672     -58.9727       0.0055     5.3\n"
        "0.0600         -58.9427     -58.9663       0.0235    22.7\n"
        "0.0800         -58.8605     -58.9352       0.0747    72.2\n"
        "0.1000         -58.7652     -58.8438       0.0786    76.0\n"
        "0.1200         -58.6686     -58.7269       0.0584    56.4\n"
        "0.1400         -58.5707     -58.6205       0.0499    48.2\n"
        "0.1600         -58.4731     -58.5533       0.0802    77.6\n"
        "0.1800         -58.3897     -58.8064       0.4167   402.9\n"
        "0.2000         -58.3404     -58.9253       0.5849   565.6\n",
        dtype=float,
        sep=" ",
    )
    assert np.isclose(logs, ref, rtol=2.1e-3, atol=1e-8).all()


def test_md_crystal_feas_log(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
):
    monkeypatch.chdir(tmp_path)  # run MD in temporary directory

    md = MolecularDynamics(
        atoms=structure,
        ensemble="nvt",
        temperature=1000,  # in k
        timestep=2,  # in fs
        trajectory="md_out.traj",
        logfile="md_out.log",
        crystal_feas_logfile="md_crystal_feas.p",
        loginterval=1,
    )
    md.run(100)

    assert os.path.isfile("md_crystal_feas.p")
    with open("md_crystal_feas.p", "rb") as file:
        data = pickle.load(file)

    crystal_feas = data["crystal_feas"]

    assert isinstance(crystal_feas, list)
    assert len(crystal_feas) == 101
    assert len(crystal_feas[0]) == 64
    assert crystal_feas[0][0] == approx(1.4411131, rel=1e-5)
    assert crystal_feas[0][1] == approx(2.652704, rel=1e-5)
    assert crystal_feas[10][0] == approx(1.4390125, rel=1e-5)
    assert crystal_feas[10][1] == approx(2.6525214, rel=1e-5)
