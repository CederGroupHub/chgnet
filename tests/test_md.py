from __future__ import annotations

import os
import pickle
from typing import TYPE_CHECKING, Literal, get_args

import numpy as np
import pytest
from ase import Atoms
from ase.io.trajectory import Trajectory
from ase.md.npt import NPT
from ase.md.nptberendsen import Inhomogeneous_NPTBerendsen
from ase.md.nvtberendsen import NVTBerendsen
from ase.md.verlet import VelocityVerlet
from numpy.testing import assert_allclose
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.core import Structure
from pymatgen.io.ase import AseAtomsAdaptor
from pytest import MonkeyPatch, approx  # noqa: PT013

from chgnet import ROOT
from chgnet.graph import CrystalGraphConverter
from chgnet.model import StructOptimizer
from chgnet.model.dynamics import CHGNetCalculator, EquationOfState, MolecularDynamics
from chgnet.model.model import CHGNet, PredTask

if TYPE_CHECKING:
    from pathlib import Path

relaxer = StructOptimizer()
structure = Structure.from_file(f"{ROOT}/examples/mp-18767-LiMnO2.cif")
structure_2 = Structure.from_file(f"{ROOT}/examples/mp-1175469-Li9Co7O16.cif")
chgnet = CHGNet.load()


def test_version_and_params():
    calculator = relaxer.calculator
    assert isinstance(calculator, CHGNetCalculator)
    model = calculator.model
    assert isinstance(model, CHGNet)
    assert relaxer.version == calculator.version == model.version
    assert relaxer.n_params == calculator.n_params == model.n_params


def test_eos():
    eos = EquationOfState()
    eos.fit(atoms=structure)

    assert eos.get_bulk_modulus() == approx(0.6481, rel=1e-4)
    assert eos.get_bulk_modulus(unit="GPa") == approx(103.845, rel=1e-4)
    assert eos.get_compressibility() == approx(1.5428, rel=1e-4)
    assert eos.get_compressibility(unit="GPa^-1") == approx(0.0096296, rel=1e-4)


@pytest.mark.parametrize("algorithm", ["legacy", "fast"])
def test_md_nvt_berendsen(
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
        return_site_energies=True,
    )
    md.run(30)

    assert isinstance(md.atoms, Atoms)
    assert isinstance(md.atoms.calc, CHGNetCalculator)
    assert isinstance(md.dyn, NVTBerendsen)
    assert set(os.listdir()) == {"md_out.log", "md_out.traj"}
    with open("md_out.log") as log_file:
        next(log_file)
        logs = log_file.read()
        logs = np.fromstring(logs, sep=" ")
    ref = np.fromstring(
        "0.0000         -58.8678     -58.8678       0.0000     0.0\n"
        "0.0200         -58.8665     -58.8692       0.0027     2.6\n"
        "0.0400         -58.8650     -58.8846       0.0196    18.9\n"
        "0.0600         -58.7870     -58.8671       0.0801    77.5\n",
        sep=" ",
    )
    assert_allclose(logs, ref, rtol=2.1e-3, atol=1e-8)

    traj = Trajectory("md_out.traj")
    assert isinstance(traj[0].get_potential_energy(), float)
    assert isinstance(traj[0].get_potential_energies(), np.ndarray)


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
    assert set(os.listdir()) == {"md_out.log", "md_out.traj"}
    with open("md_out.log") as log_file:
        logs = log_file.read()
    assert logs == (
        "Time[ps]      Etot[eV]     Epot[eV]     Ekin[eV]    T[K]\n"
        "0.0000         -58.9415     -58.9415       0.0000     0.0\n"
        "0.0100         -58.9415     -58.9417       0.0002     0.2\n"
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
        bulk_modulus=103.845,
        trajectory="md_out.traj",
        logfile="md_out.log",
        loginterval=10,
    )
    md.run(30)

    assert isinstance(md.atoms, Atoms)
    assert isinstance(md.atoms.calc, CHGNetCalculator)
    assert isinstance(md.dyn, Inhomogeneous_NPTBerendsen)
    assert md.bulk_modulus == approx(104.16, rel=1e-2)
    assert md.dyn.pressure == approx(6.324e-07, rel=1e-4)
    assert set(os.listdir()) == {"md_out.log", "md_out.traj"}
    with open("md_out.log") as log_file:
        next(log_file)
        logs = log_file.read()
        logs = np.fromstring(logs, sep=" ")
    ref = np.fromstring(
        "0.0000         -58.9415     -58.9415       0.0000     0.0\n"
        "0.0200         -58.9407     -58.9423       0.0016     1.6\n"
        "0.0400         -58.9310     -58.9415       0.0105    10.1\n"
        "0.0600         -58.8819     -58.9351       0.0495    47.9\n",
        sep=" ",
    )
    assert_allclose(logs, ref, rtol=2.1e-3, atol=1e-8)


def test_md_nvt_nose_hoover(tmp_path: Path, monkeypatch: MonkeyPatch):
    monkeypatch.chdir(tmp_path)  # run MD in temporary directory

    md = MolecularDynamics(
        atoms=structure_2,
        model=chgnet,
        ensemble="nvt",
        thermostat="nose-hoover",
        temperature=1000,  # in k
        timestep=2,  # in fs
        trajectory="md_out.traj",
        logfile="md_out.log",
        loginterval=10,
    )

    new_atoms = AseAtomsAdaptor.get_structure(md.atoms)
    assert StructureMatcher(
        primitive_cell=False, scale=False, attempt_supercell=False, allow_subset=False
    ).fit(structure_2, new_atoms)

    md.run(30)

    assert isinstance(md.atoms, Atoms)
    assert isinstance(md.atoms.calc, CHGNetCalculator)
    assert isinstance(md.dyn, NPT)
    assert_allclose(
        md.dyn.externalstress,
        [-6.324e-07, -6.324e-07, -6.324e-07, 0.0, 0.0, 0.0],
        rtol=1e-4,
        atol=1e-8,
    )
    assert set(os.listdir()) == {"md_out.log", "md_out.traj"}
    with open("md_out.log") as log_file:
        next(log_file)
        logs = log_file.read()
        logs = np.fromstring(logs, sep=" ")
    ref = np.fromstring(
        "0.0000       -199.3046   -199.3046       0.0000    0.0\n"
        "0.0200       -199.3047   -199.3890       0.0844    20.4\n"
        "0.0400       -199.3036   -199.3510       0.0475    11.5\n"
        "0.0600       -199.2999   -199.3219       0.0221     5.3\n",
        sep=" ",
    )
    if len(logs) == 15:  # pytest issue with python 3.12
        assert_allclose(logs, ref[5:], rtol=1e-2, atol=1e-7)
    else:
        assert_allclose(logs, ref, rtol=1e-2, atol=1e-7)


def test_md_npt_nose_hoover(tmp_path: Path, monkeypatch: MonkeyPatch):
    # https://github.com/CederGroupHub/chgnet/pull/68
    monkeypatch.chdir(tmp_path)  # run MD in temporary directory

    md = MolecularDynamics(
        atoms=structure_2,
        model=chgnet,
        ensemble="npt",
        thermostat="nose-hoover",
        temperature=1000,  # in k
        timestep=2,  # in fs
        trajectory="md_out.traj",
        logfile="md_out.log",
        loginterval=10,
    )

    new_atoms = AseAtomsAdaptor.get_structure(md.atoms)
    assert StructureMatcher(
        primitive_cell=False, scale=False, attempt_supercell=False, allow_subset=False
    ).fit(structure_2, new_atoms)

    md.run(30)

    assert isinstance(md.atoms, Atoms)
    assert isinstance(md.atoms.calc, CHGNetCalculator)
    assert isinstance(md.dyn, NPT)
    assert md.bulk_modulus == approx(91.14753, rel=1e-2)
    assert_allclose(
        md.dyn.externalstress,
        [-6.324e-07, -6.324e-07, -6.324e-07, 0.0, 0.0, 0.0],
        rtol=1e-4,
        atol=1e-8,
    )
    assert set(os.listdir()) == {"md_out.log", "md_out.traj"}
    with open("md_out.log") as log_file:
        next(log_file)
        logs = log_file.read()
        logs = np.fromstring(logs, sep=" ")
    ref = np.fromstring(
        "0.0000       -199.3046   -199.3046       0.0000    0.0\n"
        "0.0200       -199.3048   -199.3891       0.0843    20.4\n"
        "0.0400       -199.3038   -199.3513       0.0475    11.5\n"
        "0.0600       -199.3005   -199.3226       0.0221     5.4\n",
        sep=" ",
    )
    if len(logs) == 15:  # pytest issue with python 3.12
        assert_allclose(logs, ref[5:], rtol=1e-2, atol=1e-7)
    else:
        assert_allclose(logs, ref, rtol=1e-2, atol=1e-7)


def test_md_crystal_feas_log(tmp_path: Path, monkeypatch: MonkeyPatch):
    monkeypatch.chdir(tmp_path)  # run MD in temporary directory

    md = MolecularDynamics(
        atoms=structure,
        ensemble="nvt",
        temperature=1000,  # in k
        timestep=2,  # in fs
        trajectory="md_out.traj",
        logfile="md_out.log",
        crystal_feas_logfile=(logfile := "md_crystal_feas.pkl"),
        loginterval=1,
    )
    md.run(30)

    assert os.path.isfile(logfile)
    with open(logfile, "rb") as file:
        data = pickle.load(file)

    crystal_feas = data["crystal_feas"]

    assert isinstance(crystal_feas, list)
    assert len(crystal_feas) == 31
    assert len(crystal_feas[0]) == 64
    assert crystal_feas[0][0] == approx(-0.002082636, abs=1e-5)
    assert crystal_feas[0][1] == approx(-1.4285042, abs=1e-5)
    assert crystal_feas[10][0] == approx(-0.0020592688, abs=1e-5)
    assert crystal_feas[10][1] == approx(-1.4284436, abs=1e-5)


@pytest.mark.parametrize("task", [*get_args(PredTask)])
def test_calculator_task_valid(task: PredTask):
    """Test that the task kwarg of CHGNetCalculator.calculate() works correctly."""
    key_map = dict(e="energy", f="forces", m="magmoms", s="stress")
    calculator = CHGNetCalculator()
    atoms = AseAtomsAdaptor.get_atoms(structure)
    atoms.calc = calculator

    calculator.calculate(atoms=atoms, task=task)

    for key, prop in key_map.items():
        assert (prop in calculator.results) == (key in task)


def test_calculator_task_invalid():
    """Test that invalid task raises ValueError."""
    calculator = CHGNetCalculator()
    atoms = AseAtomsAdaptor.get_atoms(structure)
    atoms.calc = calculator

    with pytest.raises(ValueError, match="Invalid task='invalid'."):
        calculator.calculate(atoms=atoms, task="invalid")
