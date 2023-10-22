from __future__ import annotations

import os
import pickle
from typing import TYPE_CHECKING, Literal

import numpy as np
import pytest
from ase import Atoms
from ase.md.npt import NPT
from ase.md.nptberendsen import Inhomogeneous_NPTBerendsen
from ase.md.nvtberendsen import NVTBerendsen
from ase.md.verlet import VelocityVerlet
from numpy.testing import assert_allclose
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.core import Structure
from pymatgen.io.ase import AseAtomsAdaptor
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
structure_2 = Structure.from_file(f"{ROOT}/examples/mp-1175469-Li9Co7O16.cif")
chgnet = CHGNet.load()


def test_eos():
    eos = EquationOfState()
    eos.fit(atoms=structure)

    assert eos.get_bulk_modulus() == approx(0.6501, rel=1e-4)
    assert eos.get_bulk_modulus(unit="GPa") == approx(104.16, rel=1e-4)
    assert eos.get_compressibility() == approx(1.5381, rel=1e-4)
    assert eos.get_compressibility(unit="GPa^-1") == approx(0.00960, rel=1e-4)


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
    )
    md.run(100)

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
        "0.0600         -58.7870     -58.8671       0.0801    77.5\n"
        "0.0800         -58.7024     -58.8023       0.0999    96.7\n"
        "0.1000         -58.6080     -58.6803       0.0723    69.9\n"
        "0.1200         -58.5487     -58.5849       0.0362    35.0\n"
        "0.1400         -58.4648     -58.5285       0.0637    61.6\n"
        "0.1600         -58.3202     -58.5693       0.2491   240.9\n"
        "0.1800         -58.2515     -58.7861       0.5346   517.0\n"
        "0.2000         -58.2441     -58.8199       0.5758   556.8\n",
        sep=" ",
    )
    assert_allclose(logs, ref, rtol=2.1e-3, atol=1e-8)


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
    print("nve logs", logs)
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
        trajectory="md_out.traj",
        logfile="md_out.log",
        loginterval=10,
    )
    md.run(100)

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
        "0.0600         -58.8819     -58.9315       0.0495    47.9\n"
        "0.0800         -58.7860     -58.8800       0.0940    90.9\n"
        "0.1000         -58.6916     -58.7694       0.0778    75.2\n"
        "0.1200         -58.5945     -58.6458       0.0513    49.6\n"
        "0.1400         -58.4972     -58.5543       0.0571    55.2\n"
        "0.1600         -58.4008     -58.5540       0.1532   148.1\n"
        "0.1800         -58.3292     -58.8330       0.5038   487.2\n"
        "0.2000         -58.2842     -58.8526       0.5684   549.7\n",
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

    md.run(100)

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
        "0.0200       -199.3047   -199.3890       0.0844    20.4\n"
        "0.0400       -199.3036   -199.3510       0.0475    11.5\n"
        "0.0600       -199.2999   -199.3219       0.0221     5.3\n"
        "0.0800       -199.2974   -199.4012       0.1038    25.1\n"
        "0.1000       -199.2927   -199.3097       0.0170     4.1\n"
        "0.1200       -199.2847   -199.3522       0.0675    16.3\n"
        "0.1400       -199.2802   -199.3789       0.0988    23.9\n"
        "0.1600       -199.2681   -199.2785       0.0104     2.5\n"
        "0.1800       -199.2565   -199.3830       0.1265    30.6\n"
        "0.2000       -199.2463   -199.3190       0.0727    17.6\n",
        sep=" ",
    )
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

    md.run(100)

    assert isinstance(md.atoms, Atoms)
    assert isinstance(md.atoms.calc, CHGNetCalculator)
    assert isinstance(md.dyn, NPT)
    assert md.bulk_modulus == approx(88.6389, rel=1e-2)
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
        "0.0200       -199.3048   -199.3891       0.0843    20.4\n"
        "0.0400       -199.3038   -199.3513       0.0475    11.5\n"
        "0.0600       -199.3005   -199.3226       0.0221     5.4\n"
        "0.0800       -199.2988   -199.4024       0.1036    25.0\n"
        "0.1000       -199.2945   -199.3115       0.0170     4.1\n"
        "0.1200       -199.2872   -199.3550       0.0679    16.4\n"
        "0.1400       -199.2841   -199.3822       0.0981    23.7\n"
        "0.1600       -199.2729   -199.2833       0.0105     2.5\n"
        "0.1800       -199.2622   -199.3895       0.1273    30.8\n"
        "0.2000       -199.2539   -199.3247       0.0708    17.1\n",
        sep=" ",
    )

    assert_allclose(logs, ref, rtol=1e-2, atol=1e-7)


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
    print(crystal_feas[0][0], crystal_feas[0][1])
    print(crystal_feas[10][0], crystal_feas[10][1])
    assert crystal_feas[0][0] == approx(-0.002082636, abs=1e-5)
    assert crystal_feas[0][1] == approx(-1.4285042, abs=1e-5)
    assert crystal_feas[10][0] == approx(-0.0020592688, abs=1e-5)
    assert crystal_feas[10][1] == approx(-1.4284436, abs=1e-5)
