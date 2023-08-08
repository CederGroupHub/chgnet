from __future__ import annotations

import contextlib
import io
import pickle
import sys
from typing import TYPE_CHECKING

import numpy as np
import torch
from ase import Atoms, units
from ase.calculators.calculator import Calculator, all_changes, all_properties
from ase.constraints import ExpCellFilter
from ase.md.nptberendsen import Inhomogeneous_NPTBerendsen, NPTBerendsen
from ase.md.nvtberendsen import NVTBerendsen
from ase.optimize.bfgs import BFGS
from ase.optimize.bfgslinesearch import BFGSLineSearch
from ase.optimize.fire import FIRE
from ase.optimize.lbfgs import LBFGS, LBFGSLineSearch
from ase.optimize.mdmin import MDMin
from ase.optimize.sciopt import SciPyFminBFGS, SciPyFminCG
from pymatgen.analysis.eos import BirchMurnaghan
from pymatgen.core.structure import Molecule, Structure
from pymatgen.io.ase import AseAtomsAdaptor

from chgnet.model.model import CHGNet
from chgnet.utils import cuda_devices_sorted_by_free_mem

if TYPE_CHECKING:
    from ase.io import Trajectory
    from ase.optimize.optimize import Optimizer

# We would like to thank M3GNet develop team for this module
# source: https://github.com/materialsvirtuallab/m3gnet

OPTIMIZERS = {
    "FIRE": FIRE,
    "BFGS": BFGS,
    "LBFGS": LBFGS,
    "LBFGSLineSearch": LBFGSLineSearch,
    "MDMin": MDMin,
    "SciPyFminCG": SciPyFminCG,
    "SciPyFminBFGS": SciPyFminBFGS,
    "BFGSLineSearch": BFGSLineSearch,
}


class CHGNetCalculator(Calculator):
    """CHGNet Calculator for ASE applications."""

    implemented_properties = ("energy", "forces", "stress", "magmoms")

    def __init__(
        self,
        model: CHGNet | None = None,
        use_device: str | None = None,
        stress_weight: float | None = 1 / 160.21766208,
        **kwargs,
    ) -> None:
        """Provide a CHGNet instance to calculate various atomic properties using ASE.

        Args:
            model (CHGNet): instance of a chgnet model. If set to None,
                the pretrained CHGNet is loaded.
                Default = None
            use_device (str, optional): The device to be used for predictions,
                either "cpu", "cuda", or "mps". If not specified, the default device is
                automatically selected based on the available options.
                Default = None
            stress_weight (float): the conversion factor to convert GPa to eV/A^3.
                Default = 1/160.21
            **kwargs: Passed to the Calculator parent class.
        """
        super().__init__(**kwargs)

        # mps is disabled before stable version of pytorch on apple mps is released
        if use_device == "mps":
            raise NotImplementedError("'mps' backend is not supported yet")
        # elif torch.backends.mps.is_available():
        #     self.device = 'mps'

        # Determine the device to use
        self.device = use_device or ("cuda" if torch.cuda.is_available() else "cpu")
        if self.device == "cuda":
            self.device = f"cuda:{cuda_devices_sorted_by_free_mem()[-1]}"

        # Move the model to the specified device
        self.model = (model or CHGNet.load()).to(self.device)
        self.stress_weight = stress_weight
        print(f"CHGNet will run on {self.device}")

    def calculate(
        self,
        atoms: Atoms | None = None,
        properties: list | None = None,
        system_changes: list | None = None,
    ) -> None:
        """Calculate various properties of the atoms using CHGNet.

        Args:
            atoms (Atoms | None): The atoms object to calculate properties for.
            properties (list | None): The properties to calculate.
                Default is all properties.
            system_changes (list | None): The changes made to the system.
                Default is all changes.
        """
        properties = properties or all_properties
        system_changes = system_changes or all_changes
        super().calculate(
            atoms=atoms,
            properties=properties,
            system_changes=system_changes,
        )

        # Run CHGNet
        structure = AseAtomsAdaptor.get_structure(atoms)
        graph = self.model.graph_converter(structure)
        model_prediction = self.model.predict_graph(graph.to(self.device), task="efsm")

        # Convert Result
        factor = 1 if not self.model.is_intensive else structure.composition.num_atoms
        self.results.update(
            energy=model_prediction["e"] * factor,
            forces=model_prediction["f"],
            free_energy=model_prediction["e"] * factor,
            magmoms=model_prediction["m"],
            stress=model_prediction["s"] * self.stress_weight,
        )


class StructOptimizer:
    """Wrapper class for structural relaxation."""

    def __init__(
        self,
        model: CHGNet | CHGNetCalculator | None = None,
        optimizer_class: Optimizer | str | None = "FIRE",
        use_device: str | None = None,
        stress_weight: float = 1 / 160.21766208,
    ) -> None:
        """Provide a trained CHGNet model and an optimizer to relax crystal structures.

        Args:
            model (CHGNet): instance of a CHGNet model or CHGNetCalculator.
                If set to None, the pretrained CHGNet is loaded.
                Default = None
            optimizer_class (Optimizer,str): choose optimizer from ASE.
                Default = "FIRE"
            use_device (str, optional): The device to be used for predictions,
                either "cpu", "cuda", or "mps". If not specified, the default device is
                automatically selected based on the available options.
                Default = None
            stress_weight (float): the conversion factor to convert GPa to eV/A^3.
                Default = 1/160.21
        """
        if isinstance(optimizer_class, str):
            if optimizer_class in OPTIMIZERS:
                optimizer_class = OPTIMIZERS[optimizer_class]
            else:
                raise ValueError(
                    f"Optimizer instance not found. Select from {list(OPTIMIZERS)}"
                )

        self.optimizer_class: Optimizer = optimizer_class

        if isinstance(model, CHGNetCalculator):
            self.calculator = model
        else:
            self.calculator = CHGNetCalculator(
                model=model, stress_weight=stress_weight, use_device=use_device
            )

    def relax(
        self,
        atoms: Structure | Atoms,
        fmax: float | None = 0.1,
        steps: int | None = 500,
        relax_cell: bool | None = True,
        save_path: str | None = None,
        trajectory_save_interval: int | None = 1,
        verbose: bool = True,
        **kwargs,
    ) -> dict[str, Structure | TrajectoryObserver]:
        """Relax the Structure/Atoms until maximum force is smaller than fmax.

        Args:
            atoms (Structure | Atoms): A Structure or Atoms object to relax.
            fmax (float | None): The maximum force tolerance for relaxation.
                Default = 0.1
            steps (int | None): The maximum number of steps for relaxation.
                Default = 500
            relax_cell (bool | None): Whether to relax the cell as well.
                Default = True
            save_path (str | None): The path to save the trajectory.
                Default = None
            trajectory_save_interval (int | None): Trajectory save interval.
                Default = 1
            verbose (bool): Whether to print the output of the ASE optimizer.
                Default = True
            **kwargs: Additional parameters for the optimizer.

        Returns:
            dict[str, Structure | TrajectoryObserver]:
                A dictionary with 'final_structure' and 'trajectory'.
        """
        if isinstance(atoms, Structure):
            atoms = AseAtomsAdaptor.get_atoms(atoms)

        atoms.calc = self.calculator  # assign model used to predict forces

        stream = sys.stdout if verbose else io.StringIO()
        with contextlib.redirect_stdout(stream):
            obs = TrajectoryObserver(atoms)
            if relax_cell:
                atoms = ExpCellFilter(atoms)
            optimizer = self.optimizer_class(atoms, **kwargs)
            optimizer.attach(obs, interval=trajectory_save_interval)
            optimizer.run(fmax=fmax, steps=steps)
            obs()

        if save_path is not None:
            obs.save(save_path)

        if isinstance(atoms, ExpCellFilter):
            atoms = atoms.atoms
        struct = AseAtomsAdaptor.get_structure(atoms)
        for key in struct.site_properties:
            struct.remove_site_property(property_name=key)
        struct.add_site_property(
            "magmom", [float(magmom) for magmom in atoms.get_magnetic_moments()]
        )
        return {"final_structure": struct, "trajectory": obs}


class TrajectoryObserver:
    """Trajectory observer is a hook in the relaxation process that saves the
    intermediate structures.
    """

    def __init__(self, atoms: Atoms) -> None:
        """Create a TrajectoryObserver from an Atoms object.

        Args:
            atoms (Atoms): the structure to observe.
        """
        self.atoms = atoms
        self.energies: list[float] = []
        self.forces: list[np.ndarray] = []
        self.stresses: list[np.ndarray] = []
        self.magmoms: list[np.ndarray] = []
        self.atom_positions: list[np.ndarray] = []
        self.cells: list[np.ndarray] = []

    def __call__(self):
        """The logic for saving the properties of an Atoms during the relaxation."""
        self.energies.append(self.compute_energy())
        self.forces.append(self.atoms.get_forces())
        self.stresses.append(self.atoms.get_stress())
        self.magmoms.append(self.atoms.get_magnetic_moments())
        self.atom_positions.append(self.atoms.get_positions())
        self.cells.append(self.atoms.get_cell()[:])

    def __len__(self) -> int:
        """The number of steps in the trajectory."""
        return len(self.energies)

    def compute_energy(self) -> float:
        """Calculate the potential energy.

        Returns:
            energy (float): the potential energy.
        """
        return self.atoms.get_potential_energy()

    def save(self, filename: str) -> None:
        """Save the trajectory to file.

        Args:
            filename (str): filename to save the trajectory
        """
        out_pkl = {
            "energy": self.energies,
            "forces": self.forces,
            "stresses": self.stresses,
            "magmoms": self.magmoms,
            "atom_positions": self.atom_positions,
            "cell": self.cells,
            "atomic_number": self.atoms.get_atomic_numbers(),
        }
        with open(filename, "wb") as file:
            pickle.dump(out_pkl, file)


class MolecularDynamics:
    """Molecular dynamics class."""

    def __init__(
        self,
        atoms: Atoms | Structure,
        model: CHGNet | CHGNetCalculator | None = None,
        ensemble: str = "nvt",
        temperature: int = 300,
        timestep: float = 2.0,
        pressure: float = 1.01325 * units.bar,
        taut: float | None = None,
        taup: float | None = None,
        compressibility_au: float | None = None,
        trajectory: str | Trajectory | None = None,
        logfile: str | None = None,
        loginterval: int = 1,
        append_trajectory: bool = False,
        use_device: str | None = None,
    ) -> None:
        """Initialize the MD class.

        Args:
            atoms (Atoms): atoms to run the MD
            model (CHGNet): instance of a CHGNet model or CHGNetCalculator.
                If set to None, the pretrained CHGNet is loaded.
                Default = None
            ensemble (str): choose from 'nvt' or 'npt'
                Default = "nvt"
            temperature (float): temperature for MD simulation, in K
                Default = 300
            timestep (float): time step in fs
                Default = 2
            pressure (float): pressure in eV/A^3
                Default = 1.01325 * units.bar
            taut (float): time constant for Berendsen temperature coupling in fs
                Default = 100 * timestep
            taup (float): time constant for pressure coupling in fs
                Default = 1000 * timestep
            compressibility_au (float): compressibility of the material in A^3/eV
                for npt ensemble, if not provided, it will be calculated by CHGNet
                through Birch Murnaghan equation of state
                Default = None
            trajectory (str or Trajectory): Attach trajectory object
                Default = None
            logfile (str): open this file for recording MD outputs
                Default = None
            loginterval (int): write to log file every interval steps
                Default = 1
            append_trajectory (bool): Whether to append to prev trajectory.
                If false, previous trajectory gets overwritten
                Default = False
            use_device (str): the device for the MD run
                Default = None
        """
        if isinstance(atoms, (Structure, Molecule)):
            atoms = AseAtomsAdaptor.get_atoms(atoms)

        self.atoms = atoms
        if isinstance(model, CHGNetCalculator):
            self.atoms.calc = model
        else:
            self.atoms.calc = CHGNetCalculator(model=model, use_device=use_device)

        if taut is None:
            taut = 100 * timestep * units.fs
        if taup is None:
            taup = 1000 * timestep * units.fs

        if ensemble.lower() == "nvt":
            """
            Berendsen (constant N, V, T) molecular dynamics.
            """
            self.dyn = NVTBerendsen(
                atoms=self.atoms,
                timestep=timestep * units.fs,
                temperature_K=temperature,
                taut=taut * units.fs,
                trajectory=trajectory,
                logfile=logfile,
                loginterval=loginterval,
                append_trajectory=append_trajectory,
            )
        else:
            if compressibility_au is None:
                eos = EquationOfState(model=self.atoms.calc)
                eos.fit(atoms=atoms, steps=500, fmax=0.1)
                compressibility_au = eos.get_compressibility(unit="A^3/eV")
                print(
                    f"Done compressibility calculation: "
                    f"b = {round(compressibility_au, 3)} A^3/eV"
                )

            if ensemble.lower() == "npt":
                """
                NPT ensemble default to Inhomogeneous_NPTBerendsen thermo/barostat
                This is a more flexible scheme that fixes three angles of the unit
                cell but allows three lattice parameter to change independently.
                """

                self.dyn = Inhomogeneous_NPTBerendsen(
                    atoms=self.atoms,
                    timestep=timestep * units.fs,
                    temperature_K=temperature,
                    pressure_au=pressure,
                    taut=taut * units.fs,
                    taup=taup * units.fs,
                    compressibility_au=compressibility_au,
                    trajectory=trajectory,
                    logfile=logfile,
                    loginterval=loginterval,
                )

            elif ensemble.lower() == "npt_berendsen":
                """
                This is a similar scheme to the Inhomogeneous_NPTBerendsen.
                This is a less flexible scheme that fixes the shape of the
                cell - three angles are fixed and the ratios between the three
                lattice constants.
                """

                self.dyn = NPTBerendsen(
                    atoms=self.atoms,
                    timestep=timestep * units.fs,
                    temperature_K=temperature,
                    pressure_au=pressure,
                    taut=taut * units.fs,
                    taup=taup * units.fs,
                    compressibility_au=compressibility_au,
                    trajectory=trajectory,
                    logfile=logfile,
                    loginterval=loginterval,
                    append_trajectory=append_trajectory,
                )

            else:
                raise ValueError("Ensemble not supported")

        self.trajectory = trajectory
        self.logfile = logfile
        self.loginterval = loginterval
        self.timestep = timestep

    def run(self, steps: int):
        """Thin wrapper of ase MD run.

        Args:
            steps (int): number of MD steps
        Returns:
        """
        self.dyn.run(steps)

    def set_atoms(self, atoms: Atoms):
        """Set new atoms to run MD.

        Args:
            atoms (Atoms): new atoms for running MD
        Returns:
        """
        calculator = self.atoms.calc
        self.atoms = atoms
        self.dyn.atoms = atoms
        self.dyn.atoms.calc = calculator


class EquationOfState:
    """Class to calculate equation of state."""

    def __init__(
        self,
        model: CHGNet | CHGNetCalculator | None = None,
        optimizer_class: Optimizer | str | None = "FIRE",
        use_device: str | None = None,
        stress_weight: float = 1 / 160.21766208,
    ) -> None:
        """Initialize a structure optimizer object for calculation of bulk modulus.

        Args:
            model (CHGNet): instance of a CHGNet model or CHGNetCalculator.
                If set to None, the pretrained CHGNet is loaded.
                Default = None
            optimizer_class (Optimizer,str): choose optimizer from ASE.
                Default = "FIRE"
            use_device (str, optional): The device to be used for predictions,
                either "cpu", "cuda", or "mps". If not specified, the default device is
                automatically selected based on the available options.
                Default = None
            stress_weight (float): the conversion factor to convert GPa to eV/A^3.
                Default = 1/160.21
        """
        self.relaxer = StructOptimizer(
            model=model,
            optimizer_class=optimizer_class,
            use_device=use_device,
            stress_weight=stress_weight,
        )
        self.fitted = False

    def fit(
        self,
        atoms: Structure | Atoms,
        n_points: int = 11,
        fmax: float | None = 0.1,
        steps: int | None = 500,
        **kwargs,
    ):
        """Relax the Structure/Atoms and fit the Birch-Murnaghan equation of state.

        Args:
            atoms (Structure | Atoms): A Structure or Atoms object to relax.
            n_points (int): Number of structures used in fitting the equation of states
            fmax (float | None): The maximum force tolerance for relaxation.
                Default = 0.1
            steps (int | None): The maximum number of steps for relaxation.
                Default = 500
            **kwargs: Additional parameters for the optimizer.

        Returns:
            Bulk Modulus (float)
        """
        if isinstance(atoms, Atoms):
            atoms = AseAtomsAdaptor.get_structure(atoms)
        volumes, energies = [], []
        for idx in np.linspace(-0.1, 0.1, n_points):
            structure_strained = atoms.copy()
            structure_strained.apply_strain([idx, idx, idx])
            result = self.relaxer.relax(
                structure_strained,
                relax_cell=False,
                fmax=fmax,
                steps=steps,
                verbose=False,
                **kwargs,
            )
            volumes.append(result["final_structure"].volume)
            energies.append(result["trajectory"].energies[-1])
        self.bm = BirchMurnaghan(volumes=volumes, energies=energies)
        self.bm.fit()
        self.fitted = True

    def get_bulk_modulus(self, unit: str = "eV/A^3"):
        """Get the bulk modulus of from the fitted Birch-Murnaghan equation of state.

        Args:
            unit (str): The unit of bulk modulus. Can be "eV/A^3" or "GPa"
                Default = "eV/A^3"

        Returns:
            Bulk Modulus (float)
        """
        if self.fitted is False:
            raise ValueError(
                "Equation of state needs to be fitted first through self.fit()"
            )
        if unit == "eV/A^3":
            return self.bm.b0
        if unit == "GPa":
            return self.bm.b0_GPa
        raise NotImplementedError("unit has to be eV/A^3 or GPa")

    def get_compressibility(self, unit: str = "A^3/eV"):
        """Get the bulk modulus of from the fitted Birch-Murnaghan equation of state.

        Args:
            unit (str): The unit of bulk modulus. Can be "A^3/eV",
            "GPa^-1" "Pa^-1" or "m^2/N"
                Default = "A^3/eV"

        Returns:
            Bulk Modulus (float)
        """
        if self.fitted is False:
            raise ValueError(
                "Equation of state needs to be fitted first through self.fit()"
            )
        if unit == "A^3/eV":
            return 1 / self.bm.b0
        if unit == "GPa^-1":
            return 1 / self.bm.b0_GPa
        if unit in ["Pa^-1", "m^2/N"]:
            return 1 / (self.bm.b0_GPa * 1e9)
        raise NotImplementedError("unit has to be one of A^3/eV, GPa^-1 Pa^-1 or m^2/N")
