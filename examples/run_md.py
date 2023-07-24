import warnings

from pymatgen.core import Structure

from chgnet.model.dynamics import MolecularDynamics

warnings.filterwarnings("ignore", module="ase")


struct = Structure.from_file("./mp-18767-LiMnO2.cif")
ensemble = "npt"
temperature = 300

# NPT
md = MolecularDynamics(
    atoms=struct,
    ensemble=ensemble,
    temperature=temperature,  # in k
    timestep=2,  # in fs
    trajectory=f"md_out_{ensemble}_T_{temperature}.traj",
    logfile=f"md_out_{ensemble}_T_{temperature}.log",
    loginterval=100,
)
md.run(10 * 1000 * 500)  # 10 ns simulation
