<h1 align="center">CHGNet</h1>

<h4 align="center">

[![Tests](https://github.com/CederGroupHub/chgnet/actions/workflows/test.yml/badge.svg)](https://github.com/CederGroupHub/chgnet/actions/workflows/test.yml)
[![Codacy Badge](https://app.codacy.com/project/badge/Coverage/e3bdcea0382a495d96408e4f84408e85)](https://app.codacy.com/gh/CederGroupHub/chgnet/dashboard?utm_source=gh&utm_medium=referral&utm_content=&utm_campaign=Badge_coverage)
[![arXiv](https://img.shields.io/badge/arXiv-2302.14231-blue)](https://arxiv.org/abs/2302.14231)
![GitHub repo size](https://img.shields.io/github/repo-size/CederGroupHub/chgnet)
[![PyPI](https://img.shields.io/pypi/v/chgnet?logo=pypi&logoColor=white)](https://pypi.org/project/chgnet?logo=pypi&logoColor=white)
[![Docs](https://img.shields.io/badge/API-Docs-blue)](https://chgnet.lbl.gov)
[![Requires Python 3.9+](https://img.shields.io/badge/Python-3.9+-blue.svg?logo=python&logoColor=white)](https://python.org/downloads)

</h4>

A pretrained universal neural network potential for
**charge**-informed atomistic modeling ([see publication](https://nature.com/articles/s42256-023-00716-3))
![Logo](https://raw.github.com/CederGroupHub/chgnet/main/site/static/chgnet-logo.png)
**C**rystal **H**amiltonian **G**raph neural **Net**work is pretrained on the GGA/GGA+U static and relaxation trajectories from Materials Project,
a comprehensive dataset consisting of more than 1.5 Million structures from 146k compounds spanning the whole periodic table.

CHGNet highlights its ability to study electron interactions and charge distribution
in atomistic modeling with near DFT accuracy. The charge inference is realized by regularizing the atom features with
DFT magnetic moments, which carry rich information about both local ionic environments and charge distribution.

Pretrained CHGNet achieves SOTA performance on materials stability prediction from unrelaxed structures according to [Matbench Discovery](https://matbench-discovery.materialsproject.org) [[repo](https://github.com/janosh/matbench-discovery)].

## Example notebooks

| Notebooks                                                                                                                | Links&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;                                                                                                     | Descriptions                                                                                                                        |
| ------------------------------------------------------------------------------------------------------------------------ | --------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------- |
| [**CHGNet Basics**](https://github.com/CederGroupHub/chgnet/blob/main/examples/basics.ipynb)                             | [![Open in Google Colab]](https://colab.research.google.com/github/CederGroupHub/chgnet/blob/main/examples/basics.ipynb)                      | Examples for loading pre-trained CHGNet, predicting energy, force, stress, magmom as well as running structure optimization and MD. |
| [**Tuning CHGNet**](https://github.com/CederGroupHub/chgnet/blob/main/examples/fine_tuning.ipynb)                        | [![Open in Google Colab]](https://colab.research.google.com/github/CederGroupHub/chgnet/blob/main/examples/fine_tuning.ipynb)                 | Examples of fine tuning the pretrained CHGNet to your system of interest.                                                           |
| [**Visualize Relaxation**](https://github.com/CederGroupHub/chgnet/blob/main/examples/crystaltoolkit_relax_viewer.ipynb) | [![Open in Google Colab]](https://colab.research.google.com/github/CederGroupHub/chgnet/blob/main/examples/crystaltoolkit_relax_viewer.ipynb) | Crystal Toolkit that visualizes atom positions, energies and forces of a structure during CHGNet relaxation.                        |

[Open in Google Colab]: https://colab.research.google.com/assets/colab-badge.svg

## Installation

```sh
pip install chgnet
```

if PyPI installation fails or you need the latest `main` branch commits, you can install from source:

```sh
pip install git+https://github.com/CederGroupHub/chgnet
```

## Docs

View [API docs](https://cedergrouphub.github.io/chgnet/api).

## Usage

### Direct Inference (Static Calculation)

Pretrained `CHGNet` can predict the energy (eV/atom), force (eV/A), stress (GPa) and
magmom ($\mu_B$) of a given structure.

```python
from chgnet.model.model import CHGNet
from pymatgen.core import Structure

chgnet = CHGNet.load()
structure = Structure.from_file('examples/mp-18767-LiMnO2.cif')
prediction = chgnet.predict_structure(structure)

for key, unit in [
    ("energy", "eV/atom"),
    ("forces", "eV/A"),
    ("stress", "GPa"),
    ("magmom", "mu_B"),
]:
    print(f"CHGNet-predicted {key} ({unit}):\n{prediction[key[0]]}\n")
```

### Molecular Dynamics

Charge-informed molecular dynamics can be simulated with pretrained `CHGNet` through `ASE` python interface (see below),
or through [LAMMPS](https://github.com/advancesoftcorp/lammps/tree/based-on-lammps_2Jun2022/src/ML-CHGNET).

```python
from chgnet.model.model import CHGNet
from chgnet.model.dynamics import MolecularDynamics
from pymatgen.core import Structure
import warnings
warnings.filterwarnings("ignore", module="pymatgen")
warnings.filterwarnings("ignore", module="ase")

structure = Structure.from_file("examples/mp-18767-LiMnO2.cif")
chgnet = CHGNet.load()

md = MolecularDynamics(
    atoms=structure,
    model=chgnet,
    ensemble="nvt",
    temperature=1000,  # in K
    timestep=2,  # in femto-seconds
    trajectory="md_out.traj",
    logfile="md_out.log",
    loginterval=100,
)
md.run(50)  # run a 0.1 ps MD simulation
```

The MD defaults to CUDA if available, to manually set device to cpu or mps:
`MolecularDynamics(use_device='cpu')`.

MD outputs are saved to the ASE trajectory file, to visualize the MD trajectory
and magnetic moments after the MD run:

```python
from ase.io.trajectory import Trajectory
from pymatgen.io.ase import AseAtomsAdaptor
from chgnet.utils import solve_charge_by_mag

traj = Trajectory("md_out.traj")
mag = traj[-1].get_magnetic_moments()

# get the non-charge-decorated structure
structure = AseAtomsAdaptor.get_structure(traj[-1])
print(structure)

# get the charge-decorated structure
struct_with_chg = solve_charge_by_mag(structure)
print(struct_with_chg)
```

To manipulate the MD trajectory, convert to other data formats, calculate mean square displacement, etc,
please refer to [ASE trajectory documentation](https://wiki.fysik.dtu.dk/ase/ase/io/trajectory.html).

### Structure Optimization

`CHGNet` can perform fast structure optimization and provide site-wise magnetic moments. This makes it ideal for pre-relaxation and
`MAGMOM` initialization in spin-polarized DFT.

```python
from chgnet.model import StructOptimizer

relaxer = StructOptimizer()
result = relaxer.relax(structure)
print("CHGNet relaxed structure", result["final_structure"])
print("relaxed total energy in eV:", result['trajectory'].energies[-1])
```

### Available Weights

CHGNet 0.3.0 is released with new pretrained weights! (release date: 10/22/23)

`CHGNet.load()` now loads `0.3.0` by default,
previous `0.2.0` version can be loaded with `CHGNet.load('0.2.0')`

- [CHGNet_0.3.0](https://github.com/CederGroupHub/chgnet/blob/main/chgnet/pretrained/0.3.0/README.md)
- [CHGNet_0.2.0](https://github.com/CederGroupHub/chgnet/blob/main/chgnet/pretrained/0.2.0/README.md)

### Model Training / Fine-tune

Fine-tuning will help achieve better accuracy if a high-precision study is desired. To train/tune a `CHGNet`, you need to define your data in a
pytorch `Dataset` object. The example datasets are provided in `data/dataset.py`

```python
from chgnet.data.dataset import StructureData, get_train_val_test_loader
from chgnet.trainer import Trainer

dataset = StructureData(
    structures=list_of_structures,
    energies=list_of_energies,
    forces=list_of_forces,
    stresses=list_of_stresses,
    magmoms=list_of_magmoms,
)
train_loader, val_loader, test_loader = get_train_val_test_loader(
    dataset, batch_size=32, train_ratio=0.9, val_ratio=0.05
)
trainer = Trainer(
    model=chgnet,
    targets="efsm",
    optimizer="Adam",
    criterion="MSE",
    learning_rate=1e-2,
    epochs=50,
    use_device="cuda",
)

trainer.train(train_loader, val_loader, test_loader)
```

#### Notes for Training

Check [fine-tuning example notebook](https://github.com/CederGroupHub/chgnet/blob/main/examples/fine_tuning.ipynb)

1. The target quantity used for training should be energy/atom (not total energy) if you're fine-tuning the pretrained `CHGNet`.
2. The pretrained dataset of `CHGNet` comes from GGA+U DFT with [`MaterialsProject2020Compatibility`](https://github.com/materialsproject/pymatgen/blob/v2023.2.28/pymatgen/entries/compatibility.py#L826-L1102) corrections applied.
   The parameter for VASP is described in [`MPRelaxSet`](https://github.com/materialsproject/pymatgen/blob/v2023.2.28/pymatgen/io/vasp/sets.py#L862-L879).
   If you're fine-tuning with [`MPRelaxSet`](https://github.com/materialsproject/pymatgen/blob/v2023.2.28/pymatgen/io/vasp/sets.py#L862-L879), it is recommended to apply the [`MP2020`](https://github.com/materialsproject/pymatgen/blob/v2023.2.28/pymatgen/entries/compatibility.py#L826-L1102)
   compatibility to your energy labels so that they're consistent with the pretrained dataset.
3. If you're fine-tuning to functionals other than GGA, we recommend you refit the [`AtomRef`](https://github.com/CederGroupHub/chgnet/blob/main/chgnet/model/composition_model.py).
4. `CHGNet` stress is in units of GPa, and the unit conversion has already been included in
   [`dataset.py`](https://github.com/CederGroupHub/chgnet/blob/main/chgnet/data/dataset.py). So `VASP` stress can be directly fed to `StructureData`
5. To save time from graph conversion step for each training, we recommend you use [`GraphData`](https://github.com/CederGroupHub/chgnet/blob/main/chgnet/data/dataset.py) defined in
   [`dataset.py`](https://github.com/CederGroupHub/chgnet/blob/main/chgnet/data/dataset.py), which reads graphs directly from saved directory. To create saved graphs,
   see [`examples/make_graphs.py`](https://github.com/CederGroupHub/chgnet/blob/main/examples/make_graphs.py).

## MPtrj Dataset

The Materials Project trajectory (MPtrj) dataset used to pretrain CHGNet is available at
[figshare](https://figshare.com/articles/dataset/Materials_Project_Trjectory_MPtrj_Dataset/23713842).

The MPtrj dataset consists of all the GGA/GGA+U DFT calculations from the September 2022 [Materials Project](https://next-gen.materialsproject.org/).
By using the MPtrj dataset, users agree to abide the [Materials Project terms of use](https://next-gen.materialsproject.org/about/terms).

## Reference

If you use CHGNet or MPtrj dataset, please cite [this paper](https://nature.com/articles/s42256-023-00716-3):

```bib
@article{deng_2023_chgnet,
    title={CHGNet as a pretrained universal neural network potential for charge-informed atomistic modelling},
    DOI={10.1038/s42256-023-00716-3},
    journal={Nature Machine Intelligence},
    author={Deng, Bowen and Zhong, Peichen and Jun, KyuJung and Riebesell, Janosh and Han, Kevin and Bartel, Christopher J. and Ceder, Gerbrand},
    year={2023},
    pages={1–11}
}
```

## Development & Bugs

`CHGNet` is under active development, if you encounter any bugs in installation and usage,
please open an [issue](https://github.com/CederGroupHub/chgnet/issues). We appreciate your contributions!
