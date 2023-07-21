<h1 align="center">CHGNet</h1>

<h4 align="center">

[![Tests](https://github.com/CederGroupHub/chgnet/actions/workflows/test.yml/badge.svg)](https://github.com/CederGroupHub/chgnet/actions/workflows/test.yml)
[![Codacy Badge](https://app.codacy.com/project/badge/Coverage/e3bdcea0382a495d96408e4f84408e85)](https://app.codacy.com/gh/CederGroupHub/chgnet/dashboard?utm_source=gh&utm_medium=referral&utm_content=&utm_campaign=Badge_coverage)
[![arXiv](https://img.shields.io/badge/arXiv-2302.14231-blue)](https://arxiv.org/abs/2302.14231)
![GitHub repo size](https://img.shields.io/github/repo-size/CederGroupHub/chgnet)
[![PyPI](https://img.shields.io/pypi/v/chgnet?logo=pypi&logoColor=white)](https://pypi.org/project/chgnet?logo=pypi&logoColor=white)
[![Docs](https://img.shields.io/badge/API-Docs-blue)](https://chgnet.lbl.gov)

</h4>

A pretrained universal neural network potential for
**charge**-informed atomistic modeling
![Logo](https://raw.github.com/CederGroupHub/chgnet/main/site/static/chgnet-logo.png)
**C**rystal **H**amiltonian **G**raph neural **Net**work is pretrained on the GGA/GGA+U static and relaxation trajectories from Materials Project,
a comprehensive dataset consisting of 1.5 Million structures from 146k compounds spanning the whole periodic table.

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

You can install `chgnet` through `pip`:

```sh
pip install chgnet
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
for key in ("energy", "forces", "stress", "magmom"):
    print(f"CHGNet-predicted {key}={prediction[key[0]]}\n")
```

### Molecular Dynamics

Charge-informed molecular dynamics can be simulated with pretrained `CHGNet` through `ASE` environment

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
    use_device="cpu",  # use 'cuda' for faster MD
)
md.run(50)  # run a 0.1 ps MD simulation
```

Visualize the magnetic moments after the MD run

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

### Structure Optimization

`CHGNet` can perform fast structure optimization and provide site-wise magnetic moments. This makes it ideal for pre-relaxation and
`MAGMOM` initialization in spin-polarized DFT.

```python
from chgnet.model import StructOptimizer

relaxer = StructOptimizer()
result = relaxer.relax(structure)
print("CHGNet relaxed structure", result["final_structure"])
```

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

### Note

1. The energy used for training should be energy/atom if you're fine-tuning the pretrained `CHGNet`.
2. The pretrained dataset of `CHGNet` comes from GGA+U DFT with [`MaterialsProject2020Compatibility`](https://github.com/materialsproject/pymatgen/blob/v2023.2.28/pymatgen/entries/compatibility.py#L826-L1102).
   The parameter for VASP is described in [`MPRelaxSet`](https://github.com/materialsproject/pymatgen/blob/v2023.2.28/pymatgen/io/vasp/sets.py#L862-L879).
   If you're fine-tuning with [`MPRelaxSet`](https://github.com/materialsproject/pymatgen/blob/v2023.2.28/pymatgen/io/vasp/sets.py#L862-L879), it is recommended to apply the [`MP2020`](https://github.com/materialsproject/pymatgen/blob/v2023.2.28/pymatgen/entries/compatibility.py#L826-L1102)
   compatibility to your energy labels so that they're consistent with the pretrained dataset.
3. If you're fine-tuning to functionals other than GGA, we recommend you refit the [`AtomRef`](https://github.com/CederGroupHub/chgnet/blob/main/chgnet/model/composition_model.py).
4. `CHGNet` stress is in unit GPa, and the unit conversion has already been included in
   [`dataset.py`](https://github.com/CederGroupHub/chgnet/blob/main/chgnet/data/dataset.py). So `VASP` stress can be directly fed to `StructureData`
5. To save time from graph conversion step for each training, we recommend you use [`GraphData`](https://github.com/CederGroupHub/chgnet/blob/main/chgnet/data/dataset.py) defined in
   [`dataset.py`](https://github.com/CederGroupHub/chgnet/blob/main/chgnet/data/dataset.py), which reads graphs directly from saved directory. To create saved graphs,
   see [`examples/make_graphs.py`](https://github.com/CederGroupHub/chgnet/blob/main/examples/make_graphs.py).
6. Apple’s Metal Performance Shaders `MPS` is currently disabled until a stable version of `pytorch` for `MPS` is released.

## Reference

link to our paper:
<https://doi.org/10.48550/arXiv.2302.14231>

Please cite the following:

```bib
@article{deng_2023_chgnet,
  title={{CHGNet: Pretrained universal neural network potential for charge-informed atomistic modeling}},
  author={Deng, Bowen and Zhong, Peichen and Jun, KyuJung and Riebesell, Janosh and Han, Kevin and Bartel, Christopher J and Ceder, Gerbrand},
  journal={arXiv preprint arXiv:2302.14231},
  year={2023},
  url = {https://arxiv.org/abs/2302.14231}
}
```

## Development & Bugs

`CHGNet` is under active development, if you encounter any bugs in installation and usage,
please open an [issue](https://github.com/CederGroupHub/chgnet/issues). We appreciate your contributions!
