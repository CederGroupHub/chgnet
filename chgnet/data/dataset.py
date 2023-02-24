import os
import random
import functools
import torch
import numpy as np
from torch import Tensor
from pymatgen.core.structure import Structure
from torch.utils.data import Dataset, DataLoader
from gcn.chgnet.graph.crystal_graph import CrystalGraphConverter, Crystal_Graph
from gcn.toolkit import utils
from torch.utils.data.sampler import RandomSampler, SubsetRandomSampler
from typing import List, Union
import warnings

warnings.filterwarnings("ignore")

datatype = torch.float32


class StructureJsonData(Dataset):
    """
    read structure and targets from Json data
    """

    def __init__(
        self,
        json_dir: str,
        graph_converter: CrystalGraphConverter,
        targets: str = "e",
        **kwargs,
    ):
        """
        Initialize the dataset by reading Json files
        Args
            json_dir (str): json path or dir name that contain all the jsons
            targets (list[str]): list of key words for target properties.
                                 i.e. energy, force, stress
            crystal_featurizer: featurizer to convert pymatgen.core.Structure
                                to graphs (dictionaries)
        """
        self.json_dir = json_dir
        self.data = {}
        if os.path.isdir(json_dir):
            for json_path in os.listdir(json_dir):
                if json_path.endswith(".json"):
                    print(f"Importing: {json_path}")
                    self.data.update(utils.read_json(os.path.join(json_dir, json_path)))
        else:
            print(f"Importing: {json_dir}")
            self.data.update(utils.read_json(json_dir))
        self.graph_ids = list(self.data.keys())
        random.shuffle(self.graph_ids)
        print(f"{len(self.graph_ids)} structures imported")
        self.graph_converter = graph_converter

        self.energy_str = kwargs.pop("energy_str", "ef_per_atom")
        self.targets = targets
        self.failed_ids = []
        self.failed_graph_id = {}

    def __len__(self):
        return len(self.graph_ids)

    @functools.lru_cache(maxsize=None)  # Cache loaded structures
    def __getitem__(self, idx):
        """
        get one item in the dataset
        Returns:
            crystal_graph (crystal_graph): graph of the crystal structure
            targets (dict): dictionary of targets, keys include 'e' 'f' 's' 'm'
            material_id (str): material_id, which is the
        """
        if idx not in self.failed_ids:
            try:
                graph_id = self.graph_ids[idx]
                if "material_id" in self.data[graph_id].keys():
                    mp_id = self.data[graph_id]["material_id"]
                else:
                    mp_id = graph_id
                structure = Structure.from_dict(self.data[graph_id]["structure"])
                crystal_graph = self.graph_converter(
                    structure, graph_id=graph_id, mp_id=mp_id
                )
                targets = {}
                for i in self.targets:
                    if i == "e":
                        energy = self.data[graph_id][self.energy_str]
                        targets["e"] = torch.tensor(energy, dtype=datatype)
                    elif i == "f":
                        force = self.data[graph_id]["forces"]
                        targets["f"] = torch.tensor(force, dtype=datatype)
                    elif i == "s":
                        stress = self.data[graph_id]["stress"]
                        # Convert VASP stress
                        targets["s"] = torch.tensor(stress, dtype=datatype) * (-0.1)
                    elif i == "m":
                        mag = structure.site_properties["magmom"]
                        # use absolute value for magnetic moments
                        if mag != None:
                            targets["m"] = torch.abs(torch.tensor(mag, dtype=datatype))
                        else:
                            targets["m"] = None

                return crystal_graph, targets

            # Omit structures with isolated atoms. Return another random selected structure
            except:
                graph_id = self.graph_ids[idx]
                structure = Structure.from_dict(self.data[graph_id]["structure"])
                self.failed_graph_id[graph_id] = structure.composition.formula
                self.failed_ids.append(idx)
                idx = random.randint(0, len(self) - 1)
                return self.__getitem__(idx)
        else:
            idx = random.randint(0, len(self) - 1)
            return self.__getitem__(idx)


class CIFData(Dataset):
    """
    read structure and targets from Json data
    """

    def __init__(
        self,
        data_dir: str,
        graph_converter: CrystalGraphConverter,
        targets: str = "e",
        **kwargs,
    ):
        """
        Initialize the dataset by reading Json files
        Args
            json_dir (str): json path or dir name that contain all the jsons
            targets (list[str]): list of key words for target properties.
                                 i.e. energy, force, stress
            crystal_featurizer: featurizer to convert pymatgen.core.Structure
                                to graphs (dictionaries)
        """
        self.data_dir = data_dir
        self.data = utils.read_json(os.path.join(data_dir, "targets.json"))
        self.graph_ids = list(self.data.keys())
        random.shuffle(self.graph_ids)
        print(f"{data_dir}: {len(self.graph_ids)} structures imported")
        self.graph_converter = graph_converter

        self.energy_str = kwargs.pop("energy_str", "ef_per_atom")
        self.targets = targets
        self.failed_idx = []
        self.failed_graph_id = {}

    def __len__(self):
        return len(self.graph_ids)

    @functools.lru_cache(maxsize=None)  # Cache loaded structures
    def __getitem__(self, idx):
        """
        get one item in the dataset
        Returns:
            crystal_graph (dict): graph of the crystal structure
            targets (list): list of targets. i.e. energy, force, stress
            material_id (str): material_id, which is the
        """
        if idx not in self.failed_idx:
            try:
                graph_id = self.graph_ids[idx]
                if "material_id" in self.data[graph_id].keys():
                    mp_id = self.data[graph_id]["material_id"]
                else:
                    mp_id = graph_id
                structure = Structure.from_file(
                    os.path.join(self.data_dir, f"{graph_id}.cif")
                )
                crystal_graph = self.graph_converter(
                    structure, graph_id=graph_id, mp_id=mp_id
                )
                targets = {}
                for i in self.targets:
                    if i == "e":
                        energy = self.data[graph_id][self.energy_str]
                        targets["e"] = torch.tensor(energy, dtype=datatype)
                    elif i == "f":
                        force = self.data[graph_id]["forces"]
                        targets["f"] = torch.tensor(force, dtype=datatype)
                    elif i == "s":
                        stress = self.data[graph_id]["stress"]
                        # Convert VASP stress
                        targets["s"] = torch.tensor(stress, dtype=datatype) * (-0.1)
                    elif i == "m":
                        mag = self.data[graph_id]["magmom"]
                        # use absolute value for magnetic moments
                        targets["m"] = torch.abs(torch.tensor(mag, dtype=datatype))
                return crystal_graph, targets

            # Omit structures with isolated atoms. Return another random selected structure
            except:
                try:
                    graph_id = self.graph_ids[idx]
                except:
                    print(idx, len(self.graph_ids))
                structure = Structure.from_file(
                    os.path.join(self.data_dir, f"{graph_id}.cif")
                )
                self.failed_graph_id[graph_id] = structure.composition.formula
                self.failed_idx.append(idx)
                idx = random.randint(0, len(self) - 1)
                return self.__getitem__(idx)
        else:
            idx = random.randint(0, len(self) - 1)
            return self.__getitem__(idx)


class GraphData(Dataset):
    """
    Read graphs
        this is compatible with the graph.pt documents made by make_graphs.py
    """

    def __init__(
        self,
        graph_path: str,
        labels: Union[str, dict] = "labels.json",
        targets: str = "efsm",
        exclude: Union[str, list] = None,
        **kwargs,
    ):
        """
        Initialize the dataset
        Args
            graph_path (str): path that contain all the graphs, labels.json
            targets (list[str]): list of key words for target properties.
                                 i.e. 'efs'
        """
        self.graph_path = graph_path
        if isinstance(labels, str):
            labels = os.path.join(graph_path, labels)
            print(f"Importing: {labels}")
            self.labels = utils.read_json(labels)
        elif isinstance(labels, dict):
            self.labels = labels

        if isinstance(exclude, str):
            self.excluded_graph = utils.read_json(exclude)
        elif isinstance(exclude, list):
            self.excluded_graph = exclude
        else:
            self.excluded_graph = []

        self.keys = []
        for mp_id, dic in self.labels.items():
            for graph_id, _ in dic.items():
                self.keys.append((mp_id, graph_id))
        random.shuffle(self.keys)
        print(f"{len(self.labels.keys())} mp_ids, {self.__len__()} frames imported")
        if self.excluded_graph is not None:
            print(f"{len(self.excluded_graph)} graphs are pre-excluded")

        self.energy_str = kwargs.pop("energy_str", "ef_per_atom")
        self.targets = targets
        self.failed_idx = []
        self.failed_graph_id = []

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        """
        get one item in the dataset
        Returns:
            crystal_graph (dict): graph of the crystal structure
            targets (list): list of targets. i.e. energy, force, stress
            material_id (str): material_id, which is the
        """
        if idx not in self.failed_idx:
            mp_id, graph_id = self.keys[idx]
            if [mp_id, graph_id] in self.excluded_graph:
                self.failed_graph_id.append(graph_id)
                self.failed_idx.append(idx)
                idx = random.randint(0, len(self) - 1)
                return self.__getitem__(idx)
            try:
                graph_path = os.path.join(self.graph_path, f"{graph_id}.pt")
                crystal_graph = Crystal_Graph.from_file(graph_path)
                targets = {}
                for i in self.targets:
                    if i == "e":
                        energy = self.labels[mp_id][graph_id][self.energy_str]
                        targets["e"] = torch.tensor(energy, dtype=datatype)
                    elif i == "f" or i == "force":
                        force = self.labels[mp_id][graph_id]["force"]
                        targets["f"] = torch.tensor(force, dtype=datatype)
                    elif i == "s" or i == "stresses":
                        stress = self.labels[mp_id][graph_id]["stress"]
                        # Convert VASP stress
                        targets["s"] = torch.tensor(stress, dtype=datatype) * (-0.1)
                    elif i == "m":
                        mag = self.labels[mp_id][graph_id]["magmom"]
                        # use absolute value for magnetic moments
                        if mag is None:
                            targets["m"] = None
                        else:
                            targets["m"] = torch.abs(torch.tensor(mag, dtype=datatype))
                return crystal_graph, targets

            # Omit failed structures. Return another random selected structure
            except:
                self.failed_graph_id.append(graph_id)
                self.failed_idx.append(idx)
                idx = random.randint(0, len(self) - 1)
                return self.__getitem__(idx)
        else:
            idx = random.randint(0, len(self) - 1)
            return self.__getitem__(idx)

    def get_train_val_test_loader(
        self,
        train_key,
        val_key,
        test_key,
        batch_size=32,
        num_workers=0,
        pin_memory=True,
    ):
        """
        get data loaders
        Returns
        -------
        train_loader: torch.utils.data.DataLoader
          DataLoader that random samples the training data.
        val_loader: torch.utils.data.DataLoader
          DataLoader that random samples the validation data.
        test_loader: torch.utils.data.DataLoader
          DataLoader that random samples the test data, returns if
            return_test=True.
        """
        train_labels, val_labels, test_labels = {}, {}, {}
        for mp_id in train_key:
            try:
                train_labels[mp_id] = self.labels.pop(mp_id)
            except:
                continue
        train_dataset = GraphData(
            graph_path=self.graph_path,
            labels=train_labels,
            targets=self.targets,
            exclude=self.excluded_graph,
            energy_str=self.energy_str,
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            collate_fn=collate_graphs,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )

        # Val
        for mp_id in val_key:
            try:
                val_labels[mp_id] = self.labels.pop(mp_id)
            except:
                continue
        val_dataset = GraphData(
            graph_path=self.graph_path,
            labels=val_labels,
            targets=self.targets,
            exclude=self.excluded_graph,
            energy_str=self.energy_str,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            collate_fn=collate_graphs,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )

        # Test
        if test_key is not None:
            for mp_id in test_key:
                try:
                    test_labels[mp_id] = self.labels.pop(mp_id)
                except:
                    continue
            test_dataset = GraphData(
                graph_path=self.graph_path,
                labels=test_labels,
                targets=self.targets,
                exclude=self.excluded_graph,
                energy_str=self.energy_str,
            )
            test_loader = DataLoader(
                test_dataset,
                batch_size=batch_size,
                collate_fn=collate_graphs,
                num_workers=num_workers,
                pin_memory=pin_memory,
            )
        else:
            test_loader = None
        return train_loader, val_loader, test_loader


class StructureJsonData(Dataset):
    """
    read structure and targets from MPtrj dataset
    """

    def __init__(
        self,
        data: Union[str, dict],
        graph_converter: CrystalGraphConverter,
        targets: str = "efsm",
        **kwargs,
    ):
        """
        Initialize the dataset by reading Json files
        Args
            json_dir (str): json path or dir name that contain all the jsons
            graph_converter: converter to convert pymatgen.core.Structure
                    to graphs
            targets (list[str]): list of key words for target properties.
                                 i.e. energy, force, stress
        """
        if isinstance(data, str):
            self.data = {}
            if os.path.isdir(data):
                for json_path in os.listdir(data):
                    if json_path.endswith(".json"):
                        print(f"Importing: {json_path}")
                        self.data.update(utils.read_json(os.path.join(data, json_path)))
            else:
                print(f"Importing: {data}")
                self.data.update(utils.read_json(data))
        elif isinstance(data, dict):
            self.data = data
        else:
            raise Exception("please provide a json path or dictionary")

        self.keys = []
        for mp_id, dic in self.data.items():
            for graph_id, _ in dic.items():
                self.keys.append((mp_id, graph_id))
        random.shuffle(self.keys)
        print(f"{len(self.data.keys())} mp_ids, {self.__len__()} structures imported")
        self.graph_converter = graph_converter
        self.energy_str = kwargs.pop("energy_str", "energy_per_atom")
        self.targets = targets
        self.failed_idx = []
        self.failed_graph_id = {}

    def __len__(self):
        return len(self.keys)

    @functools.lru_cache(maxsize=None)  # Cache loaded structures
    def __getitem__(self, idx):
        """
        get one item in the dataset
        Returns:
            crystal_graph (dict): graph of the crystal structure
            targets (list): list of targets. i.e. energy, force, stress
            material_id (str): material_id, which is the
        """
        if idx not in self.failed_idx:
            mp_id, graph_id = self.keys[idx]
            try:
                struc = Structure.from_dict(self.data[mp_id][graph_id]["structure"])
                crystal_graph = self.graph_converter(
                    struc, graph_id=graph_id, mp_id=mp_id
                )

                targets = {}
                for i in self.targets:
                    if i == "e":
                        energy = self.data[mp_id][graph_id][self.energy_str]
                        targets["e"] = torch.tensor(energy, dtype=datatype)
                    elif i == "f" or i == "force":
                        force = self.data[mp_id][graph_id]["force"]
                        targets["f"] = torch.tensor(force, dtype=datatype)
                    elif i == "s" or i == "stresses":
                        stress = self.data[mp_id][graph_id]["stress"]
                        # Convert VASP stress
                        targets["s"] = torch.tensor(stress, dtype=datatype) * (-0.1)
                    elif i == "m":
                        mag = self.data[mp_id][graph_id]["magmom"]
                        # use absolute value for magnetic moments
                        if mag is None:
                            targets["m"] = None
                        else:
                            targets["m"] = torch.abs(torch.tensor(mag, dtype=datatype))
                return crystal_graph, targets

            # Omit structures with isolated atoms. Return another random selected structure
            except:
                structure = Structure.from_dict(self.data[mp_id][graph_id]["structure"])
                self.failed_graph_id[graph_id] = structure.composition.formula
                self.failed_idx.append(idx)
                idx = random.randint(0, len(self) - 1)
                return self.__getitem__(idx)
        else:
            idx = random.randint(0, len(self) - 1)
            return self.__getitem__(idx)

    def get_train_val_test_loader(
        self,
        train_key=None,
        val_key=None,
        test_key=None,
        train_ratio=0.8,
        val_ratio=0.1,
        batch_size=64,
        num_workers=0,
        pin_memory=True,
    ):
        """
        get data loaders
        Returns
        -------
        train_loader: torch.utils.data.DataLoader
          DataLoader that random samples the training data.
        val_loader: torch.utils.data.DataLoader
          DataLoader that random samples the validation data.
        test_loader: torch.utils.data.DataLoader
          DataLoader that random samples the test data, returns if
            return_test=True.
        """
        train_data, val_data, test_data = {}, {}, {}
        if train_key == None:
            mp_ids = list(self.data.keys())
            random.shuffle(mp_ids)
            n_train = int(train_ratio * len(mp_ids))
            n_val = int(val_ratio * len(mp_ids))
            train_key = mp_ids[:n_train]
            val_key = mp_ids[n_train : n_train + n_val]
            test_key = mp_ids[n_train + n_val :]
        for mp_id in train_key:
            train_data[mp_id] = self.data.pop(mp_id)
        train_dataset = StructureJsonData(
            data=train_data,
            graph_converter=self.graph_converter,
            targets=self.targets,
            energy_str=self.energy_str,
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            collate_fn=collate_graphs,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
        for mp_id in val_key:
            val_data[mp_id] = self.data.pop(mp_id)
        val_dataset = StructureJsonData(
            data=val_data,
            graph_converter=self.graph_converter,
            targets=self.targets,
            energy_str=self.energy_str,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            collate_fn=collate_graphs,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
        if test_key is not None:
            for mp_id in test_key:
                test_data[mp_id] = self.data.pop(mp_id)
            test_dataset = StructureJsonData(
                data=test_data,
                graph_converter=self.graph_converter,
                targets=self.targets,
                energy_str=self.energy_str,
            )
            test_loader = DataLoader(
                test_dataset,
                batch_size=batch_size,
                collate_fn=collate_graphs,
                num_workers=num_workers,
                pin_memory=pin_memory,
            )
        else:
            test_loader = None
        return train_loader, val_loader, test_loader


class StructureData(Dataset):
    """
    dataset of structures
    """

    def __init__(
        self,
        structures: List,
        energies: List,
        forces: List,
        stresses: List = None,
        magmoms: List = None,
        graph_converter: CrystalGraphConverter = None,
    ):
        """
        Initialize the dataset by reading Json files
        Args
            json_dir (str): json path or dir name that contain all the jsons
            graph_converter: converter to convert pymatgen.core.Structure
                    to graphs
            targets (list[str]): list of key words for target properties.
                                 i.e. energy, force, stress
        """
        self.structures = structures
        self.energies = energies
        self.forces = forces
        self.stresses = stresses
        self.magmoms = magmoms
        self.keys = np.arange(len(structures))
        random.shuffle(self.keys)
        print(f"{len(self.structures)} structures imported")
        if graph_converter is not None:
            self.graph_converter = graph_converter
        else:
            self.graph_converter = CrystalGraphConverter(
                atom_graph_cutoff=5, bond_graph_cutoff=3
            )
        self.failed_idx = []
        self.failed_graph_id = {}

    def __len__(self):
        return len(self.keys)

    @functools.lru_cache(maxsize=None)  # Cache loaded structures
    def __getitem__(self, idx):
        """
        get one item in the dataset
        Returns:
            crystal_graph (dict): graph of the crystal structure
            targets (list): list of targets. i.e. energy, force, stress
            material_id (str): material_id, which is the
        """
        if idx not in self.failed_idx:
            graph_id = self.keys[idx]
            try:
                struc = Structure.from_dict(self.structures[graph_id])
                crystal_graph = self.graph_converter(
                    struc, graph_id=graph_id, mp_id=graph_id
                )
                targets = {
                    "e": torch.tensor(self.energies[graph_id], dtype=datatype),
                    "f": torch.tensor(self.forces[graph_id], dtype=datatype),
                }
                if self.stresses is not None:
                    # Convert VASP stress
                    targets["s"] = torch.tensor(
                        self.stresses[graph_id], dtype=datatype
                    ) * (-0.1)
                if self.magmoms is not None:
                    mag = self.magmoms[graph_id]
                    # use absolute value for magnetic moments
                    if mag is None:
                        targets["m"] = None
                    else:
                        targets["m"] = torch.abs(torch.tensor(mag, dtype=datatype))

                return crystal_graph, targets

            # Omit structures with isolated atoms. Return another random selected structure
            except:
                struc = Structure.from_dict(self.structures[graph_id])
                self.failed_graph_id[graph_id] = struc.composition.formula
                self.failed_idx.append(idx)
                idx = random.randint(0, len(self) - 1)
                return self.__getitem__(idx)
        else:
            idx = random.randint(0, len(self) - 1)
            return self.__getitem__(idx)


def collate_graphs(batch_data: List):
    """
    Collate of list of (graph, target) into batch (a large graph),
    this customized collate function ensures auto diff
    Args:
        batch_data (list): list of (graph, target(dict))
    Returns:
        graphs (List): a list of graphs
        targets (Dict): dictionary of targets, where key and values are:
            e (Tensor): energies of the structures [batch_size]
            f (Tensor): forces of the structures [n_batch_atoms, 3]
            s (Tensor): stresses of the structures [3*batch_size, 3]
            m (Tensor): magmom of the structures [n_batch_atoms]
    """
    graphs, energy = [], []
    all_targets = {key: [] for key in batch_data[0][1].keys()}
    for (graph, targets) in batch_data:
        graphs.append(graph)
        for target, value in targets.items():
            all_targets[target].append(value)
    if "e" in all_targets.keys():
        all_targets["e"] = torch.tensor(all_targets["e"], dtype=datatype)
    return graphs, all_targets


def get_train_val_test_loader(
    dataset,
    batch_size=64,
    train_ratio=0.8,
    val_ratio=0.1,
    return_test=True,
    num_workers=0,
    pin_memory=True,
):
    """
    Utility function for dividing a dataset to train, val, test datasets.
    !!! The dataset needs to be shuffled before using the function,
        otherwise the train,val,test partition is not random
    Parameters
    ----------
    dataset: torch.utils.data.Dataset
      The full dataset to be divided.
    collate_fn: torch.utils.data.DataLoader
    batch_size: int
    train_ratio: float
    val_ratio: float
    test_ratio: float
    return_test: bool
      Whether to return the test dataset loader. If False, the last test_size
      data will be hidden.
    num_workers: int
    pin_memory: bool
    Returns
    -------
    train_loader: torch.utils.data.DataLoader
      DataLoader that random samples the training data.
    val_loader: torch.utils.data.DataLoader
      DataLoader that random samples the validation data.
    (test_loader): torch.utils.data.DataLoader
      DataLoader that random samples the test data, returns if
        return_test=True.
    """
    total_size = len(dataset)
    indices = list(range(total_size))
    random.shuffle(indices)
    train_size = int(train_ratio * total_size)
    val_size = int(val_ratio * total_size)

    train_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=collate_graphs,
        sampler=SubsetRandomSampler(indices=indices[0:train_size]),
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=collate_graphs,
        sampler=SubsetRandomSampler(
            indices=indices[train_size : train_size + val_size]
        ),
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    if return_test:
        test_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            collate_fn=collate_graphs,
            sampler=SubsetRandomSampler(indices=indices[train_size + val_size :]),
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
        return train_loader, val_loader, test_loader
    else:
        return train_loader, val_loader


def get_loader(dataset, batch_size=64, num_workers=0, pin_memory=True):
    """
    Get dataloader from dataset
    Args:
    dataset (torch.utils.data.Dataset): the dataset
    batch_size (int): batch size
    num_workers (int):
    pin_memory (bool):
    Returns:
    data_loader (torch.utils.data.DataLoader): dataloader object ready to train
    """

    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=collate_graphs,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return data_loader
