from __future__ import print_function, division
import os
import sys
import torch
import torch.nn as nn
from torch import Tensor
from pymatgen.core.structure import Structure
from .graph import Graph, Node

datatype = torch.float32


class Crystal_Graph(object):
    """
    A data class for crystal graph
    """
    def __init__(
        self,
        atomic_number: Tensor,
        atom_frac_coord: Tensor,
        atom_graph: Tensor,
        atom_graph_cutoff: float,
        neighbor_image: Tensor,
        directed2undirected: Tensor,
        undirected2directed: Tensor,
        bond_graph: Tensor,
        bond_graph_cutoff: float,
        lattice: Tensor,
        graph_id: str = None,
        mp_id: str = None,
        composition: str = None,
    ):
        """
        Initialize the crystal graph.
        Attention! This data class is not intended to be created manually.
                   Crystal Graph should be returned by a CrystalGraphConverter
        Args:
            atomic_number (Tensor): the atomic numbers of atoms in the structure [n_atom]
            atom_frac_coord (Tensor): the fractional coordinates of the atoms [n_atom, 3]
            atom_graph (Tensor): a directed graph adjacency list,
                (center atom indices, neighbor atom indices, undirected bond index)
                for bonds in bond_fea [2*n_bond, 3]
            atom_graph_cutoff (float): the cutoff radius to draw edges in atom_graph
            neighbor_image (Tensor): the periodic image specifying the location of neighboring atom [2*n_bond, 3]
            directed2undirected (Tensor): the mapping from directed edge index to undirected edge index
                for the atom graph
            undirected2directed (Tensor): the mapping from undirected edge index to directed edge index,
                this is essentially the inverse mapping of half of the third column in atom_graph,
                this tensor is needed for computation efficiency. [n_bond]
            bond_graph (Tensor): a directed graph adjacency list,
                (atom indices, 1st undirected bond idx, 1st directed bond idx,
                 2nd undirected bond idx, 2nd directed bond idx)
                for angles in angle_fea [n_angle, 5]
            bond_graph_cutoff (float): the cutoff bond length to include bond as nodes in bond_graph
            lattice (Tensor): lattices of the input structure [3, 3]
            graph_id (str or None): an id to keep track of this crystal graph
                Default = None
            mp_id (str) or None: Materials Project id of this structure
                Default = None
            composition: the chemical composition of the compound, used just for better tracking
                Default = None
        Returns:
            Crystal Graph
        """
        super().__init__()
        self.atomic_number = atomic_number
        self.atom_frac_coord = atom_frac_coord
        self.atom_graph = atom_graph
        self.atom_graph_cutoff = atom_graph_cutoff
        self.neighbor_image = neighbor_image
        self.directed2undirected = directed2undirected
        self.undirected2directed = undirected2directed
        self.bond_graph = bond_graph
        self.bond_graph_cutoff = bond_graph_cutoff
        self.lattice = lattice
        self.graph_id = graph_id
        self.mp_id = mp_id
        self.composition = composition
        assert len(undirected2directed) == 0.5 * len(
            directed2undirected
        ), f"Error: {graph_id} number of directed index != 2 * number of undirected index!"

    def to(self, device="cpu"):
        return Crystal_Graph(
            atomic_number=self.atomic_number.to(device),
            atom_frac_coord=self.atom_frac_coord.to(device),
            atom_graph=self.atom_graph.to(device),
            atom_graph_cutoff=self.atom_graph_cutoff,
            neighbor_image=self.neighbor_image.to(device),
            directed2undirected=self.directed2undirected.to(device),
            undirected2directed=self.undirected2directed.to(device),
            bond_graph=self.bond_graph.to(device),
            bond_graph_cutoff=self.bond_graph_cutoff,
            lattice=self.lattice.to(device),
            graph_id=self.graph_id,
            mp_id=self.mp_id,
            composition=self.composition,
        )

    def to_dict(self):
        return {
            "atomic_number": self.atomic_number,
            "atom_frac_coord": self.atom_frac_coord,
            "atom_graph": self.atom_graph,
            "atom_graph_cutoff": self.atom_graph_cutoff,
            "neighbor_image": self.neighbor_image,
            "directed2undirected": self.directed2undirected,
            "undirected2directed": self.undirected2directed,
            "bond_graph": self.bond_graph,
            "bond_graph_cutoff": self.bond_graph_cutoff,
            "lattice": self.lattice,
            "graph_id": self.graph_id,
            "mp_id": self.mp_id,
            "composition": self.composition,
        }

    def save(self, fname=None, save_dir="./"):
        if fname is not None:
            save_name = os.path.join(save_dir, fname)
        elif self.graph_id is not None:
            save_name = os.path.join(save_dir, f"{self.graph_id}.pt")
        else:
            save_name = os.path.join(save_dir, f"{self.composition}.pt")
        torch.save(self.to_dict(), f=save_name)
        return save_name

    @classmethod
    def from_file(cls, file_name):
        graph = torch.load(file_name)
        return graph

    @classmethod
    def from_dict(cls, dic):
        return Crystal_Graph(**dic)

    def __str__(self):
        """
        Details of the graph
        """
        return (
            f"Crystal Graph {self.composition} \n"
            f"constructed using atom_graph_cutoff={self.atom_graph_cutoff}, "
            f"bond_graph_cutoff={self.bond_graph_cutoff} \n"
            f"(n_atoms={len(self.atomic_number)},"
            f"atom_graph={len(self.atom_graph)}, "
            f"bond_graph={len(self.bond_graph)})"
        )

    def __repr__(self):
        return str(self)


class CrystalGraphConverter(nn.Module):
    """
    Convert a pymatgen.core.Structure to a Crystal_Graph,
    where only the minimal essential information is kept
    """

    def __init__(
        self,
        atom_graph_cutoff: float,
        bond_graph_cutoff: float = None,
        verbose: bool = True,
    ):
        """
        Initialize the Crystal Graph Converter
        Args:
            atom_graph_cutoff (float): cutoff radius to search for neighboring atom in atom_graph
            bond_graph_cutoff (float): bond length threshold to include bond in bond_graph
            verbose (bool): whether to print initialization message
        """
        super().__init__()
        self.atom_graph_cutoff = atom_graph_cutoff
        if bond_graph_cutoff is None:
            self.bond_graph_cutoff = atom_graph_cutoff
        else:
            self.bond_graph_cutoff = bond_graph_cutoff
        if verbose:
            print(
                f"CrystalGraphConverter initialized with atom_cutoff{atom_graph_cutoff}, "
                f"bond_cutoff{bond_graph_cutoff}"
            )

    def forward(self, structure: Structure, graph_id=None, mp_id=None) -> Crystal_Graph:
        """
        convert a structure, return a Crystal_Graph
        Args:
            structure (pymatgen.core.Structure): structure to convert
            graph_id (str): an id to keep track of this crystal graph
                Default = None
            mp_id (str): Materials Project id of this structure
                Default = None
        Return:
            Crystal_Graph
        """
        n_atom = int(structure.composition.num_atoms)
        atomic_number = torch.tensor(
            [i.specie.Z for i in structure], dtype=int, requires_grad=False
        )
        atom_frac_coord = torch.tensor(
            structure.frac_coords, dtype=datatype, requires_grad=True
        )
        lattice = torch.tensor(
            structure.lattice.matrix, dtype=datatype, requires_grad=True
        )
        center_index, neighbor_index, image, distance = self.get_neighbors(structure)

        # Make Graph
        graph = Graph([Node(index=i) for i in range(n_atom)])
        for i, j, im, d in zip(center_index, neighbor_index, image, distance):
            graph.add_edge(center_index=i, neighbor_index=j, image=im, distance=d)

        # Atom Graph
        atom_graph, directed2undirected = graph.adjacency_list()
        atom_graph = torch.tensor(atom_graph, dtype=torch.int64)
        directed2undirected = torch.tensor(directed2undirected, dtype=torch.int64)

        # Bond Graph
        try:
            bond_graph, undirected2directed = graph.line_graph_adjacency_list(
                cutoff=self.bond_graph_cutoff
            )
        except:
            structure.to(filename="bond_graph_error.cif")
            sys.exit()
        bond_graph = torch.tensor(bond_graph, dtype=torch.int64)
        undirected2directed = torch.tensor(undirected2directed, dtype=torch.int64)

        # Check if graph has isolated atom
        has_no_isolated_atom = set(set(range(n_atom))).issubset(center_index)
        if has_no_isolated_atom is False:
            # Discard this structure if it has isolated atom in the graph
            raise ValueError(
                f"{graph_id} has isolated atom with r_cutoff={self.atom_graph_cutoff}, should be skipped"
            )
        return Crystal_Graph(
            atomic_number=atomic_number,
            atom_frac_coord=atom_frac_coord,
            atom_graph=atom_graph,
            neighbor_image=torch.tensor(image, dtype=datatype),
            directed2undirected=directed2undirected,
            undirected2directed=undirected2directed,
            bond_graph=bond_graph,
            lattice=lattice,
            graph_id=graph_id,
            mp_id=mp_id,
            composition=structure.composition.formula,
            atom_graph_cutoff=self.atom_graph_cutoff,
            bond_graph_cutoff=self.bond_graph_cutoff,
        )

    def get_neighbors(self, structure: Structure):
        """
        Get neighbot information from pymatgen utility function

        Args:
            structure(pymatgen.core.Structure): a structure to compute

        Returns:
            center_index, neighbor_index, image, distance
        """
        center_index, neighbor_index, image, distance = structure.get_neighbor_list(
            r=self.atom_graph_cutoff, sites=structure.sites, numerical_tol=1e-8
        )
        return center_index, neighbor_index, image, distance

    def as_dict(self):
        """
        Save the args of the graph converter
        """
        return {
            "atom_graph_cutoff": self.atom_graph_cutoff,
            "bond_graph_cutoff": self.bond_graph_cutoff
        }

    @classmethod
    def from_dict(cls, dict):
        """
        Create converter from dictionary
        """
        return CrystalGraphConverter(**dict)