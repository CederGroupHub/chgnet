from __future__ import annotations

import os
from typing import Any

import torch
from torch import Tensor

datatype = torch.float32


class CrystalGraph:
    """A data class for crystal graph."""

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
        graph_id: str | None = None,
        mp_id: str | None = None,
        composition: str | None = None,
    ) -> None:
        """Initialize the crystal graph.
        Attention! This data class is not intended to be created manually.
                   Crystal Graph should be returned by a CrystalGraphConverter
        Args:
            atomic_number (Tensor): the atomic numbers of atoms in the structure
                [n_atom]
            atom_frac_coord (Tensor): the fractional coordinates of the atoms
                [n_atom, 3]
            atom_graph (Tensor): a directed graph adjacency list,
                (center atom indices, neighbor atom indices, undirected bond index)
                for bonds in bond_fea
                [num_directed_bonds, 2]
            atom_graph_cutoff (float): the cutoff radius to draw edges in atom_graph
            neighbor_image (Tensor): the periodic image specifying the location of
                neighboring atom
                see: https://github.com/materialsproject/pymatgen/blob/ca2175c762e37ea7
                c9f3950ef249bc540e683da1/pymatgen/core/structure.py#L1485-L1541
                [num_directed_bonds, 3]
            directed2undirected (Tensor): the mapping from directed edge index to
                undirected edge index for the atom graph
                [num_directed_bonds]
            undirected2directed (Tensor): the mapping from undirected edge index to
                one of its directed edge index, this is essentially the inverse
                mapping of the directed2undirected this tensor is needed for
                computation efficiency.
                Note that num_directed_bonds = 2 * num_undirected_bonds
                [num_undirected_bonds]
            bond_graph (Tensor): a directed graph adjacency list,
                (atom indices, 1st undirected bond idx, 1st directed bond idx,
                 2nd undirected bond idx, 2nd directed bond idx)
                for angles in angle_fea
                [n_angle, 5]
            bond_graph_cutoff (float): the cutoff bond length to include bond
                as nodes in bond_graph
            lattice (Tensor): lattices of the input structure
                [3, 3]
            graph_id (str or None): an id to keep track of this crystal graph
                Default = None
            mp_id (str) or None: Materials Project id of this structure
                Default = None
            composition: Chemical composition of the compound, used just for
                better tracking of the graph
                Default = None.

        Raises:
            ValueError: if len(directed2undirected) != 2 * len(undirected2directed)
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
        if len(directed2undirected) != 2 * len(undirected2directed):
            raise ValueError(
                f"{graph_id} number of directed indices != 2 * number of undirected indices!"
            )

    def to(self, device: str = "cpu") -> CrystalGraph:
        """Move the graph to a device. Default = 'cpu'."""
        return CrystalGraph(
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

    def to_dict(self) -> dict[str, Any]:
        """Convert the graph to a dictionary."""
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

    def save(self, fname: str | None = None, save_dir: str = ".") -> str:
        """Save the graph to a file.

        Args:
            fname (str, optional): File name. Defaults to None.
            save_dir (str, optional): Directory to save the file. Defaults to ".".

        Returns:
            str: The path to the saved file.
        """
        if fname is not None:
            save_name = os.path.join(save_dir, fname)
        elif self.graph_id is not None:
            save_name = os.path.join(save_dir, f"{self.graph_id}.pt")
        else:
            save_name = os.path.join(save_dir, f"{self.composition}.pt")
        torch.save(self.to_dict(), f=save_name)
        return save_name

    @classmethod
    def from_file(cls, file_name: str) -> CrystalGraph:
        """Load a crystal graph from a file.

        Args:
            file_name (str): The path to the file.

        Returns:
            CrystalGraph: The loaded graph.
        """
        return torch.load(file_name)

    @classmethod
    def from_dict(cls, dic: dict[str, Any]) -> CrystalGraph:
        """Load a CrystalGraph from a dictionary."""
        return CrystalGraph(**dic)

    def __repr__(self) -> str:
        """Details of the graph."""
        return (
            f"Crystal Graph {self.composition} \n"
            f"constructed using atom_graph_cutoff={self.atom_graph_cutoff}, "
            f"bond_graph_cutoff={self.bond_graph_cutoff} \n"
            f"(n_atoms={len(self.atomic_number)}, "
            f"atom_graph={len(self.atom_graph)}, "
            f"bond_graph={len(self.bond_graph)})"
        )
