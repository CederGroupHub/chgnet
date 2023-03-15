from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Sequence

import torch
from torch import Tensor, nn

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
        graph_id: str = None,
        mp_id: str = None,
        composition: str = None,
    ) -> None:
        """Initialize the crystal graph.
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
                Default = None.

        Returns:
            Crystal Graph.
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

    def to(self, device="cpu") -> CrystalGraph:
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

    def save(self, fname: str = None, save_dir: str = "./") -> str:
        if fname is not None:
            save_name = os.path.join(save_dir, fname)
        elif self.graph_id is not None:
            save_name = os.path.join(save_dir, f"{self.graph_id}.pt")
        else:
            save_name = os.path.join(save_dir, f"{self.composition}.pt")
        torch.save(self.to_dict(), f=save_name)
        return save_name

    @classmethod
    def from_file(cls, file_name) -> CrystalGraph:
        graph = torch.load(file_name)
        return graph

    @classmethod
    def from_dict(cls, dic) -> CrystalGraph:
        return CrystalGraph(**dic)

    def __str__(self) -> str:
        """Details of the graph."""
        return (
            f"Crystal Graph {self.composition} \n"
            f"constructed using atom_graph_cutoff={self.atom_graph_cutoff}, "
            f"bond_graph_cutoff={self.bond_graph_cutoff} \n"
            f"(n_atoms={len(self.atomic_number)},"
            f"atom_graph={len(self.atom_graph)}, "
            f"bond_graph={len(self.bond_graph)})"
        )

    def __repr__(self) -> str:
        return str(self)


@dataclass
class BatchedGraph:
    """Batched crystal graph for parallel computing.

    Attributes:
        atomic_numbers (Tensor): atomic numbers vector
            [num_batch_atoms]
        bond_bases_ag (Tensor): bond bases vector for atom_graph
            [num_batch_bonds, num_radial]
        bond_bases_bg (Tensor): bond bases vector for atom_graph
            [num_batch_bonds, num_radial]
        angle_bases (Tensor): angle bases vector
            [num_batch_angles, num_angular]
        batched_atom_graph (Tensor) : batched atom graph adjacency list
            [num_batch_bonds, 2]
        batched_bond_graph (Tensor) : bond graph adjacency list
            [num_batch_angles, 2]
        atom_owners (Tensor): graph indices for each atom, used aggregate batched
            graph back to single graph
            [num_batch_atoms]
        directed2undirected (Tensor): the utility tensor used to quickly
            map directed edges to undirected edges in graph
            [num_directed]
        atom_positions (List[Tensor]): cartesian coordinates of the atoms
            from structures
            [num_batch_atoms]
        strains (List[Tensor]): a list of strains that's initialized to be zeros
            [batch_size]
        volumes (Tensor): the volume of each structure in the batch
            [batch_size]
    """

    atomic_numbers: Tensor
    bond_bases_ag: Tensor
    bond_bases_bg: Tensor
    angle_bases: Tensor
    batched_atom_graph: Tensor
    batched_bond_graph: Tensor
    atom_owners: Tensor
    directed2undirected: Tensor
    atom_positions: Sequence[Tensor]
    strains: Sequence[Tensor]
    volumes: Sequence[Tensor]

    @classmethod
    def from_graphs(
        cls,
        graphs: Sequence[CrystalGraph],
        bond_basis_expansion: nn.Module,
        angle_basis_expansion: nn.Module,
        compute_stress: bool = False,
    ) -> BatchedGraph:
        """Featurize and assemble a list of graphs.

        Args:
            graphs (List[Tensor]): a list of Crystal_Graphs
            bond_basis_expansion (nn.Module): bond basis expansion layer in CHGNet
            angle_basis_expansion (nn.Module): angle basis expansion layer in CHGNet
            compute_stress (bool): whether to compute stress

        Returns:
            assembled batch_graph that is ready for batched forward pass in CHGNet
        """
        atomic_numbers, atom_positions = [], []
        strains, volumes = [], []
        bond_bases_ag, bond_bases_bg, angle_bases = [], [], []
        batched_atom_graph, batched_bond_graph = [], []
        directed2undirected = []
        atom_owners = []
        atom_offset_idx = 0
        n_undirected = 0

        for graph_idx, graph in enumerate(graphs):
            # Atoms
            n_atom = graph.atomic_number.shape[0]
            atomic_numbers.append(graph.atomic_number)

            # Lattice
            if compute_stress:
                strain = graph.lattice.new_zeros([3, 3], requires_grad=True)
                lattice = graph.lattice @ (torch.eye(3).to(strain.device) + strain)
            else:
                strain = None
                lattice = graph.lattice
            volumes.append(torch.det(lattice))
            strains.append(strain)

            # Bonds
            atom_cart_coords = graph.atom_frac_coord @ lattice
            bond_basis_ag, bond_basis_bg, bond_vectors = bond_basis_expansion(
                center=atom_cart_coords[graph.atom_graph[:, 0]],
                neighbor=atom_cart_coords[graph.atom_graph[:, 1]],
                undirected2directed=graph.undirected2directed,
                image=graph.neighbor_image,
                lattice=lattice,
            )
            atom_positions.append(atom_cart_coords)
            bond_bases_ag.append(bond_basis_ag)
            bond_bases_bg.append(bond_basis_bg)

            # Indexes
            batched_atom_graph.append(graph.atom_graph + atom_offset_idx)
            directed2undirected.append(graph.directed2undirected + n_undirected)

            # Angles
            if len(graph.bond_graph) != 0:
                bond_vecs_i = torch.index_select(
                    bond_vectors, 0, graph.bond_graph[:, 2]
                )
                bond_vecs_j = torch.index_select(
                    bond_vectors, 0, graph.bond_graph[:, 4]
                )
                angle_basis = angle_basis_expansion(bond_vecs_i, bond_vecs_j)
                angle_bases.append(angle_basis)

                bond_graph = graph.bond_graph.new_zeros([graph.bond_graph.shape[0], 3])
                bond_graph[:, 0] = graph.bond_graph[:, 0] + atom_offset_idx
                bond_graph[:, 1] = graph.bond_graph[:, 1] + n_undirected
                bond_graph[:, 2] = graph.bond_graph[:, 3] + n_undirected
                batched_bond_graph.append(bond_graph)

            atom_owners.append(torch.ones(n_atom, requires_grad=False) * graph_idx)
            atom_offset_idx += n_atom
            n_undirected += len(bond_basis_ag)

        # Make Torch Tensors
        atomic_numbers = torch.cat(atomic_numbers, dim=0)
        bond_bases_ag = torch.cat(bond_bases_ag, dim=0)
        bond_bases_bg = torch.cat(bond_bases_bg, dim=0)
        angle_bases = (
            torch.cat(angle_bases, dim=0) if len(angle_bases) != 0 else torch.tensor([])
        )
        batched_atom_graph = torch.cat(batched_atom_graph, dim=0)
        if batched_bond_graph != []:
            batched_bond_graph = torch.cat(batched_bond_graph, dim=0)
        else:  # when bond graph is empty or disabled
            batched_bond_graph = torch.tensor([])
        atom_owners = (
            torch.cat(atom_owners, dim=0).type(torch.int).to(atomic_numbers.device)
        )
        directed2undirected = torch.cat(directed2undirected, dim=0)
        volumes = torch.tensor(volumes, dtype=datatype, device=atomic_numbers.device)

        return cls(
            atomic_numbers=atomic_numbers,
            bond_bases_ag=bond_bases_ag,
            bond_bases_bg=bond_bases_bg,
            angle_bases=angle_bases,
            batched_atom_graph=batched_atom_graph,
            batched_bond_graph=batched_bond_graph,
            atom_owners=atom_owners,
            directed2undirected=directed2undirected,
            atom_positions=atom_positions,
            strains=strains,
            volumes=volumes,
        )
