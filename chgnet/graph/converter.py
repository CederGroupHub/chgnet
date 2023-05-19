from __future__ import annotations

import sys
from typing import TYPE_CHECKING, Literal

import torch
from torch import Tensor, nn

from chgnet.graph.crystalgraph import CrystalGraph
from chgnet.graph.graph import Graph, Node

if TYPE_CHECKING:
    from pymatgen.core import Structure

datatype = torch.float32


class CrystalGraphConverter(nn.Module):
    """Convert a pymatgen.core.Structure to a CrystalGraph.

    Only the minimal essential information is kept.
    """

    def __init__(
        self, atom_graph_cutoff: float = 5, bond_graph_cutoff: float = 3
    ) -> None:
        """Initialize the Crystal Graph Converter.

        Args:
            atom_graph_cutoff (float): cutoff radius to search for neighboring atom in
                atom_graph. Default = 5
            bond_graph_cutoff (float): bond length threshold to include bond in bond_graph
                Default = 3
            verbose (bool): whether to print initialization message
                Default = True
        """
        super().__init__()
        self.atom_graph_cutoff = atom_graph_cutoff
        if bond_graph_cutoff is None:
            self.bond_graph_cutoff = atom_graph_cutoff
        else:
            self.bond_graph_cutoff = bond_graph_cutoff

    def forward(
        self,
        structure: Structure,
        graph_id=None,
        mp_id=None,
        on_isolated_atoms: Literal["ignore", "warn", "error"] = "error",
    ) -> CrystalGraph:
        """Convert a structure, return a CrystalGraph.

        Args:
            structure (pymatgen.core.Structure): structure to convert
            graph_id (str): an id to keep track of this crystal graph
                Default = None
            mp_id (str): Materials Project id of this structure
                Default = None
            on_isolated_atoms ('ignore' | 'warn' | 'error'): how to handle Structures
                with isolated atoms. Default = 'error'

        Return:
            Crystal_Graph that is ready to use by CHGNet
        """
        n_atoms = len(structure)
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
        graph = Graph([Node(index=i) for i in range(n_atoms)])
        for ii, jj, img, dist in zip(center_index, neighbor_index, image, distance):
            graph.add_edge(center_index=ii, neighbor_index=jj, image=img, distance=dist)

        # Atom Graph
        atom_graph, directed2undirected = graph.adjacency_list()
        atom_graph = torch.tensor(atom_graph, dtype=torch.int64)
        directed2undirected = torch.tensor(directed2undirected, dtype=torch.int64)

        # Bond Graph
        try:
            bond_graph, undirected2directed = graph.line_graph_adjacency_list(
                cutoff=self.bond_graph_cutoff
            )
        except Exception as exc:
            # Report structures that failed creating bond graph
            # This happen occasionally with pymatgen version issue
            structure.to(filename="bond_graph_error.cif")
            raise SystemExit(
                f"Failed creating bond graph for {graph_id}, check bond_graph_error.cif"
            ) from exc
        bond_graph = torch.tensor(bond_graph, dtype=torch.int64)
        undirected2directed = torch.tensor(undirected2directed, dtype=torch.int64)

        # Check if graph has isolated atom
        has_isolated_atom = not set(range(n_atoms)).issubset(center_index)
        if has_isolated_atom:
            r_cutoff = self.atom_graph_cutoff
            msg = f"{graph_id=} has isolated atom with {r_cutoff=}, should be skipped"
            if on_isolated_atoms == "ignore":
                return None
            if on_isolated_atoms == "warn":
                print(msg, file=sys.stderr)
                return None
            # Discard this structure if it has isolated atom in the graph
            raise ValueError(msg)

        return CrystalGraph(
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

    def get_neighbors(
        self, structure: Structure
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Get neighbor information from pymatgen utility function.

        Args:
            structure(pymatgen.core.Structure): a structure to compute

        Returns:
            center_index, neighbor_index, image, distance
        """
        center_index, neighbor_index, image, distance = structure.get_neighbor_list(
            r=self.atom_graph_cutoff, sites=structure.sites, numerical_tol=1e-8
        )
        return center_index, neighbor_index, image, distance

    def as_dict(self) -> dict[str, float]:
        """Save the args of the graph converter."""
        return {
            "atom_graph_cutoff": self.atom_graph_cutoff,
            "bond_graph_cutoff": self.bond_graph_cutoff,
        }

    @classmethod
    def from_dict(cls, dict) -> CrystalGraphConverter:
        """Create converter from dictionary."""
        return CrystalGraphConverter(**dict)
