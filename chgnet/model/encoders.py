from __future__ import annotations

import torch
from torch import Tensor, nn

from chgnet.model.basis import Fourier, RadialBessel


class AtomEmbedding(nn.Module):
    """Encode an atom by its atomic number using a learnable embedding layer."""

    def __init__(self, atom_feature_dim: int, max_num_elements: int = 94) -> None:
        """Initialize the Atom featurizer.

        Args:
            atom_feature_dim (int): dimension of atomic embedding.
            max_num_elements (int): maximum number of elements in the dataset. Default = 94
        """
        super().__init__()
        self.embedding = nn.Embedding(max_num_elements, atom_feature_dim)

    def forward(self, atomic_numbers: Tensor) -> Tensor:
        """Convert the structure to a atom embedding tensor.

        Args:
            atomic_numbers (Tensor): [n_atom, 1].

        Returns:
            atom_fea (Tensor): atom embeddings [n_atom, atom_feature_dim].
        """
        return self.embedding(atomic_numbers)


class BondEncoder(nn.Module):
    """Encode a chemical bond given the position of two atoms using Gaussian Distance."""

    def __init__(
        self,
        atom_graph_cutoff: float = 5,
        bond_graph_cutoff: float = 3,
        num_radial: int = 9,
        cutoff_coeff: int = 5,
        learnable: bool = False,
    ) -> None:
        """Initialize the bond encoder.

        Args:
            atom_graph_cutoff (float): The cutoff for constructing AtomGraph default = 5
            bond_graph_cutoff (float): The cutoff for constructing BondGraph default = 3
            num_radial (int): The number of radial component. Default = 9
            cutoff_coeff (int): Strength for graph cutoff smoothness. Default = 5
            learnable(bool): Whether the frequency in rbf expansion is learnable.
                Default = False
        """
        super().__init__()
        self.rbf_expansion_ag = RadialBessel(
            num_radial=num_radial,
            cutoff=atom_graph_cutoff,
            smooth_cutoff=cutoff_coeff,
            learnable=learnable,
        )
        self.rbf_expansion_bg = RadialBessel(
            num_radial=num_radial,
            cutoff=bond_graph_cutoff,
            smooth_cutoff=cutoff_coeff,
            learnable=learnable,
        )

    def forward(
        self,
        center: Tensor,
        neighbor: Tensor,
        undirected2directed: Tensor,
        image: Tensor,
        lattice: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Compute the pairwise distance between 2 3d coordinates.

        Args:
            center (Tensor): 3d cartesian coordinates of center atoms [n_bond, 3]
            neighbor (Tensor): 3d cartesian coordinates of neighbor atoms [n_bond, 3]
            undirected2directed (Tensor): mapping from undirected bond to one of its
                directed bond [n_bond]
            image (Tensor): the periodic image specifying the location of neighboring
                atom [n_bond, 3]
            lattice (Tensor): the lattice of this structure [3, 3]

        Returns:
            bond_basis_ag (Tensor): the bond basis in AtomGraph [n_bond, num_radial]
            bond_basis_ag (Tensor): the bond basis in BondGraph [n_bond, num_radial]
            bond_vectors (Tensor): normalized bond vectors, for tracking the bond
                directions [n_bond, 3]
        """
        neighbor = neighbor + image @ lattice
        bond_vectors = center - neighbor
        bond_lengths = torch.norm(bond_vectors, dim=1)
        # Normalize the bond vectors
        bond_vectors = bond_vectors / bond_lengths[:, None]

        # We create bond features only for undirected bonds
        # atom1 -> atom2 and atom2 -> atom1 should share same bond_basis
        undirected_bond_lengths = torch.index_select(
            bond_lengths, 0, undirected2directed
        )
        bond_basis_ag = self.rbf_expansion_ag(undirected_bond_lengths)
        bond_basis_bg = self.rbf_expansion_bg(undirected_bond_lengths)
        return bond_basis_ag, bond_basis_bg, bond_vectors


class AngleEncoder(nn.Module):
    """Encode an angle given the two bond vectors using Fourier Expansion."""

    def __init__(self, num_angular: int = 9, learnable: bool = True) -> None:
        """Initialize the angle encoder.

        Args:
            num_angular (int): number of angular basis to use. Must be an odd integer.
            learnable (bool): whether to set the frequencies of the Fourier expansion
                as learnable parameters. Default = False
        """
        super().__init__()
        if num_angular % 2 != 1:
            raise ValueError(f"{num_angular=} must be an odd integer")
        circular_harmonics_order = (num_angular - 1) // 2
        self.fourier_expansion = Fourier(
            order=circular_harmonics_order, learnable=learnable
        )

    def forward(self, bond_i: Tensor, bond_j: Tensor) -> Tensor:
        """Compute the angles between normalized vectors.

        Args:
            bond_i (Tensor): normalized left bond vector [n_angle, 3]
            bond_j (Tensor): normalized right bond vector [n_angle, 3]

        Returns:
            angle_fea (Tensor):  expanded cos_ij [n_angle, angle_feature_dim]
        """
        # 1 - 1e-6 for torch.acos stability
        cosine_ij = torch.sum(bond_i * bond_j, dim=1) * (1 - 1e-6)
        angle = torch.acos(cosine_ij)
        return self.fourier_expansion(angle)
