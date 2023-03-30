from __future__ import annotations

import pytest
import torch

from chgnet.model.basis import Fourier, RadialBessel
from chgnet.model.encoders import AngleEncoder, AtomEmbedding, BondEncoder

center = torch.tensor([[1, 0, 0], [0, 0, 0]])
neighbor = torch.tensor([[0, 0, 0], [1, 0, 0]])

bond_i = torch.tensor([[1, 0, 0], [0, 1, 0]])
bond_j = torch.tensor([[0, 1, 0], [1, 0, 0]])


@pytest.mark.parametrize("atom_feature_dim", [16, 32, 64])
@pytest.mark.parametrize("max_num_elements", [94, 89])
def test_atom_embedding(atom_feature_dim: int, max_num_elements) -> None:
    atom_embedding = AtomEmbedding(atom_feature_dim, max_num_elements=max_num_elements)
    atomic_numbers = torch.tensor([6, 7, 8])

    result = atom_embedding(atomic_numbers)
    assert result.shape == (3, atom_feature_dim)

    with pytest.raises(IndexError) as exc_info:
        atom_embedding(torch.tensor(100))

    assert "index out of range" in str(exc_info.value)


@pytest.mark.parametrize("atom_graph_cutoff, bond_graph_cutoff", [(5, 3), (6, 4)])
def test_bond_encoder(atom_graph_cutoff, bond_graph_cutoff) -> None:
    undirected2directed = torch.tensor([0, 1])
    image = torch.zeros((2, 3))
    lattice = torch.eye(3)

    bond_encoder = BondEncoder(
        atom_graph_cutoff=atom_graph_cutoff, bond_graph_cutoff=bond_graph_cutoff
    )
    bond_basis_ag, bond_basis_bg, bond_vectors = bond_encoder(
        center, neighbor, undirected2directed, image, lattice
    )

    assert bond_basis_ag.shape == (2, bond_encoder.rbf_expansion_ag.num_radial)
    assert bond_basis_bg.shape == (2, bond_encoder.rbf_expansion_bg.num_radial)
    assert bond_vectors.shape == (2, 3)


@pytest.mark.parametrize("num_angular", [9, 21])
@pytest.mark.parametrize("learnable", [True, False])
def test_angle_encoder(num_angular: int, learnable: bool) -> None:
    angle_encoder = AngleEncoder(num_angular=num_angular, learnable=learnable)
    result = angle_encoder(bond_i, bond_j)

    assert result.shape == (2, num_angular)
    assert isinstance(angle_encoder.fourier_expansion, Fourier)


@pytest.mark.parametrize("num_angular", [-2, 8])
def test_angle_encoder_num_angular(num_angular: int) -> None:
    with pytest.raises(ValueError) as exc_info:
        AngleEncoder(num_angular=num_angular)

    assert f"{num_angular=} must be an odd integer" in str(exc_info.value)


@pytest.mark.parametrize("learnable", [True, False])
def test_bond_encoder_learnable(learnable: bool) -> None:
    undirected2directed = torch.tensor([0, 1])
    image = torch.zeros((2, 3))
    lattice = torch.eye(3)

    bond_encoder = BondEncoder(learnable=learnable)
    bond_basis_ag, bond_basis_bg, bond_vectors = bond_encoder(
        center, neighbor, undirected2directed, image, lattice
    )

    assert isinstance(bond_encoder.rbf_expansion_ag, RadialBessel)
    assert isinstance(bond_encoder.rbf_expansion_bg, RadialBessel)
    assert bond_basis_ag.shape == (2, bond_encoder.rbf_expansion_ag.num_radial)
    assert bond_basis_bg.shape == (2, bond_encoder.rbf_expansion_bg.num_radial)
    assert bond_vectors.shape == (2, 3)


def test_bond_encoder_zero_bond_length() -> None:
    center = torch.tensor([[0, 0, 0]])
    neighbor = torch.tensor([[0, 0, 0]])
    undirected2directed = torch.tensor([0])
    image = torch.zeros((1, 3))
    lattice = torch.eye(3)

    bond_encoder = BondEncoder()
    bond_basis_ag, bond_basis_bg, bond_vectors = bond_encoder(
        center, neighbor, undirected2directed, image, lattice
    )

    for tensor in (bond_basis_ag, bond_basis_bg, bond_vectors):
        assert tensor.isnan().all()
