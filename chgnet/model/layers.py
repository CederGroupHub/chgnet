import torch
import torch.nn as nn
from torch import Tensor
from chgnet.model.functions import find_activation, find_normalization
from chgnet.model.functions import aggregate, MLP, GatedMLP
from typing import List, Tuple


class AtomConv(nn.Module):
    """
    Convolution Layer to update atom_feas
    """

    def __init__(
        self,
        atom_fea_dim: int,
        bond_fea_dim: int,
        hidden_dim: int = 0,
        dropout: float = 0,
        activation: str = "silu",
        norm: str = None,
        use_mlp_out: bool = True,
        resnet: bool = True,
        **kwargs
    ):
        super().__init__()
        self.use_mlp_out = use_mlp_out
        self.resnet = resnet
        self.activation = find_activation(activation)
        self.twoBody_atom = GatedMLP(
            input_dim=2 * atom_fea_dim + bond_fea_dim,
            output_dim=atom_fea_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
            norm=kwargs.pop("gMLP_norm", "batch"),
            activation=activation,
        )
        if self.use_mlp_out:
            self.mlp_out = MLP(
                input_dim=atom_fea_dim, output_dim=atom_fea_dim, hidden_dim=0
            )
        self.atom_norm = find_normalization(name=norm, dim=atom_fea_dim)

    def forward(
        self,
        atom_feas: Tensor,
        bond_feas: Tensor,
        bond_weights: Tensor,
        atom_graph: Tensor,
        directed2undirected: Tensor,
    ) -> (Tensor, Tensor):
        """
        Update the atom_feas, and optionally bond_feas
        Note: num_batch_atoms = sum(num_atoms) in batch
        Args:
            atom_feas (Tensor): [num_batch_atoms, atom_fea_dim]
            bond_feas (Tensor): [num_undirected_bonds, bond_fea_dim]
            bond_weights (Tensor): [num_undirected_bonds, bond_fea_dim]
            atom_graph (Tensor) : [num_directed_bonds, 2]
        Returns:
            atom_feas (Tensor): [num_batch_atom, atom_fea_dim]
            bond_feas (Tensor): [num_undirected_bonds, bond_fea_dim]
        """
        # Make directional messages
        center_atoms = torch.index_select(atom_feas, 0, atom_graph[:, 0])
        nbr_atoms = torch.index_select(atom_feas, 0, atom_graph[:, 1])
        bonds = torch.index_select(bond_feas, 0, directed2undirected)
        messages = torch.cat([center_atoms, bonds, nbr_atoms], dim=1)
        messages = self.twoBody_atom(messages)

        # smooth out message by bond_weights
        bond_weight = torch.index_select(bond_weights, 0, directed2undirected)
        messages = messages * bond_weight

        # Aggregate messages
        new_atom_feas = aggregate(messages, atom_graph[:, 0], average=False)

        # New atom features
        if self.use_mlp_out:
            new_atom_feas = self.mlp_out(new_atom_feas)
        if self.resnet:
            new_atom_feas += atom_feas

        # Optionally, normalize new atom features
        if self.atom_norm is not None:
            new_atom_feas = self.atom_norm(new_atom_feas)
        return new_atom_feas


class BondConv(nn.Module):
    """
    Convolution Layer to update bond_feas
    """

    def __init__(
        self,
        atom_fea_dim: int,
        bond_fea_dim: int,
        angle_fea_dim: int,
        hidden_dim: int = 0,
        dropout: float = 0,
        activation: str = "silu",
        norm: str = None,
        use_mlp_out: bool = True,
        resnet=True,
        **kwargs
    ):
        super().__init__()
        self.use_mlp_out = use_mlp_out
        self.resnet = resnet
        self.activation = find_activation(activation)
        self.twoBody_bond = GatedMLP(
            input_dim=atom_fea_dim + 2 * bond_fea_dim + angle_fea_dim,
            output_dim=bond_fea_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
            norm=kwargs.pop("gMLP_norm", "batch"),
            activation=activation,
        )
        if self.use_mlp_out:
            self.mlp_out = MLP(
                input_dim=bond_fea_dim, output_dim=bond_fea_dim, hidden_dim=0
            )
        self.bond_norm = find_normalization(name=norm, dim=bond_fea_dim)

    def forward(
        self,
        atom_feas: Tensor,
        bond_feas: Tensor,
        bond_weights: Tensor,
        angle_feas: Tensor,
        bond_graph: Tensor,
    ) -> (Tensor, Tensor):
        """
        Update the bond_feas, and optionally angle_feas
        Note: num_batch_bonds = sum(num_bonds) in batch
        Args:
            atom_feas (Tensor): [num_batch_atoms, atom_fea_dim]
            bond_feas (Tensor): [num_undirected_bonds, bond_fea_dim]
            bond_weights (Tensor): [num_undirected_bonds, bond_fea_dim]
            angle_feas (Tensor): [num_batch_angles, atom_fea_dim]
            bond_graph (Tensor) : [num_batch_angles, 5]
        Returns:
            bond_feas (Tensor): [num_batch_atom, atom_fea_dim]
        """
        # Make directional Message
        center_atoms = torch.index_select(atom_feas, 0, bond_graph[:, 0])
        bond_feas_i = torch.index_select(bond_feas, 0, bond_graph[:, 1])
        bond_feas_j = torch.index_select(bond_feas, 0, bond_graph[:, 2])
        total_fea = torch.cat(
            [bond_feas_i, bond_feas_j, angle_feas, center_atoms], dim=1
        )
        bond_update = self.twoBody_bond(total_fea)

        # Smooth out messages
        bond_weights_i = torch.index_select(bond_weights, 0, bond_graph[:, 1])
        bond_weights_j = torch.index_select(bond_weights, 0, bond_graph[:, 2])
        bond_update = bond_update * bond_weights_i * bond_weights_j

        # Aggregate messages
        new_bond_feas = aggregate(
            bond_update, bond_graph[:, 1], average=False, num_owner=len(bond_feas)
        )

        # New bond features
        if self.use_mlp_out:
            new_bond_feas = self.mlp_out(new_bond_feas)
        if self.resnet:
            new_bond_feas += bond_feas

        # Optionally, normalize the bond features
        if self.bond_norm is not None:
            new_bond_feas = self.bond_norm(new_bond_feas)
        return new_bond_feas


class AngleUpdate(nn.Module):
    """
    update angle_feas
    """

    def __init__(
        self,
        atom_fea_dim: int,
        bond_fea_dim: int,
        angle_fea_dim: int,
        hidden_dim: int = 0,
        dropout: float = 0,
        activation: str = "silu",
        norm: str = None,
        resnet: bool = True,
        **kwargs
    ):
        super().__init__()
        self.resnet = resnet
        self.activation = find_activation(activation)
        self.twoBody_bond = GatedMLP(
            input_dim=atom_fea_dim + 2 * bond_fea_dim + angle_fea_dim,
            output_dim=angle_fea_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
            norm=kwargs.pop("gMLP_norm", "batch"),
            activation=activation,
        )
        self.angle_norm = find_normalization(norm, dim=angle_fea_dim)

    def forward(
        self,
        atom_feas: Tensor,
        bond_feas: Tensor,
        angle_feas: Tensor,
        bond_graph: Tensor,
    ) -> (Tensor, Tensor):
        """
        Update the angle_feas
        Note: num_batch_bonds = sum(num_bonds) in batch
        Args:
            atom_feas (Tensor): [num_batch_atoms, atom_fea_dim]
            bond_feas (Tensor): [num_undirected_bonds, bond_fea_dim]
            angle_feas (Tensor): [num_batch_angles, atom_fea_dim]
            bond_graph (Tensor) : [num_batch_angles, 5]
        Returns:
            bond_feas (Tensor): [num_batch_atom, atom_fea_dim]
        """
        # Assemble features
        center_atoms = torch.index_select(atom_feas, 0, bond_graph[:, 0])
        bond_feas_i = torch.index_select(bond_feas, 0, bond_graph[:, 1])
        bond_feas_j = torch.index_select(bond_feas, 0, bond_graph[:, 2])
        total_fea = torch.cat(
            [bond_feas_i, bond_feas_j, angle_feas, center_atoms], dim=1
        )

        # Update angle features
        new_angle_feas = self.twoBody_bond(total_fea)

        # Resnet and Norm
        if self.resnet:
            new_angle_feas += angle_feas
        if self.angle_norm is not None:
            new_angle_feas = self.angle_norm(new_angle_feas)
        return new_angle_feas


class GraphPooling(nn.Module):
    """
    Pooling the sub-graphs in the batched graph
    """

    def __init__(self, average: bool = False):
        """
        Args:
            average (bool): whether to average the features
        """
        super().__init__()
        self.average = average

    def forward(self, atom_feas: Tensor, atom_owner: Tensor) -> Tensor:
        """
        Merge the atom features that belong to same graph in a batched graph
        Args:
            atom_feas (Tensor): batched atom features after convolution layers
                                [num_batch_atoms, atom_fea_dim]
            atom_owner (Tensor): graph indices for each atom [num_batch_atoms]
        Returns:
            crystal_feas (Tensor): crystal feature matrix
                                   [n_crystals, atom_fea_dim]
        """
        return aggregate(atom_feas, atom_owner, average=self.average)


class GraphAttentionReadOut(nn.Module):
    """
    Multi Head Attention Read Out Layer
    merge the information from atom_feas to crystal_fea
    """

    def __init__(
        self, atom_fea_dim: int, num_head: int = 3, hidden_dim: int = 32, average=False
    ):
        """
        Initialize the layer
        Args:
            atom_fea_dim (int): atom feature dimension
            num_head (int): number of attention heads used
            average (bool): whether to average the features
        """
        super().__init__()
        self.key = MLP(
            input_dim=atom_fea_dim, output_dim=num_head, hidden_dim=hidden_dim
        )
        self.softmax = nn.Softmax(dim=0)
        self.average = average

    def forward(self, atom_feas: Tensor, atom_owner: Tensor) -> Tensor:
        """
        Merge the atom features that belong to same graph in a batched graph
        Args:
            atom_feas (Tensor): batched atom features after convolution layers
                                [num_batch_atoms, atom_fea_dim]
            atom_owner (Tensor): graph indices for each atom [num_batch_atoms]
        Returns:
            crystal_feas (Tensor): crystal feature matrix
                                   [n_crystals, atom_fea_dim]
        """
        crystal_feas = []
        weights = self.key(atom_feas)  # [n_batch_atom, n_heads]
        bincount = torch.bincount(atom_owner)
        start_index = 0
        for n_atom in bincount:
            # find atoms belong to this crystal
            atom_fea = atom_feas[
                start_index : start_index + n_atom, :
            ]  # [n_atom, atom_fea_dim]

            # find weight belong to these atoms
            weight = self.softmax(
                weights[start_index : start_index + n_atom, :]
            )  # [n_atom, n_heads]

            # Weighted summation from multiple attention heads
            crystal_fea = (atom_fea.T @ weight).view(-1)  # [n_heads * atom_fea_dim]

            # Normalize the crystal feature if the model output is intensive
            if self.average:
                crystal_fea = crystal_fea / n_atom

            crystal_feas.append(crystal_fea)
            start_index += n_atom
        return torch.stack(crystal_feas, dim=0)


class ElementalReadOut(nn.Module):
    """
    Read out defined by type of element:
        assign different layer to different types of elements
    """

    def __init__(
        self,
        atom_fea_dim: int,
        hidden_dim: int = 32,
        max_num_elements: int = 94,
        average=False,
    ):
        """
        Initialize the layer
        Args:
            atom_fea_dim (int): atom feature dimension
            average (bool): whether to average the features
        """
        super().__init__()
        self.linear = nn.Linear(in_features=max_num_elements, out_features=hidden_dim)
        self.average = average

    def forward(
        self, atom_feas: Tensor, atomic_numbers: Tensor, atom_owner: Tensor
    ) -> Tensor:
        """
        Merge the atom features that belong to same graph in a batched graph
        Args:
            atom_feas (Tensor): batched atom features after convolution layers
                                [num_batch_atoms, atom_fea_dim]
            atom_owner (Tensor): graph indices for each atom [num_batch_atoms]
        Returns:
            crystal_feas (Tensor): crystal feature matrix
                                   [n_crystals, atom_fea_dim]
        """
        crystal_feas = []
        weights = self.key(atom_feas)  # [n_batch_atom, n_heads]
        bincount = torch.bincount(atom_owner)
        start_index = 0
        for n_atom in bincount:
            # find atoms belong to this crystal
            atom_fea = atom_feas[
                start_index : start_index + n_atom, :
            ]  # [n_atom, atom_fea_dim]

            # find weight belong to these atoms
            weight = self.softmax(
                weights[start_index : start_index + n_atom, :]
            )  # [n_atom, n_heads]

            # Weighted summation from multiple attention heads
            crystal_fea = (atom_fea.T @ weight).view(-1)  # [n_heads * atom_fea_dim]

            # Normalize the crystal feature if the model output is intensive
            if self.average:
                crystal_fea = crystal_fea / n_atom

            crystal_feas.append(crystal_fea)
            start_index += n_atom
        return torch.stack(crystal_feas, dim=0)
