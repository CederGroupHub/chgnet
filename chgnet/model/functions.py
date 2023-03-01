import torch
from torch import Tensor
import torch.nn as nn
from typing import List, Union


def aggregate(data: Tensor, owners: Tensor, average=True, num_owner=None) -> Tensor:
    """
    aggregate rows in data by specifying the owners

    Args:
        data (Tensor): data tensor to aggregate [n_row, feature_dim]
        owners (Tensor): specify the owner of each row [n_row, 1]
        average (bool): if True, average the rows,
                        if False, sum the rows
            Default = True
        num_owner (int, optional): the number of owners, this is needed if the
            max idx of owner is not presented in owners tensor
            Default = None

    Returns:
        output (Tensor): [num_owner, feature_dim]
    """
    bincount = torch.bincount(owners)
    bincount = bincount.where(bincount != 0, bincount.new_ones(1))

    # If there exist 5 owners, but only the first 4 appear in the owners tensor,
    # we would like the output to have shape [5, fea_dim] with the last row = 0
    # So, we need to optionally add the fifth owner as a row with zero in the output
    if (num_owner is not None) and (bincount.shape[0] != num_owner):
        difference = num_owner - bincount.shape[0]
        bincount = torch.cat([bincount, bincount.new_ones(difference)])

    # make sure this operation is done on the same device of data and owners
    output = data.new_zeros([bincount.shape[0], data.shape[1]])
    output = output.index_add_(0, owners, data)
    if average:
        output = (output.T / bincount).T
    return output


class MLP(nn.Module):
    """
    Multi-Layer Perceptron used for non-linear regression
    """

    def __init__(
        self,
        input_dim: int = None,
        output_dim: int = 1,
        hidden_dim: Union[List[int], int] = [64, 64],
        dropout=0,
        activation="silu",
    ):
        """
        Args:
            input_dim (int): the input dimension
            output_dim (int): the output dimension
            hidden_dim (Union[List[int], int]): a list of integers or a single integer representing
                the number of hidden units in each layer of the MLP. Default = [64, 64]
            dropout (float): the dropout rate before each linear layer. Default: 0
            activation (str, optional): The name of the activation function to use in the gated MLP.
                Must be one of "relu", "silu", "tanh", or "gelu". Default: "silu".
        """
        super(MLP, self).__init__()
        if hidden_dim is None or hidden_dim == 0:
            layers = [nn.Dropout(dropout), nn.Linear(input_dim, output_dim)]
        elif type(hidden_dim) == int:
            layers = [
                nn.Linear(input_dim, hidden_dim),
                find_activation(activation),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, output_dim),
            ]
        else:  # type(hidden_dim) == list:
            layers = [nn.Linear(input_dim, hidden_dim[0]), find_activation(activation)]
            if len(hidden_dim) != 1:
                for h_in, h_out in zip(hidden_dim[0:-1], hidden_dim[1:]):
                    layers.append(nn.Linear(h_in, h_out))
                    layers.append(find_activation(activation))
            layers.append(nn.Dropout(dropout))
            layers.append(nn.Linear(hidden_dim[-1], output_dim))
        self.layers = nn.Sequential(*layers)

    def forward(self, X: Tensor) -> Tensor:
        """
        Performs a forward pass through the MLP.

        Args:
            x (Tensor): a tensor of shape (batch_size, input_dim)

        Returns:
            Tensor: a tensor of shape (batch_size, output_dim)
        """
        return self.layers(X)


class GatedMLP(nn.Module):
    """
    Gated MLP
    similar model structure is used in CGCNN and M3GNet
    """

    def __init__(
        self,
        input_dim: int = None,
        output_dim: int = None,
        hidden_dim: Union[int, List[int]] = None,
        dropout=0,
        activation="silu",
        norm="batch",
    ):
        """
        Args:
            input_dim (int): the input dimension
            output_dim (int): the output dimension
            hidden_dim (Union[List[int], int]): a list of integers or a single integer representing
                the number of hidden units in each layer of the MLP. Default = None
            dropout (float): the dropout rate before each linear layer. Default: 0
            activation (str, optional): The name of the activation function to use in the gated MLP.
                Must be one of "relu", "silu", "tanh", or "gelu". Default: "silu".
            norm (str, optional): The name of the normalization layer to use on the updated atom features.
                Must be one of "batch", "layer", or None. Default: "batch".
        """
        super().__init__()
        self.mlp_core = MLP(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
            activation=activation,
        )
        self.mlp_gate = MLP(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
            activation=activation,
        )
        self.activation = find_activation(activation)
        self.sigmoid = nn.Sigmoid()
        self.norm = norm
        self.bn1 = find_normalization(name=norm, dim=output_dim)
        self.bn2 = find_normalization(name=norm, dim=output_dim)

    def forward(self, X: Tensor) -> Tensor:
        """
        Performs a forward pass through the MLP.

        Args:
            X (Tensor): a tensor of shape (batch_size, input_dim)

        Returns:
            Tensor: a tensor of shape (batch_size, output_dim)
        """
        if self.norm is None:
            core = self.activation(self.mlp_core(X))
            gate = self.sigmoid(self.mlp_gate(X))
        else:
            core = self.activation(self.bn1(self.mlp_core(X)))
            gate = self.sigmoid(self.bn2(self.mlp_gate(X)))
        return core * gate


class ScaledSiLU(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.scale_factor = 1 / 0.6
        self._activation = torch.nn.SiLU()

    def forward(self, x):
        return self._activation(x) * self.scale_factor


def find_activation(name: str):
    """
    Return an activation function using name
    """
    if name in ["relu"]:
        return nn.ReLU()
    elif name in ["silu", "SILU"]:
        return nn.SiLU()
    elif name in ["scaledSILU", "ScaledSILU"]:
        return ScaledSiLU()
    elif name in ["gelu", "GELU"]:
        return nn.GELU
    elif name in ["softplus"]:
        return nn.Softplus()
    elif name in ["sigmoid"]:
        return nn.Sigmoid()
    elif name in ["tanh"]:
        return nn.Tanh()
    else:
        raise NotImplementedError


def find_normalization(name: str, dim: int = None):
    """
    Return an normalization fuction using name
    """
    if name == "batch":
        return nn.BatchNorm1d(dim)
    elif name == "layer":
        return nn.LayerNorm(dim)
    else:
        return None
