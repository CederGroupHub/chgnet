from __future__ import annotations

import collections
from typing import TYPE_CHECKING

import numpy as np
import torch
from pymatgen.core import Structure
from torch import Tensor, nn

from chgnet.model.functions import GatedMLP, find_activation

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    from chgnet.graph.crystalgraph import CrystalGraph


class CompositionModel(nn.Module):
    """A simple FC model that takes in a chemical composition (no structure info)
    and outputs energy.
    """

    def __init__(
        self,
        *,
        atom_fea_dim: int = 64,
        activation: str = "silu",
        is_intensive: bool = True,
        max_num_elements: int = 94,
    ) -> None:
        """Initialize a CompositionModel."""
        super().__init__()
        self.is_intensive = is_intensive
        self.max_num_elements = max_num_elements
        self.fc1 = nn.Linear(max_num_elements, atom_fea_dim)
        self.activation = find_activation(activation)
        self.gated_mlp = GatedMLP(
            input_dim=atom_fea_dim,
            output_dim=atom_fea_dim,
            hidden_dim=atom_fea_dim,
            activation=activation,
        )
        self.fc2 = nn.Linear(atom_fea_dim, 1)

    def _get_energy(self, composition_feas: Tensor) -> Tensor:
        """Predict the energy given composition encoding.

        Args:
            composition_feas: batched atom feature matrix of shape
                [batch_size, total_num_elements].

        Returns:
            prediction associated with each composition [batchsize].
        """
        composition_feas = self.activation(self.fc1(composition_feas))
        composition_feas += self.gated_mlp(composition_feas)
        return self.fc2(composition_feas).view(-1)

    def forward(self, graphs: list[CrystalGraph]) -> Tensor:
        """Get the energy of a list of CrystalGraphs as Tensor."""
        composition_feas = self._assemble_graphs(graphs)
        return self._get_energy(composition_feas)

    def _assemble_graphs(self, graphs: list[CrystalGraph]) -> Tensor:
        """Assemble a list of graphs into one-hot composition encodings.

        Args:
            graphs (list[CrystalGraph]): a list of CrystalGraphs

        Returns:
            assembled batch_graph that contains all information for model.
        """
        composition_feas = []
        for graph in graphs:
            composition_fea = torch.bincount(
                graph.atomic_number - 1, minlength=self.max_num_elements
            )
            if self.is_intensive:
                n_atom = graph.atomic_number.shape[0]
                composition_fea = composition_fea / n_atom
            composition_feas.append(composition_fea)
        return torch.stack(composition_feas, dim=0)


class AtomRef(nn.Module):
    """A linear regression for elemental energy.
    From: https://github.com/materialsvirtuallab/m3gnet/.
    """

    def __init__(
        self, *, is_intensive: bool = True, max_num_elements: int = 94
    ) -> None:
        """Initialize an AtomRef model."""
        super().__init__()
        self.is_intensive = is_intensive
        self.max_num_elements = max_num_elements
        self.fc = nn.Linear(max_num_elements, 1, bias=False)
        self.fitted = False

    def forward(self, graphs: list[CrystalGraph]) -> Tensor:
        """Get the energy of a list of CrystalGraphs.

        Args:
            graphs (List(CrystalGraph)): a list of Crystal Graph to compute

        Returns:
            energy (tensor)
        """
        if not self.fitted:
            raise ValueError("composition model needs to be fitted first!")
        composition_feas = self._assemble_graphs(graphs)
        return self._get_energy(composition_feas)

    def _get_energy(self, composition_feas: Tensor) -> Tensor:
        """Predict the energy given composition encoding.

        Args:
            composition_feas: batched atom feature matrix of shape
                [batch_size, total_num_elements].

        Returns:
            prediction associated with each composition [batchsize].
        """
        return self.fc(composition_feas).view(-1)

    def fit(
        self,
        structures_or_graphs: Sequence[Structure | CrystalGraph],
        energies: Sequence[float],
    ) -> None:
        """Fit the model to a list of crystals and energies.

        Args:
            structures_or_graphs (list[Structure  |  CrystalGraph]): Any iterable of
                pymatgen structures and/or graphs.
            energies (list[float]): Target energies.
        """
        num_data = len(energies)
        composition_feas = torch.zeros([num_data, self.max_num_elements])
        e = torch.zeros([num_data])
        for index, (structure, energy) in enumerate(
            zip(structures_or_graphs, energies, strict=True)
        ):
            if isinstance(structure, Structure):
                atomic_number = torch.tensor(
                    [site.specie.Z for site in structure],
                    dtype=torch.int32,
                    requires_grad=False,
                )
            else:
                atomic_number = structure.atomic_number
            composition_fea = torch.bincount(
                atomic_number - 1, minlength=self.max_num_elements
            )
            if self.is_intensive:
                composition_fea = composition_fea / atomic_number.shape[0]
            composition_feas[index, :] = composition_fea
            e[index] = energy

        # Use numpy for pinv
        self.feature_matrix = composition_feas.detach().numpy()
        self.energies = e.detach().numpy()
        state_dict = collections.OrderedDict()
        weight = (
            np.linalg.pinv(self.feature_matrix.T @ self.feature_matrix)
            @ self.feature_matrix.T
            @ self.energies
        )
        state_dict["weight"] = torch.tensor(weight).view(1, 94)
        self.fc.load_state_dict(state_dict)
        self.fitted = True

    def _assemble_graphs(self, graphs: list[CrystalGraph]) -> Tensor:
        """Assemble a list of graphs into one-hot composition encodings
        Args:
            graphs (list[Tensor]): a list of CrystalGraphs
        Returns:
            assembled batch_graph that contains all information for model.
        """
        composition_feas = []
        for graph in graphs:
            composition_fea = torch.bincount(
                graph.atomic_number - 1, minlength=self.max_num_elements
            )
            if self.is_intensive:
                n_atom = graph.atomic_number.shape[0]
                composition_fea = composition_fea / n_atom
            composition_feas.append(composition_fea)
        return torch.stack(composition_feas, dim=0).float()

    def get_site_energies(self, graphs: list[CrystalGraph]) -> list[Tensor]:
        """Predict the site energies given a list of CrystalGraphs.

        Args:
            graphs (List(CrystalGraph)): a list of Crystal Graph to compute

        Returns:
            a list of tensors corresponding to site energies of each graph [batchsize].
        """
        return [
            self.fc.state_dict()["weight"][0, graph.atomic_number - 1]
            for graph in graphs
        ]

    def initialize_from(self, dataset: str) -> None:
        """Initialize pre-fitted weights from a dataset."""
        if dataset in {"MPtrj", "MPtrj_e"}:
            self.initialize_from_MPtrj()
        elif dataset == "MPF":
            self.initialize_from_MPF()
        elif dataset == "MP-r2SCAN":
            self.initialize_from_MP_r2SCAN()
        else:
            raise NotImplementedError(f"{dataset=} not supported yet")

    def initialize_from_MPtrj(self) -> None:  # noqa: N802
        """Initialize pre-fitted weights from MPtrj dataset."""
        state_dict = collections.OrderedDict()
        state_dict["weight"] = torch.tensor(
            [
                -3.4431,
                -0.1279,
                -2.8300,
                -3.4737,
                -7.4946,
                -8.2354,
                -8.1611,
                -8.3861,
                -5.7498,
                -0.0236,
                -1.7406,
                -1.6788,
                -4.2833,
                -6.2002,
                -6.1315,
                -5.8405,
                -3.8795,
                -0.0703,
                -1.5668,
                -3.4451,
                -7.0549,
                -9.1465,
                -9.2594,
                -9.3514,
                -8.9843,
                -8.0228,
                -6.4955,
                -5.6057,
                -3.4002,
                -0.9217,
                -3.2499,
                -4.9164,
                -4.7810,
                -5.0191,
                -3.3316,
                0.5130,
                -1.4043,
                -3.2175,
                -7.4994,
                -9.3816,
                -10.4386,
                -9.9539,
                -7.9555,
                -8.5440,
                -7.3245,
                -5.2771,
                -1.9014,
                -0.4034,
                -2.6002,
                -4.0054,
                -4.1156,
                -3.9928,
                -2.7003,
                2.2170,
                -1.9671,
                -3.7180,
                -6.8133,
                -7.3502,
                -6.0712,
                -6.1699,
                -5.1471,
                -6.1925,
                -11.5829,
                -15.8841,
                -5.9994,
                -6.0798,
                -5.9513,
                -6.0400,
                -5.9773,
                -2.5091,
                -6.0767,
                -10.6666,
                -11.8761,
                -11.8491,
                -10.7397,
                -9.6100,
                -8.4755,
                -6.2070,
                -3.0337,
                0.4726,
                -1.6425,
                -3.1295,
                -3.3328,
                -0.1221,
                -0.3448,
                -0.4364,
                -0.1661,
                -0.3680,
                -4.1869,
                -8.4233,
                -10.0467,
                -12.0953,
                -12.5228,
                -14.2530,
            ]
        ).view([1, 94])
        self.fc.load_state_dict(state_dict)
        self.is_intensive = True
        self.fitted = True

    def initialize_from_MPF(self) -> None:  # noqa: N802
        """Initialize pre-fitted weights from MPF dataset."""
        state_dict = collections.OrderedDict()
        state_dict["weight"] = torch.tensor(
            [
                -3.4654e00,
                -6.2617e-01,
                -3.4622e00,
                -4.7758e00,
                -8.0362e00,
                -8.4038e00,
                -7.7681e00,
                -7.3892e00,
                -4.9472e00,
                -5.4833e00,
                -2.4783e00,
                -2.0202e00,
                -5.1548e00,
                -7.9121e00,
                -6.9135e00,
                -4.6228e00,
                -3.0155e00,
                -2.1285e00,
                -2.3174e00,
                -4.7595e00,
                -8.1742e00,
                -1.1421e01,
                -8.9229e00,
                -8.4901e00,
                -8.1664e00,
                -6.5826e00,
                -5.2614e00,
                -4.4841e00,
                -3.2737e00,
                -1.3498e00,
                -3.6264e00,
                -4.6727e00,
                -4.1316e00,
                -3.6755e00,
                -2.8030e00,
                6.4728e00,
                -2.2469e00,
                -4.2510e00,
                -1.0245e01,
                -1.1666e01,
                -1.1802e01,
                -8.6551e00,
                -9.3641e00,
                -7.5716e00,
                -5.6990e00,
                -4.9716e00,
                -1.8871e00,
                -6.7951e-01,
                -2.7488e00,
                -3.7945e00,
                -3.3883e00,
                -2.5588e00,
                -1.9621e00,
                9.9793e00,
                -2.5566e00,
                -4.8803e00,
                -8.8604e00,
                -9.0537e00,
                -7.9431e00,
                -8.1259e00,
                -6.3212e00,
                -8.3025e00,
                -1.2289e01,
                -1.7310e01,
                -7.5512e00,
                -8.1959e00,
                -8.3493e00,
                -7.2591e00,
                -8.4170e00,
                -3.3873e00,
                -7.6823e00,
                -1.2630e01,
                -1.3626e01,
                -9.5299e00,
                -1.1840e01,
                -9.7990e00,
                -7.5561e00,
                -5.4690e00,
                -2.6508e00,
                4.1746e-01,
                -2.3255e00,
                -3.4830e00,
                -3.1808e00,
                -1.6934e-02,
                -3.6191e-02,
                -1.0842e-02,
                1.3170e-02,
                -6.5371e-02,
                -5.4892e00,
                -1.0335e01,
                -1.1130e01,
                -1.4312e01,
                -1.4700e01,
                -1.5473e01,
            ]
        ).view([1, 94])
        self.fc.load_state_dict(state_dict)
        self.is_intensive = False
        self.fitted = True

    def initialize_from_MP_r2SCAN(self) -> None:  
        """Initialize pre-fitted weights from MP-r2SCAN dataset."""
        state_dict = collections.OrderedDict()

        state_dict["weight"] = torch.tensor(
        [-3.4690e+00, -3.0982e-01, -3.3199e+00, -4.7963e+00, -8.0507e+00,
         -9.5759e+00, -9.8677e+00, -9.1242e+00, -6.7546e+00, -1.9120e+00,
         -4.5438e+00, -4.0474e+00, -7.2176e+00, -9.6473e+00, -9.6514e+00,
         -9.5449e+00, -7.9040e+00, -4.8555e+00, -7.0955e+00, -8.4121e+00,
         -1.2896e+01, -1.4512e+01, -1.5121e+01, -1.5248e+01, -1.4923e+01,
         -1.4040e+01, -1.2751e+01, -1.1945e+01, -1.0464e+01, -8.9017e+00,
         -1.1722e+01, -1.4170e+01, -1.5067e+01, -1.5418e+01, -1.4794e+01,
         -1.1486e+01, -1.5029e+01, -1.6974e+01, -2.1922e+01, -2.4265e+01,
         -2.5605e+01, -2.6075e+01, -2.5442e+01, -2.5286e+01, -2.4571e+01,
         -2.3376e+01, -2.0786e+01, -2.0013e+01, -2.2626e+01, -2.4799e+01,
         -2.5832e+01, -2.5982e+01, -2.5459e+01, -2.2229e+01, -2.6402e+01,
         -2.8426e+01, -3.1738e+01, -3.2878e+01, -3.0945e+01, -3.0967e+01,
         -2.9942e+01, -3.1421e+01, -4.0080e+01, -4.5251e+01, -3.2790e+01,
         -3.3584e+01, -3.4371e+01, -3.5534e+01, -3.6623e+01,  5.6469e-14,
         -3.9644e+01, -4.6709e+01, -4.9586e+01, -5.1200e+01, -5.1762e+01,
         -5.2404e+01, -5.2657e+01, -5.2166e+01, -5.0671e+01, -4.8918e+01,
         -5.2844e+01, -5.6015e+01, -5.8066e+01,  1.8537e-14, -1.0885e-15,
         -1.0417e-16, -2.1228e-16,  5.6561e-16, -6.9083e+01, -7.4960e+01,
         -7.8234e+01, -8.1985e+01, -8.4724e+01, -8.7538e+01]
        ).view([1, 94])

        self.fc.load_state_dict(state_dict)
        self.is_intensive = False
        self.fitted = True
        

    def initialize_from_numpy(self, file_name: str | Path) -> None:
        """Initialize pre-fitted weights from numpy file."""
        atom_ref_np = np.load(file_name)
        state_dict = collections.OrderedDict()
        state_dict["weight"] = torch.tensor(atom_ref_np).view([1, 94])
        self.fc.load_state_dict(state_dict)
        self.is_intensive = False
        self.fitted = True
