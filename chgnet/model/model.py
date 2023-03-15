from __future__ import annotations

import math
import os
from typing import Literal, Sequence

import torch
from pymatgen.core import Structure
from torch import Tensor, nn

from chgnet import PredTask
from chgnet.graph import CrystalGraph, CrystalGraphConverter, BatchedGraph
from chgnet.model.composition_model import AtomRef
from chgnet.model.encoders import AngleEncoder, AtomEmbedding, BondEncoder
from chgnet.model.functions import MLP, GatedMLP, find_normalization
from chgnet.model.layers import (
    AngleUpdate,
    AtomConv,
    BondConv,
    GraphAttentionReadOut,
    GraphPooling,
)

datatype = torch.float32


class CHGNet(nn.Module):
    """Crystal Hamiltonian Graph neural Network
    A model that takes in a crystal graph and output energy, force, magmom, stress.
    """

    def __init__(
        self,
        atom_fea_dim: int = 64,
        bond_fea_dim: int = 64,
        angle_fea_dim: int = 64,
        composition_model: str | nn.Module = None,
        num_radial: int = 9,
        num_angular: int = 9,
        n_conv: int = 4,
        atom_conv_hidden_dim: Sequence[int] | int = 64,
        update_bond: bool = True,
        bond_conv_hidden_dim: Sequence[int] | int = 64,
        update_angle: bool = True,
        angle_layer_hidden_dim: Sequence[int] | int = 0,
        conv_dropout: float = 0,
        read_out: str = "ave",
        mlp_hidden_dims: Sequence[int] | int = (64, 64),
        mlp_dropout: float = 0,
        mlp_first: bool = True,
        is_intensive: bool = True,
        non_linearity: Literal["silu", "relu", "tanh", "gelu"] = "relu",
        atom_graph_cutoff: int = 5,
        bond_graph_cutoff: int = 3,
        learnable_rbf: bool = True,
        **kwargs,
    ) -> None:
        """Initialize the CHGNet.

        Args:
            atom_fea_dim (int): atom feature vector embedding dimension.
                Default = 64
            bond_fea_dim (int): bond feature vector embedding dimension.
                Default = 64
            angle_fea_dim (int): angle feature vector embedding dimension.
                Default = 64
            bond_fea_dim (int): angle feature vector embedding dimension.
                Default = 64
            composition_model (nn.Module, optional): attach a composition model to
                predict energy or initialize a pretrained linear regression (AtomRef).
                Default = None
            num_radial (int): number of radial basis used in bond basis expansion.
                Default = 9
            num_angular (int): number of angular basis used in angle basis expansion.
                Default = 9
            n_conv (int): number of interaction blocks.
                Default = 4
                Note: last interaction block contain only an atom_conv layer
            atom_conv_hidden_dim (List or int): hidden dimensions of
                atom convolution layers.
                Default = 64
            update_bond (bool): whether to use bond_conv_layer in bond graph to
                update bond embeddings
                Default = True.
            bond_conv_hidden_dim (List or int): hidden dimensions of
                bond convolution layers.
                Default = 64
            update_angle (bool): whether to use angle_update_layer to
                update angle embeddings.
                Default = True
            angle_layer_hidden_dim (List or int): hidden dimensions of angle layers.
                Default = 0
            conv_dropout (float): dropout rate in all conv_layers.
                Default = 0
            read_out (str): method for pooling layer, 'ave' for standard
                average pooling, 'attn' for multi-head attention.
                Default = "ave"
            mlp_hidden_dims (int or list): readout multilayer perceptron
                hidden dimensions.
                Default = [64, 64]
            mlp_dropout (float): dropout rate in readout MLP.
                Default = 0.
            is_intensive (bool): whether the energy training label is intensive
                i.e. energy per atom.
                Default = True
            non_linearity ('silu' | 'relu' | 'tanh' | 'gelu'): The name of the
                activation function to use in the gated MLP.
                Default = "silu".
            mlp_first (bool): whether to apply mlp fist then pooling.
                Default = True
            atom_graph_cutoff (float): cutoff radius (A) in creating atom_graph,
                this need to be consistent with the value in training dataloader
                Default = 5
            bond_graph_cutoff (float): cutoff radius (A) in creating bond_graph,
                this need to be consistent with value in training dataloader
                Default = 3
            cutoff_coeff (float): cutoff strength used in graph smooth cutoff function.
                Default = 5
            learnable_rbf (bool): whether to set the frequencies in rbf and Fourier
                basis functions learnable.
                Default = True
            **kwargs: Additional keyword arguments
        """
        # Store model args for reconstruction
        self.model_args = {
            k: v
            for k, v in locals().items()
            if k not in ["self", "__class__", "kwargs"]
        }
        self.model_args.update(kwargs)

        super().__init__()
        self.atom_fea_dim = atom_fea_dim
        self.bond_fea_dim = bond_fea_dim
        self.is_intensive = is_intensive
        self.n_conv = n_conv

        # Optionally, define composition model
        if isinstance(composition_model, nn.Module):
            self.composition_model = composition_model
        else:
            self.composition_model = AtomRef(is_intensive=is_intensive)
            self.composition_model.initialize_from(composition_model)
        if self.composition_model is not None:
            # fixed composition_model weights
            for param in self.composition_model.parameters():
                param.requires_grad = False

        # Define Crystal Graph Converter
        self.graph_converter = CrystalGraphConverter(
            atom_graph_cutoff=atom_graph_cutoff, bond_graph_cutoff=bond_graph_cutoff
        )

        # Define embedding layers
        self.atom_embedding = AtomEmbedding(atom_feature_dim=atom_fea_dim)
        cutoff_coeff = kwargs.pop("cutoff_coeff", 6)
        self.bond_basis_expansion = BondEncoder(
            atom_graph_cutoff=atom_graph_cutoff,
            bond_graph_cutoff=bond_graph_cutoff,
            num_radial=num_radial,
            cutoff_coeff=cutoff_coeff,
            learnable=learnable_rbf,
        )
        self.bond_embedding = nn.Linear(
            in_features=num_radial, out_features=bond_fea_dim, bias=False
        )
        self.bond_weights_ag = nn.Linear(
            in_features=num_radial, out_features=atom_fea_dim, bias=False
        )
        self.bond_weights_bg = nn.Linear(
            in_features=num_radial, out_features=bond_fea_dim, bias=False
        )
        self.angle_basis_expansion = AngleEncoder(
            num_angular=num_angular, learnable=learnable_rbf
        )
        self.angle_embedding = nn.Linear(
            in_features=num_angular, out_features=angle_fea_dim, bias=False
        )

        # Define convolutional layers
        conv_norm = kwargs.pop("conv_norm", None)
        gMLP_norm = kwargs.pop("gMLP_norm", None)
        atom_graph_layers = []
        for _i in range(n_conv):
            atom_graph_layers.append(
                AtomConv(
                    atom_fea_dim=atom_fea_dim,
                    bond_fea_dim=bond_fea_dim,
                    hidden_dim=atom_conv_hidden_dim,
                    dropout=conv_dropout,
                    activation=non_linearity,
                    norm=conv_norm,
                    gMLP_norm=gMLP_norm,
                    use_mlp_out=True,
                    resnet=True,
                )
            )
        self.atom_conv_layers = nn.ModuleList(atom_graph_layers)

        if update_bond is True:
            bond_graph_layers = [
                BondConv(
                    atom_fea_dim=atom_fea_dim,
                    bond_fea_dim=bond_fea_dim,
                    angle_fea_dim=angle_fea_dim,
                    hidden_dim=bond_conv_hidden_dim,
                    dropout=conv_dropout,
                    activation=non_linearity,
                    norm=conv_norm,
                    gMLP_norm=gMLP_norm,
                    use_mlp_out=True,
                    resnet=True,
                )
                for _ in range(n_conv - 1)
            ]
            self.bond_conv_layers = nn.ModuleList(bond_graph_layers)
        else:
            self.bond_conv_layers = [None for _ in range(n_conv - 1)]

        if update_angle is True:
            angle_layers = [
                AngleUpdate(
                    atom_fea_dim=atom_fea_dim,
                    bond_fea_dim=bond_fea_dim,
                    angle_fea_dim=angle_fea_dim,
                    hidden_dim=angle_layer_hidden_dim,
                    dropout=conv_dropout,
                    activation=non_linearity,
                    norm=conv_norm,
                    gMLP_norm=gMLP_norm,
                    resnet=True,
                )
                for _ in range(n_conv - 1)
            ]
            self.angle_layers = nn.ModuleList(angle_layers)
        else:
            self.angle_layers = [None for _ in range(n_conv - 1)]

        # Define readout layer
        self.site_wise = nn.Linear(atom_fea_dim, 1)
        self.readout_norm = find_normalization(
            name=kwargs.pop("readout_norm", None), dim=atom_fea_dim
        )
        self.mlp_first = mlp_first
        if mlp_first:
            self.read_out_type = "sum"
            input_dim = atom_fea_dim
            self.pooling = GraphPooling(average=False)
        else:
            if read_out in ["attn", "weighted"]:
                self.read_out_type = "attn"
                num_heads = kwargs.pop("num_heads", 3)
                self.pooling = GraphAttentionReadOut(
                    atom_fea_dim, num_head=num_heads, average=True
                )
                input_dim = atom_fea_dim * num_heads
            else:
                self.read_out_type = "ave"
                input_dim = atom_fea_dim
                self.pooling = GraphPooling(average=True)
        if kwargs.pop("final_mlp", "MLP") in ["normal", "MLP"]:
            self.mlp = MLP(
                input_dim=input_dim,
                hidden_dim=mlp_hidden_dims,
                output_dim=1,
                dropout=mlp_dropout,
                activation=non_linearity,
            )
        else:
            self.mlp = nn.Sequential(
                GatedMLP(
                    input_dim=input_dim,
                    hidden_dim=mlp_hidden_dims,
                    output_dim=mlp_hidden_dims[-1],
                    dropout=mlp_dropout,
                    activation=non_linearity,
                ),
                nn.Linear(in_features=mlp_hidden_dims[-1], out_features=1),
            )

        print(
            f"CHGNet initialized with {sum(p.numel() for p in self.parameters()):,} parameters"
        )

    def forward(
        self,
        graphs: Sequence[CrystalGraph],
        task: PredTask = "e",
        return_atom_feas: bool = False,
        return_crystal_feas: bool = False,
    ) -> dict:
        """Get prediction associated with input graphs
        Args:
            graphs (List): a list of Crystal_Graphs
            task (str): the prediction task
                        eg: 'e', 'em', 'ef', 'efs', 'efsm'
                Default = 'e'
            return_atom_feas (bool): whether to return the atom features before last
                conv layer
                Default = False
            return_crystal_feas (bool): whether to return crystal feature
                Default = False
        Returns:
            model output (dict).
        """
        compute_force = "f" in task
        compute_stress = "s" in task
        site_wise = "m" in task

        # Optionally, make composition model prediction
        comp_energy = (
            0 if self.composition_model is None else self.composition_model(graphs)
        )

        # Make batched graph
        batched_graph = BatchedGraph.from_graphs(
            graphs,
            bond_basis_expansion=self.bond_basis_expansion,
            angle_basis_expansion=self.angle_basis_expansion,
            compute_stress=compute_stress,
        )

        # Pass to model
        prediction = self._compute(
            batched_graph,
            site_wise=site_wise,
            compute_force=compute_force,
            compute_stress=compute_stress,
            return_atom_feas=return_atom_feas,
            return_crystal_feas=return_crystal_feas,
        )
        prediction["e"] += comp_energy
        return prediction

    def _compute(
        self,
        g,
        site_wise: bool = False,
        compute_force: bool = False,
        compute_stress: bool = False,
        return_atom_feas: bool = False,
        return_crystal_feas: bool = False,
    ) -> dict:
        """Get Energy, Force, Stress, Magmom associated with input graphs
        force = - d(Energy)/d(atom_positions)
        stress = 1/V * d(Energy)/d(strain).

        Args:
            g (BatchedGraph): batched graph
            site_wise (bool): whether to compute magmom.
                Default = False
            compute_force (bool): whether to compute force.
                Default = False
            compute_stress (bool): whether to compute stress.
                Default = False
            return_atom_feas (bool): whether to return atom features
                Default = False
            return_crystal_feas (bool): whether to return crystal features,
                only available if self.mlp_first is False
                Default = False

        Returns:
            prediction (dict): containing the fields:
                e (Tensor) : energy of structures [batch_size, 1]
                f (Tensor) : force on atoms [num_batch_atoms, 3]
                s (Tensor) : stress of structure [3 * batch_size, 3]
                m (Tensor) : magnetic moments of sites [num_batch_atoms, 3]
        """
        prediction = {}
        atoms_per_graph = torch.bincount(g.atom_owners)
        prediction["atoms_per_graph"] = atoms_per_graph

        # Embed Atoms, Bonds and Angles
        atom_feas = self.atom_embedding(
            g.atomic_numbers - 1
        )  # let H be the first embedding column
        bond_feas = self.bond_embedding(g.bond_bases_ag)
        bond_weights_ag = self.bond_weights_ag(g.bond_bases_ag)
        bond_weights_bg = self.bond_weights_bg(g.bond_bases_bg)
        if len(g.angle_bases) != 0:
            angle_feas = self.angle_embedding(g.angle_bases)

        # Message Passing
        for idx, (atom_layer, bond_layer, angle_layer) in enumerate(
            zip(self.atom_conv_layers[:-1], self.bond_conv_layers, self.angle_layers)
        ):
            # Atom Conv
            atom_feas = atom_layer(
                atom_feas=atom_feas,
                bond_feas=bond_feas,
                bond_weights=bond_weights_ag,
                atom_graph=g.batched_atom_graph,
                directed2undirected=g.directed2undirected,
            )

            # Bond Conv
            if len(g.angle_bases) != 0 and bond_layer is not None:
                bond_feas = bond_layer(
                    atom_feas=atom_feas,
                    bond_feas=bond_feas,
                    bond_weights=bond_weights_bg,
                    angle_feas=angle_feas,
                    bond_graph=g.batched_bond_graph,
                )

                # Angle Update
                if angle_layer is not None:
                    angle_feas = angle_layer(
                        atom_feas=atom_feas,
                        bond_feas=bond_feas,
                        angle_feas=angle_feas,
                        bond_graph=g.batched_bond_graph,
                    )
            if idx == self.n_conv - 2:
                if return_atom_feas is True:
                    prediction["atom_fea"] = torch.split(
                        atom_feas, atoms_per_graph.tolist()
                    )
                # Compute site-wise magnetic moments
                if site_wise:
                    magmom = torch.abs(self.site_wise(atom_feas))
                    prediction["m"] = list(
                        torch.split(magmom.view(-1), atoms_per_graph.tolist())
                    )

        # Last conv layer
        atom_feas = self.atom_conv_layers[-1](
            atom_feas=atom_feas,
            bond_feas=bond_feas,
            bond_weights=bond_weights_ag,
            atom_graph=g.batched_atom_graph,
            directed2undirected=g.directed2undirected,
        )
        if self.readout_norm is not None:
            atom_feas = self.readout_norm(atom_feas)

        # Aggregate nodes and ReadOut
        if self.mlp_first:
            energies = self.mlp(atom_feas)
            energy = self.pooling(energies, g.atom_owners).view(-1)
        else:  # ave or attn to create crystal_fea first
            crystal_feas = self.pooling(atom_feas, g.atom_owners)
            energy = self.mlp(crystal_feas).view(-1) * atoms_per_graph
            if return_crystal_feas is True:
                prediction["crystal_fea"] = crystal_feas

        # Compute force
        if compute_force:
            # Need to retain_graph here, because energy is used in loss function,
            # so its gradient need to be calculated later
            # The graphs of force and stress need to be created for same reason.
            force = torch.autograd.grad(
                energy.sum(), g.atom_positions, create_graph=True, retain_graph=True
            )
            force = [-1 * i for i in force]
            prediction["f"] = force

        # Compute stress
        if compute_stress:
            stress = torch.autograd.grad(
                energy.sum(), g.strains, create_graph=True, retain_graph=True
            )
            # Convert Stress unit from eV/A^3 to GPa
            scale = 1 / g.volumes * 160.21766208
            stress = [i * j for i, j in zip(stress, scale)]
            prediction["s"] = stress

        # Normalize energy if model is intensive
        if self.is_intensive:
            energy = energy / atoms_per_graph
        prediction["e"] = energy

        return prediction

    def predict_structure(
        self,
        structure: Structure | Sequence[Structure],
        task: PredTask = "efsm",
        return_atom_feas: bool = False,
        return_crystal_feas: bool = False,
        batch_size: int = 100,
    ) -> dict[str, Tensor]:
        """Predict from pymatgen.core.Structure.

        Args:
            structure (Structure, List(Structure)): structure or a list of structures
                to predict.
            task (str): can be 'e' 'ef', 'em', 'efs', 'efsm'
                Default = "efsm"
            return_atom_feas (bool): whether to return atom features.
                Default = False
            return_crystal_feas (bool): whether to return crystal features.
                only available if self.mlp_first is False
                Default = False
            batch_size (int): batch_size for predict structures.
                Default = 100

        Returns:
            prediction (dict): containing the fields:
                e (Tensor) : energy of structures [batch_size, 1]
                f (Tensor) : force on atoms [num_batch_atoms, 3]
                s (Tensor) : stress of structure [3 * batch_size, 3]
                m (Tensor) : magnetic moments of sites [num_batch_atoms, 3]
        """
        assert (
            self.graph_converter is not None
        ), "self.graph_converter needs to be initialized first!"
        if type(structure) == Structure:
            graph = self.graph_converter(structure)
            return self.predict_graph(
                graph,
                task=task,
                return_atom_feas=return_atom_feas,
                return_crystal_feas=return_crystal_feas,
                batch_size=batch_size,
            )
        elif type(structure) == list:
            graphs = [self.graph_converter(i) for i in structure]
            return self.predict_graph(
                graphs,
                task=task,
                return_atom_feas=return_atom_feas,
                return_crystal_feas=return_crystal_feas,
                batch_size=batch_size,
            )
        else:
            raise Exception("input should either be a structure or list of structures!")

    def predict_graph(
        self,
        graph: CrystalGraph | Sequence[CrystalGraph],
        task: PredTask = "efsm",
        return_atom_feas: bool = False,
        return_crystal_feas: bool = False,
        batch_size: int = 100,
    ) -> dict[str, Tensor]:
        """Args:
            graph (CrystalGraph): Crystal_Graph or a list of Crystal_Graphs to predict.
            task (str): can be 'e' 'ef', 'em', 'efs', 'efsm'
                Default = "efsm"
            return_atom_feas (bool): whether to return atom features.
                Default = False
            return_crystal_feas (bool): whether to return crystal features.
                only available if self.mlp_first is False
                Default = False
            batch_size (int): batch_size for predict structures.
                Default = 100.

        Returns:
            prediction (dict): containing the fields:
                e (Tensor) : energy of structures [batch_size, 1]
                f (Tensor) : force on atoms [num_batch_atoms, 3]
                s (Tensor) : stress of structure [3 * batch_size, 3]
                m (Tensor) : magnetic moments of sites [num_batch_atoms, 3]
        """
        if type(graph) == CrystalGraph:
            self.eval()
            prediction = self.forward(
                [graph],
                task=task,
                return_atom_feas=return_atom_feas,
                return_crystal_feas=return_crystal_feas,
            )
            out = {}
            for key, pred in prediction.items():
                if key == "e":
                    out[key] = pred.item()
                elif key in ["f", "s", "m", "atom_fea"]:
                    assert len(pred) == 1
                    out[key] = pred[0].cpu().detach().numpy()
                elif key == "crystal_fea":
                    out[key] = pred.view(-1).cpu().detach().numpy()
            return out
        elif type(graph) == list:
            self.eval()
            predictions: list[dict[str, Tensor]] = [{} for _ in range(len(graph))]
            n_steps = math.ceil(len(graph) / batch_size)
            for n in range(n_steps):
                prediction = self.forward(
                    graph[batch_size * n : batch_size * (n + 1)],
                    task=task,
                    return_atom_feas=return_atom_feas,
                    return_crystal_feas=return_crystal_feas,
                )
                for key, pred in prediction.items():
                    if key in ["e"]:
                        for i, e in enumerate(pred.cpu().detach().numpy()):
                            predictions[n * batch_size + i][key] = e
                    elif key in ["f", "s", "m"]:
                        for i, tmp in enumerate(pred):
                            predictions[n * batch_size + i][key] = (
                                tmp.cpu().detach().numpy()
                            )
                    elif key == "atom_fea":
                        for i, atom_fea in enumerate(pred):
                            predictions[n * batch_size + i][key] = (
                                atom_fea.cpu().detach().numpy()
                            )
                    elif key == "crystal_fea":
                        for i, crystal_fea in enumerate(pred.cpu().detach().numpy()):
                            predictions[n * batch_size + i][key] = crystal_fea
            return predictions
        else:
            raise Exception("input should either be a graph or list of graphs!")

    @staticmethod
    def split(x: Tensor, n: Tensor) -> Sequence[Tensor]:
        """Split a batched result Tensor into a list of Tensors."""
        print(x, n)
        start = 0
        result = []
        for i in n:
            result.append(x[start : start + i])
            start += i
        assert start == len(x), "Error: source tensor not correctly split!"
        return result

    def as_dict(self):
        """Return the CHGNet weights and args in a dictionary."""
        out = {"state_dict": self.state_dict(), "model_args": self.model_args}
        return out

    @classmethod
    def from_dict(cls, dict, **kwargs):
        """Build a CHGNet from a saved dictionary."""
        chgnet = CHGNet(**dict["model_args"])
        chgnet.load_state_dict(dict["state_dict"], **kwargs)
        return chgnet

    @classmethod
    def from_file(cls, path, **kwargs):
        """Build a CHGNet from a saved file."""
        state = torch.load(path, map_location=torch.device("cpu"))
        chgnet = CHGNet.from_dict(state["model"], **kwargs)
        return chgnet

    @classmethod
    def load(cls, model_name="MPtrj-efsm"):
        """Load pretrained CHGNet."""
        current_dir = os.path.dirname(os.path.abspath(__file__))
        if model_name == "MPtrj-efsm":
            return cls.from_file(
                os.path.join(current_dir, "../pretrained/e30f77s348m32.pth.tar")
            )
        else:
            raise Exception("model_name not supported")
