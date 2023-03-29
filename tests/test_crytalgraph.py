from __future__ import annotations

import numpy as np
from pymatgen.core import Structure

from chgnet import ROOT
from chgnet.graph import CrystalGraphConverter

structure = Structure.from_file(f"{ROOT}/examples/o-LiMnO2_unit.cif")


def test_crystalgraph1():
    converter = CrystalGraphConverter(atom_graph_cutoff=5, bond_graph_cutoff=3)
    graph = converter(structure)

    assert graph.composition == "Li2 Mn2 O4"
    assert [i.item() for i in graph.atomic_number] == [3, 3, 25, 25, 8, 8, 8, 8]
    assert list(graph.atom_frac_coord.shape) == [8, 3]
    assert list(graph.atom_graph.shape) == [384, 2]
    assert [i.item() for i in graph.atom_graph[100]] == [2, 4]
    assert [i.item() for i in graph.atom_graph[200]] == [4, 2]
    assert list(graph.bond_graph.shape) == [744, 5]
    assert [i.item() for i in graph.bond_graph[100]] == [5, 37, 286, 142, 279]
    assert [i.item() for i in graph.bond_graph[200]] == [7, 65, 368, 190, 359]
    assert list(graph.lattice.shape) == [3, 3]
    assert list(graph.undirected2directed.shape) == [192]
    assert list(graph.directed2undirected.shape) == [384]


def test_crystalgraph_different_cutoff():
    converter = CrystalGraphConverter(atom_graph_cutoff=5.5, bond_graph_cutoff=3.5)
    graph = converter(structure)

    assert list(graph.atom_frac_coord.shape) == [8, 3]
    assert list(graph.atom_graph.shape) == [624, 2]
    assert [i.item() for i in graph.atom_graph[100]] == [1, 2]
    assert [i.item() for i in graph.atom_graph[200]] == [2, 3]
    assert list(graph.bond_graph.shape) == [2448, 5]
    assert [i.item() for i in graph.bond_graph[100]] == [3, 37, 244, 135, 293]
    assert [i.item() for i in graph.bond_graph[200]] == [5, 41, 416, 285, 437]
    assert list(graph.lattice.shape) == [3, 3]
    assert list(graph.undirected2directed.shape) == [312]
    assert list(graph.directed2undirected.shape) == [624]


def test_crystalgraph_perturb():
    converter = CrystalGraphConverter(atom_graph_cutoff=5, bond_graph_cutoff=3)
    structure_perturbed = structure.copy()
    np.random.seed(100)
    structure_perturbed.perturb(distance=0.1)
    graph = converter(structure_perturbed)

    assert list(graph.atom_frac_coord.shape) == [8, 3]
    assert list(graph.atom_graph.shape) == [416, 2]
    assert [i.item() for i in graph.atom_graph[100]] == [1, 5]
    assert [i.item() for i in graph.atom_graph[200]] == [3, 2]
    assert list(graph.bond_graph.shape) == [934, 5]
    assert [i.item() for i in graph.bond_graph[100]] == [4, 36, 242, 173, 249]
    assert [i.item() for i in graph.bond_graph[200]] == [1, 55, 57, 60, 62]
    assert list(graph.lattice.shape) == [3, 3]
    assert list(graph.undirected2directed.shape) == [208]
    assert list(graph.directed2undirected.shape) == [416]


def test_crystalgraph_strained():
    converter = CrystalGraphConverter(atom_graph_cutoff=5, bond_graph_cutoff=3)
    structure_strained = structure.copy()
    structure_strained.apply_strain([0.1, -0.3, 0.5])
    graph = converter(structure_strained)

    assert list(graph.atom_frac_coord.shape) == [8, 3]
    assert list(graph.atom_graph.shape) == [336, 2]
    assert [i.item() for i in graph.atom_graph[100]] == [2, 7]
    assert [i.item() for i in graph.atom_graph[200]] == [4, 3]
    assert list(graph.bond_graph.shape) == [360, 5]
    assert [i.item() for i in graph.bond_graph[100]] == [7, 52, 321, 87, 334]
    assert [i.item() for i in graph.bond_graph[200]] == [2, 100, 121, 87, 103]
    assert list(graph.lattice.shape) == [3, 3]
    assert list(graph.undirected2directed.shape) == [168]
    assert list(graph.directed2undirected.shape) == [336]


def test_crystalgraph_supercell():
    converter = CrystalGraphConverter(atom_graph_cutoff=5, bond_graph_cutoff=3)
    structure_supercell = structure.copy()
    structure_supercell.make_supercell([2, 3, 4])
    graph = converter(structure_supercell)

    assert graph.composition == "Li48 Mn48 O96"
    assert list(graph.atom_frac_coord.shape) == [192, 3]
    assert list(graph.atom_graph.shape) == [9216, 2]
    assert [i.item() for i in graph.atom_graph[1000]] == [20, 24]
    assert [i.item() for i in graph.atom_graph[2000]] == [41, 5]
    assert list(graph.bond_graph.shape) == [17856, 5]
    assert [i.item() for i in graph.bond_graph[1000]] == [6, 314, 317, 300, 302]
    assert [i.item() for i in graph.bond_graph[10000]] == [74, 3074, 3592, 3050, 3561]
    assert list(graph.lattice.shape) == [3, 3]
    assert list(graph.undirected2directed.shape) == [4608]
    assert list(graph.directed2undirected.shape) == [9216]
