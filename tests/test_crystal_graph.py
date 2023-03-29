from __future__ import annotations

import numpy as np
from pymatgen.core import Structure

from chgnet import ROOT
from chgnet.graph import CrystalGraphConverter

np.random.seed(0)

structure = Structure.from_file(f"{ROOT}/examples/o-LiMnO2_unit.cif")
converter = CrystalGraphConverter(atom_graph_cutoff=5, bond_graph_cutoff=3)


def test_crystal_graph():
    graph = converter(structure)

    assert graph.composition == "Li2 Mn2 O4"
    assert graph.atomic_number.tolist() == [3, 3, 25, 25, 8, 8, 8, 8]
    assert list(graph.atom_frac_coord.shape) == [8, 3]
    assert list(graph.atom_graph.shape) == [384, 2]
    assert graph.atom_graph[100].tolist() == [2, 4]
    assert graph.atom_graph[200].tolist() == [4, 2]
    assert list(graph.bond_graph.shape) == [744, 5]
    assert graph.bond_graph[100].tolist() == [5, 37, 286, 142, 279]
    assert graph.bond_graph[200].tolist() == [7, 65, 368, 190, 359]
    assert list(graph.lattice.shape) == [3, 3]
    assert list(graph.undirected2directed.shape) == [192]
    assert list(graph.directed2undirected.shape) == [384]


def test_crystal_graph_different_cutoff():
    converter = CrystalGraphConverter(atom_graph_cutoff=5.5, bond_graph_cutoff=3.5)
    graph = converter(structure)

    assert list(graph.atom_frac_coord.shape) == [8, 3]
    assert list(graph.atom_graph.shape) == [624, 2]
    assert graph.atom_graph[100].tolist() == [1, 2]
    assert graph.atom_graph[200].tolist() == [2, 3]
    assert list(graph.bond_graph.shape) == [2448, 5]
    assert graph.bond_graph[100].tolist() == [3, 37, 244, 135, 293]
    assert graph.bond_graph[200].tolist() == [5, 41, 416, 285, 437]
    assert list(graph.lattice.shape) == [3, 3]
    assert list(graph.undirected2directed.shape) == [312]
    assert list(graph.directed2undirected.shape) == [624]


def test_crystal_graph_perturb():
    structure_perturbed = structure.copy()
    structure_perturbed.perturb(distance=0.1)
    graph = converter(structure_perturbed)

    assert list(graph.atom_frac_coord.shape) == [8, 3]
    assert list(graph.atom_graph.shape) == [410, 2]
    assert graph.atom_graph[100].tolist() == [2, 6]
    assert graph.atom_graph[200].tolist() == [3, 7]
    assert list(graph.bond_graph.shape) == [688, 5]
    assert graph.bond_graph[100].tolist() == [7, 36, 400, 68, 393]
    assert graph.bond_graph[200].tolist() == [1, 59, 62, 56, 59]
    assert list(graph.lattice.shape) == [3, 3]
    assert list(graph.undirected2directed.shape) == [205]
    assert list(graph.directed2undirected.shape) == [410]


def test_crystal_graph_strained():
    structure_strained = structure.copy()
    structure_strained.apply_strain([0.1, -0.3, 0.5])
    graph = converter(structure_strained)

    assert list(graph.atom_frac_coord.shape) == [8, 3]
    assert list(graph.atom_graph.shape) == [336, 2]
    assert graph.atom_graph[100].tolist() == [2, 7]
    assert graph.atom_graph[200].tolist() == [4, 3]
    assert list(graph.bond_graph.shape) == [360, 5]
    assert graph.bond_graph[100].tolist() == [7, 52, 321, 87, 334]
    assert graph.bond_graph[200].tolist() == [2, 100, 121, 87, 103]
    assert list(graph.lattice.shape) == [3, 3]
    assert list(graph.undirected2directed.shape) == [168]
    assert list(graph.directed2undirected.shape) == [336]


def test_crystal_graph_supercell():
    structure_supercell = structure.copy()
    structure_supercell.make_supercell([2, 3, 4])
    graph = converter(structure_supercell)

    assert graph.composition == "Li48 Mn48 O96"
    assert list(graph.atom_frac_coord.shape) == [192, 3]
    assert list(graph.atom_graph.shape) == [9216, 2]
    assert graph.atom_graph[1000].tolist() == [20, 24]
    assert graph.atom_graph[2000].tolist() == [41, 5]
    assert list(graph.bond_graph.shape) == [17856, 5]
    assert graph.bond_graph[1000].tolist() == [6, 314, 317, 300, 302]
    assert graph.bond_graph[10000].tolist() == [74, 3074, 3592, 3050, 3561]
    assert list(graph.lattice.shape) == [3, 3]
    assert list(graph.undirected2directed.shape) == [4608]
    assert list(graph.directed2undirected.shape) == [9216]
