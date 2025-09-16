from __future__ import annotations

from time import perf_counter
from unittest.mock import patch

import numpy as np
from pymatgen.core import Structure

from chgnet import ROOT
from chgnet.graph import CrystalGraphConverter

structure = Structure.from_file(f"{ROOT}/examples/mp-18767-LiMnO2.cif")
converter = CrystalGraphConverter(atom_graph_cutoff=5, bond_graph_cutoff=3)
converter_legacy = CrystalGraphConverter(
    atom_graph_cutoff=5, bond_graph_cutoff=3, algorithm="legacy", verbose=True
)
converter_fast = CrystalGraphConverter(
    atom_graph_cutoff=5, bond_graph_cutoff=3, algorithm="fast", verbose=True
)


def test_crystal_graph_legacy():
    assert converter_legacy.algorithm == "legacy"
    start = perf_counter()
    graph = converter_legacy(structure)
    print("Legacy test_crystal_graph time:", perf_counter() - start)  # noqa: T201

    assert graph.composition == "Li2 Mn2 O4"
    assert graph.atomic_number.tolist() == [3, 3, 25, 25, 8, 8, 8, 8]
    assert list(graph.atom_frac_coord.shape) == [8, 3]
    assert list(graph.atom_graph.shape) == [384, 2]
    assert (graph.atom_graph[:, 0] == 0).sum().item() == 48
    assert (graph.atom_graph[:, 1] == 0).sum().item() == 48
    assert (graph.atom_graph[:, 0] == 4).sum().item() == 48
    assert (graph.atom_graph[:, 0] == 7).sum().item() == 48

    assert list(graph.bond_graph.shape) == [744, 5]
    assert (graph.bond_graph[:, 0] == 1).sum().item() == 72
    assert (graph.bond_graph[:, 1] == 100).sum().item() == 16
    assert (graph.bond_graph[:, 3] == 100).sum().item() == 16
    assert (graph.bond_graph[:, 2] == 348).sum().item() == 8
    assert (graph.bond_graph[:, 4] == 121).sum().item() == 8
    assert list(graph.lattice.shape) == [3, 3]
    assert list(graph.undirected2directed.shape) == [192]
    assert list(graph.directed2undirected.shape) == [384]


def test_crystal_graph_fast():
    assert converter_fast.algorithm == "fast"
    start = perf_counter()
    graph = converter_fast(structure)
    print("Fast test_crystal_graph time:", perf_counter() - start)  # noqa: T201

    assert graph.composition == "Li2 Mn2 O4"
    assert graph.atomic_number.tolist() == [3, 3, 25, 25, 8, 8, 8, 8]
    assert list(graph.atom_frac_coord.shape) == [8, 3]
    assert list(graph.atom_graph.shape) == [384, 2]
    assert (graph.atom_graph[:, 0] == 0).sum().item() == 48
    assert (graph.atom_graph[:, 1] == 0).sum().item() == 48
    assert (graph.atom_graph[:, 0] == 4).sum().item() == 48
    assert (graph.atom_graph[:, 0] == 7).sum().item() == 48

    assert list(graph.bond_graph.shape) == [744, 5]
    assert (graph.bond_graph[:, 0] == 1).sum().item() == 72
    assert (graph.bond_graph[:, 1] == 100).sum().item() == 16
    assert (graph.bond_graph[:, 3] == 100).sum().item() == 16
    assert (graph.bond_graph[:, 2] == 348).sum().item() == 8
    assert (graph.bond_graph[:, 4] == 121).sum().item() == 8
    assert list(graph.lattice.shape) == [3, 3]
    assert list(graph.undirected2directed.shape) == [192]
    assert list(graph.directed2undirected.shape) == [384]


def test_crystal_graph_different_cutoff_legacy():
    converter_legacy_2 = CrystalGraphConverter(
        atom_graph_cutoff=5.5, bond_graph_cutoff=3.5, algorithm="legacy"
    )
    assert converter_legacy_2.algorithm == "legacy"

    start = perf_counter()
    graph = converter_legacy_2(structure)
    print("Legacy test_crystal_graph_different_cutoff time:", perf_counter() - start)  # noqa: T201

    assert list(graph.atom_frac_coord.shape) == [8, 3]
    assert list(graph.atom_graph.shape) == [624, 2]
    assert (graph.atom_graph[:, 0] == 5).sum().item() == 78
    assert (graph.atom_graph[:, 1] == 5).sum().item() == 78
    assert (graph.atom_graph[:, 1] == 7).sum().item() == 78

    assert list(graph.bond_graph.shape) == [2448, 5]
    assert (graph.bond_graph[:, 0] == 1).sum().item() == 306
    assert (graph.bond_graph[:, 1] == 100).sum().item() == 0
    assert (graph.bond_graph[:, 3] == 100).sum().item() == 0
    assert (graph.bond_graph[:, 2] == 250).sum().item() == 17
    assert (graph.bond_graph[:, 4] == 50).sum().item() == 17
    assert list(graph.lattice.shape) == [3, 3]
    assert list(graph.undirected2directed.shape) == [312]
    assert list(graph.directed2undirected.shape) == [624]


def test_crystal_graph_different_cutoff_fast():
    converter_fast_2 = CrystalGraphConverter(
        atom_graph_cutoff=5.5, bond_graph_cutoff=3.5, algorithm="fast"
    )
    assert converter_fast_2.algorithm == "fast"

    start = perf_counter()
    graph = converter_fast_2(structure)
    print("Fast test_crystal_graph_different_cutoff time:", perf_counter() - start)  # noqa: T201

    assert list(graph.atom_frac_coord.shape) == [8, 3]
    assert list(graph.atom_graph.shape) == [624, 2]
    assert (graph.atom_graph[:, 0] == 5).sum().item() == 78
    assert (graph.atom_graph[:, 1] == 5).sum().item() == 78
    assert (graph.atom_graph[:, 1] == 7).sum().item() == 78

    assert list(graph.bond_graph.shape) == [2448, 5]
    assert (graph.bond_graph[:, 0] == 1).sum().item() == 306
    assert (graph.bond_graph[:, 1] == 100).sum().item() == 0
    assert (graph.bond_graph[:, 3] == 100).sum().item() == 0
    assert (graph.bond_graph[:, 2] == 250).sum().item() == 17
    assert (graph.bond_graph[:, 4] == 50).sum().item() == 17
    assert list(graph.lattice.shape) == [3, 3]
    assert list(graph.undirected2directed.shape) == [312]
    assert list(graph.directed2undirected.shape) == [624]


def test_crystal_graph_perturb_legacy():
    structure_perturbed = structure.copy()
    fixed_rng = np.random.default_rng(0)
    with patch("numpy.random.default_rng", return_value=fixed_rng):
        structure_perturbed.perturb(distance=0.1)

    start = perf_counter()
    graph = converter_legacy(structure_perturbed)
    print("Legacy test_crystal_graph_perturb time:", perf_counter() - start)  # noqa: T201

    assert list(graph.atom_frac_coord.shape) == [8, 3]
    assert list(graph.atom_graph.shape) == [392, 2]
    assert (graph.atom_graph[:, 0] == 3).sum().item() == 48
    assert (graph.atom_graph[:, 1] == 3).sum().item() == 48
    assert (graph.atom_graph[:, 1] == 6).sum().item() == 50

    assert list(graph.bond_graph.shape) == [732, 5]
    assert (graph.bond_graph[:, 0] == 1).sum().item() == 90
    assert (graph.bond_graph[:, 1] == 36).sum().item() == 0
    assert (graph.bond_graph[:, 3] == 36).sum().item() == 0
    assert (graph.bond_graph[:, 2] == 306).sum().item() == 0
    assert (graph.bond_graph[:, 4] == 120).sum().item() == 0
    assert list(graph.lattice.shape) == [3, 3]
    assert list(graph.undirected2directed.shape) == [196]
    assert list(graph.directed2undirected.shape) == [392]


def test_crystal_graph_perturb_fast():
    structure_perturbed = structure.copy()
    fixed_rng = np.random.default_rng(0)
    with patch("numpy.random.default_rng", return_value=fixed_rng):
        structure_perturbed.perturb(distance=0.1)

    start = perf_counter()
    graph = converter_fast(structure_perturbed)
    print("Fast test_crystal_graph_perturb time:", perf_counter() - start)  # noqa: T201

    assert list(graph.atom_frac_coord.shape) == [8, 3]
    assert list(graph.atom_graph.shape) == [392, 2]
    assert (graph.atom_graph[:, 0] == 3).sum().item() == 48
    assert (graph.atom_graph[:, 1] == 3).sum().item() == 48
    assert (graph.atom_graph[:, 1] == 6).sum().item() == 50

    assert list(graph.bond_graph.shape) == [732, 5]
    assert (graph.bond_graph[:, 0] == 1).sum().item() == 90
    assert (graph.bond_graph[:, 1] == 36).sum().item() == 0
    assert (graph.bond_graph[:, 3] == 36).sum().item() == 0
    assert (graph.bond_graph[:, 2] == 306).sum().item() == 0
    assert (graph.bond_graph[:, 4] == 120).sum().item() == 0
    assert list(graph.lattice.shape) == [3, 3]
    assert list(graph.undirected2directed.shape) == [196]
    assert list(graph.directed2undirected.shape) == [392]


def test_crystal_graph_isotropic_strained_legacy():
    structure_strained = structure.copy()
    structure_strained.apply_strain([0.1, 0.1, 0.1])

    start = perf_counter()
    graph = converter_legacy(structure_strained)
    print("Legacy test_crystal_graph_isotropic_strained time:", perf_counter() - start)  # noqa: T201

    assert list(graph.atom_frac_coord.shape) == [8, 3]
    assert list(graph.atom_graph.shape) == [264, 2]
    assert (graph.atom_graph[:, 0] == 3).sum().item() == 34
    assert (graph.atom_graph[:, 1] == 3).sum().item() == 34
    assert (graph.atom_graph[:, 0] == 7).sum().item() == 32

    assert list(graph.bond_graph.shape) == [288, 5]
    assert list(graph.lattice.shape) == [3, 3]
    assert list(graph.undirected2directed.shape) == [132]
    assert list(graph.directed2undirected.shape) == [264]


def test_crystal_graph_isotropic_strained_fast():
    structure_strained = structure.copy()
    structure_strained.apply_strain([0.1, 0.1, 0.1])

    start = perf_counter()
    graph = converter_fast(structure_strained)
    print("Fast test_crystal_graph_isotropic_strained time:", perf_counter() - start)  # noqa: T201

    assert list(graph.atom_frac_coord.shape) == [8, 3]
    assert list(graph.atom_graph.shape) == [264, 2]
    assert (graph.atom_graph[:, 0] == 3).sum().item() == 34
    assert (graph.atom_graph[:, 1] == 3).sum().item() == 34
    assert (graph.atom_graph[:, 0] == 7).sum().item() == 32

    assert list(graph.bond_graph.shape) == [288, 5]
    assert list(graph.lattice.shape) == [3, 3]
    assert list(graph.undirected2directed.shape) == [132]
    assert list(graph.directed2undirected.shape) == [264]


def test_crystal_graph_anisotropic_strained_legacy():
    structure_strained = structure.copy()
    structure_strained.apply_strain([0.2, -0.3, 0.5])

    start = perf_counter()
    graph = converter_legacy(structure_strained)
    print(  # noqa: T201
        "Legacy test_crystal_graph_anisotropic_strained time:", perf_counter() - start
    )

    assert list(graph.atom_frac_coord.shape) == [8, 3]
    assert list(graph.atom_graph.shape) == [336, 2]
    assert (graph.atom_graph[:, 0] == 3).sum().item() == 42
    assert (graph.atom_graph[:, 1] == 3).sum().item() == 42
    assert (graph.atom_graph[:, 0] == 7).sum().item() == 42

    assert list(graph.bond_graph.shape) == [256, 5]
    assert list(graph.lattice.shape) == [3, 3]
    assert list(graph.undirected2directed.shape) == [168]
    assert list(graph.directed2undirected.shape) == [336]


def test_crystal_graph_anisotropic_strained_fast():
    structure_strained = structure.copy()
    structure_strained.apply_strain([0.2, -0.3, 0.5])

    start = perf_counter()
    graph = converter_fast(structure_strained)
    print("Fast test_crystal_graph_anisotropic_strained time:", perf_counter() - start)  # noqa: T201

    assert list(graph.atom_frac_coord.shape) == [8, 3]
    assert list(graph.atom_graph.shape) == [336, 2]
    assert (graph.atom_graph[:, 0] == 3).sum().item() == 42
    assert (graph.atom_graph[:, 1] == 3).sum().item() == 42
    assert (graph.atom_graph[:, 0] == 7).sum().item() == 42

    assert list(graph.bond_graph.shape) == [256, 5]
    assert list(graph.lattice.shape) == [3, 3]
    assert list(graph.undirected2directed.shape) == [168]
    assert list(graph.directed2undirected.shape) == [336]


def test_crystal_graph_supercell_legacy():
    supercell = structure.make_supercell([2, 3, 4], in_place=False)

    start = perf_counter()
    graph = converter_legacy(supercell)
    print("Legacy test_crystal_graph_supercell time:", perf_counter() - start)  # noqa: T201

    assert graph.composition == "Li48 Mn48 O96"
    assert list(graph.atom_frac_coord.shape) == [192, 3]
    assert list(graph.atom_graph.shape) == [9216, 2]
    assert (graph.atom_graph[:, 0] == 30).sum().item() == 48
    assert (graph.atom_graph[:, 1] == 30).sum().item() == 48
    assert (graph.atom_graph[:, 0] == 70).sum().item() == 48

    assert list(graph.bond_graph.shape) == [17856, 5]
    assert (graph.bond_graph[:, 0] == 100).sum().item() == 72
    assert (graph.bond_graph[:, 1] == 623).sum().item() == 16
    assert (graph.bond_graph[:, 3] == 623).sum().item() == 16
    assert (graph.bond_graph[:, 2] == 2938).sum().item() == 8
    assert (graph.bond_graph[:, 4] == 121).sum().item() == 8
    assert list(graph.lattice.shape) == [3, 3]
    assert list(graph.undirected2directed.shape) == [4608]
    assert list(graph.directed2undirected.shape) == [9216]


def test_crystal_graph_supercell_fast():
    supercell = structure.make_supercell([2, 3, 4], in_place=False)

    start = perf_counter()
    graph = converter_fast(supercell)
    print("Fast test_crystal_graph_supercell time:", perf_counter() - start)  # noqa: T201# noqa: T201

    assert graph.composition == "Li48 Mn48 O96"
    assert list(graph.atom_frac_coord.shape) == [192, 3]
    assert list(graph.atom_graph.shape) == [9216, 2]
    assert (graph.atom_graph[:, 0] == 30).sum().item() == 48
    assert (graph.atom_graph[:, 1] == 30).sum().item() == 48
    assert (graph.atom_graph[:, 0] == 70).sum().item() == 48

    assert list(graph.bond_graph.shape) == [17856, 5]
    assert (graph.bond_graph[:, 0] == 100).sum().item() == 72
    assert (graph.bond_graph[:, 1] == 623).sum().item() == 16
    assert (graph.bond_graph[:, 3] == 623).sum().item() == 16
    assert (graph.bond_graph[:, 2] == 2938).sum().item() == 8
    assert (graph.bond_graph[:, 4] == 121).sum().item() == 8
    assert list(graph.lattice.shape) == [3, 3]
    assert list(graph.undirected2directed.shape) == [4608]
    assert list(graph.directed2undirected.shape) == [9216]


def test_crystal_graph_stability_legacy():
    total_time = 0
    for _ in range(20):
        structure_perturbed = structure.make_supercell([2, 2, 2], in_place=False)
        structure_perturbed.perturb(distance=0.5)
        start = perf_counter()
        graph = converter_legacy(structure_perturbed)
        total_time += perf_counter() - start

        assert (
            graph.directed2undirected.shape[0] == 2 * graph.undirected2directed.shape[0]
        )
        assert graph.atom_graph.shape[0] == graph.directed2undirected.shape[0]
    print("Legacy test_crystal_graph_stability time:", total_time)  # noqa: T201


def test_crystal_graph_stability_fast():
    total_time = 0
    for _ in range(20):
        structure_perturbed = structure.make_supercell([2, 2, 2], in_place=False)
        structure_perturbed.perturb(distance=0.5)
        start = perf_counter()
        graph = converter_fast(structure_perturbed)
        total_time += perf_counter() - start

        assert (
            graph.directed2undirected.shape[0] == 2 * graph.undirected2directed.shape[0]
        )
        assert graph.atom_graph.shape[0] == graph.directed2undirected.shape[0]
    print("Fast test_crystal_graph_stability time:", total_time)  # noqa: T201


def test_crystal_graph_repr():
    graph = converter_legacy(structure)
    assert (
        repr(graph) == "CrystalGraph(composition='Li2 Mn2 O4', atom_graph_cutoff=5, "
        "bond_graph_cutoff=3, n_atoms=8, atom_graph_len=384, bond_graph_len=744)"
    )


def test_isolated_atoms():
    converter = CrystalGraphConverter(
        atom_graph_cutoff=5, bond_graph_cutoff=3, on_isolated_atoms="warn"
    )
    graph = converter(structure)
    assert graph.num_isolated_atoms == 0
