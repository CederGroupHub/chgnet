from __future__ import annotations

import pytest
from pymatgen.core import Lattice, Structure
from pytest import CaptureFixture

from chgnet.graph import CrystalGraph
from chgnet.graph.converter import CrystalGraphConverter

lattice = Lattice.cubic(4)
species = ["Na", "Cl"]
coords = [[0, 0, 0], [0.5, 0.5, 0.5]]
NaCl = Structure(lattice, species, coords)


@pytest.fixture()
def _set_make_graph():
    # fixture to force make_graph to be None and then restore it after test
    from chgnet.graph import converter

    make_graph = converter.make_graph  # save original value
    converter.make_graph = None  # force make_graph to be None
    yield  # allows us to have cleanup after the test
    converter.make_graph = make_graph  # restore original value


@pytest.mark.parametrize(
    "atom_graph_cutoff, bond_graph_cutoff", [(5, 3), (5, None), (4, 2)]
)
def test_crystal_graph_converter_cutoff(atom_graph_cutoff, bond_graph_cutoff):
    converter = CrystalGraphConverter(
        atom_graph_cutoff=atom_graph_cutoff, bond_graph_cutoff=bond_graph_cutoff
    )
    assert converter.atom_graph_cutoff == atom_graph_cutoff
    assert converter.bond_graph_cutoff == bond_graph_cutoff or atom_graph_cutoff


@pytest.mark.parametrize("algorithm", ["legacy", "fast"])
def test_crystal_graph_converter_algorithm(algorithm):
    converter = CrystalGraphConverter(
        atom_graph_cutoff=5, bond_graph_cutoff=3, algorithm=algorithm
    )
    assert converter.algorithm == algorithm


def test_crystal_graph_converter_warns(_set_make_graph):
    with pytest.warns(UserWarning, match="Unknown algorithm='foobar', using `legacy`"):
        CrystalGraphConverter(
            atom_graph_cutoff=5, bond_graph_cutoff=3, algorithm="foobar"
        )
    with pytest.warns(
        UserWarning, match="`fast` algorithm is not available, using `legacy`"
    ):
        CrystalGraphConverter(
            atom_graph_cutoff=5, bond_graph_cutoff=3, algorithm="fast"
        )


@pytest.mark.parametrize("on_isolated_atoms", ["ignore", "warn", "error"])
def test_crystal_graph_converter_forward(
    on_isolated_atoms, capsys: CaptureFixture[str]
):
    atom_graph_cutoff = 5
    converter = CrystalGraphConverter(
        atom_graph_cutoff=atom_graph_cutoff,
        bond_graph_cutoff=3,
        on_isolated_atoms=on_isolated_atoms,
    )
    strained = NaCl.copy()
    strained.apply_strain(5)
    graph_id = "strained"
    err_msg = (
        f"Structure {graph_id=} has 2 isolated atom(s) with "
        f"{atom_graph_cutoff=}. "
        f"CHGNet calculation will likely go wrong"
    )

    if on_isolated_atoms == "error":
        with pytest.raises(ValueError) as exc_info:
            converter.forward(strained, graph_id=graph_id)
        assert err_msg in str(exc_info.value)
    else:
        crystal_graph = converter.forward(strained, graph_id=graph_id)
        assert isinstance(crystal_graph, CrystalGraph)
        stdout, stderr = capsys.readouterr()
        assert stdout == ""
        if on_isolated_atoms == "warn":
            assert err_msg in stderr
        else:
            assert stderr == ""


def test_crystal_graph_converter_as_dict_round_trip():
    expected = {"atom_graph_cutoff": 5, "bond_graph_cutoff": 3}
    converter = CrystalGraphConverter(**expected)
    converter2 = CrystalGraphConverter.from_dict(converter.as_dict())
    assert converter.atom_graph_cutoff == converter2.atom_graph_cutoff
    assert converter.bond_graph_cutoff == converter2.bond_graph_cutoff
    assert converter.algorithm == converter2.algorithm
