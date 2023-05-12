from __future__ import annotations

import numpy as np
import pytest

from chgnet.graph.graph import Graph, Node


@pytest.fixture()
def graph() -> Graph:
    """Create a graph with 3 nodes and 3 directed edges."""
    nodes = [Node(index=idx) for idx in range(3)]
    graph = Graph(nodes)
    graph.add_edge(0, 1, np.array([0, 0, 0]), 1.0)
    graph.add_edge(0, 2, np.array([0, 0, 0]), 2.0)
    graph.add_edge(1, 2, np.array([0, 0, 0]), 3.0)
    return graph


def test_add_edge(graph: Graph) -> None:
    assert len(graph.directed_edges_list) == 3
    assert len(graph.undirected_edges_list) == 3


def test_adjacency_list(graph: Graph) -> None:
    adj_list, directed2undirected = graph.adjacency_list()
    assert len(adj_list) == 3
    assert len(directed2undirected) == 3
    assert adj_list[0] == [0, 1]
    assert adj_list[1] == [0, 2]
    assert adj_list[2] == [1, 2]
    assert directed2undirected[0] == 0
    assert directed2undirected[1] == 1
    assert directed2undirected[2] == 2


def test_undirected2directed(graph: Graph) -> None:
    undirected2directed = graph.undirected2directed()
    assert len(undirected2directed) == 3
    assert undirected2directed[0] == 0
    assert undirected2directed[1] == 1


def test_as_dict(graph: Graph) -> None:
    graph_dict = graph.as_dict()
    assert len(graph_dict["nodes"]) == 3
    assert len(graph_dict["directed_edges"]) == 0
    assert len(graph_dict["directed_edges_list"]) == 3
    assert len(graph_dict["undirected_edges"]) == 3
    assert len(graph_dict["undirected_edges_list"]) == 3


@pytest.fixture()
def bigraph() -> Graph:
    """Create a bi-directional graph with 3 nodes and 4 bi-directed edges."""
    nodes = [Node(index=idx) for idx in range(3)]
    bigraph = Graph(nodes)
    bigraph.add_edge(0, 1, np.array([0, 0, 0]), 1.0)
    bigraph.add_edge(0, 2, np.array([0, 0, 0]), 2.0)
    bigraph.add_edge(1, 0, np.array([0, 0, 0]), 1.0)
    bigraph.add_edge(1, 2, np.array([0, 0, 0]), 5.0)
    bigraph.add_edge(1, 1, np.array([0, 0, 1]), 4.0)
    bigraph.add_edge(1, 1, np.array([0, 0, -1]), 4.0)
    bigraph.add_edge(2, 0, np.array([0, 0, 0]), 2.0)
    bigraph.add_edge(2, 1, np.array([0, 0, 0]), 5.0)
    return bigraph


def test_add_biedge(bigraph: Graph) -> None:
    assert len(bigraph.directed_edges_list) == 8
    assert len(bigraph.undirected_edges_list) == 4


def test_line_graph(bigraph: Graph) -> None:
    adj_list, directed2undirected = bigraph.adjacency_list()
    line_adj_list, undirected2directed = bigraph.line_graph_adjacency_list(cutoff=7)
    # adj_list
    assert len(adj_list) == 8
    assert len(directed2undirected) == 8
    assert adj_list[0] == [0, 1]
    assert adj_list[1] == [0, 2]
    assert adj_list[2] == [1, 0]
    assert adj_list[6] == [2, 0]
    assert directed2undirected[0] == 0
    assert directed2undirected[1] == 1
    assert directed2undirected[2] == 0

    # line_adj_list
    assert len(line_adj_list) == 16
    assert line_adj_list[0] == [0, 0, 0, 1, 1]
    assert line_adj_list[5] == [2, 1, 6, 2, 7]
    assert line_adj_list[10] == [1, 3, 4, 0, 2]
    assert len(undirected2directed) == 4
    assert undirected2directed[0] == 0
    assert undirected2directed[1] == 1
    assert undirected2directed[2] == 3
