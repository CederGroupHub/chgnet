from __future__ import annotations

import pytest

from chgnet.graph.graph import Graph, Node


@pytest.fixture
def graph() -> Graph:
    """Create a graph with 3 nodes and 3 edges."""
    nodes = [Node(index=idx) for idx in range(3)]
    graph = Graph(nodes)
    graph.add_edge(0, 1, [0, 0, 0], 1.0)
    graph.add_edge(0, 2, [0, 0, 0], 2.0)
    graph.add_edge(1, 2, [0, 0, 0], 3.0)
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


# def test_line_graph_adjacency_list(graph: Graph) -> None:
#     line_graph, undirected2directed = graph.line_graph_adjacency_list(cutoff=2.0)
#     assert len(line_graph) == 2
#     assert len(undirected2directed) == 2
#     assert line_graph[0] == [0, 0, 0, 1, 1]
#     assert line_graph[1] == [1, 1, 2, 2, 2]


def test_as_dict(graph: Graph) -> None:
    graph_dict = graph.as_dict()
    assert len(graph_dict["nodes"]) == 3
    assert len(graph_dict["directed_edges"]) == 0
    assert len(graph_dict["directed_edges_list"]) == 3
    assert len(graph_dict["undirected_edges"]) == 3
    assert len(graph_dict["undirected_edges_list"]) == 3
