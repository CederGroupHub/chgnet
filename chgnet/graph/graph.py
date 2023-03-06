from __future__ import annotations

from chgnet import utils


class Node:
    """a node in a graph."""

    def __init__(self, index: int, info: dict = None):
        self.index = index
        self.info = info
        self.neighbors = {}

    def add_neighbor(self, index, edge):
        """Draw an edge between self and node
        Args:
            node (Node): the neighboring node
            edge (int): for simplicity, just use an index to track this edge.
        """
        if index not in self.neighbors:
            self.neighbors[index] = [edge]
        else:
            self.neighbors[index].append(edge)


class UndirectedEdge:
    """An edge in a graph."""

    def __init__(self, nodes: list, index: int = None, info: dict = None):
        self.nodes = nodes
        self.index = index
        self.info = info

    def __str__(self):
        return f"UndirectedEdge{self.index}, Nodes{self.nodes}, info={self.info}"

    def __eq__(self, other):
        if self.nodes == other.nodes and self.info == other.info:
            return True
        return False

    def __repr__(self):
        return str(self)


class DirectedEdge:
    """An edge in a graph."""

    def __init__(self, nodes: list, index: int, info: dict = None):
        self.nodes = nodes
        self.index = index
        self.info = info

    def make_undirected(self, index, info=None):
        if info is None:
            info = {}
        info["distance"] = self.info["distance"]
        return UndirectedEdge(self.nodes, index, info)

    def __eq__(self, other):
        if (
            self.nodes == other.nodes
            and (self.info["image"] == other.info["image"]).all()
        ):
            # print(self.nodes, other.nodes)
            # print(self.info['image'], other.info['image'])
            print(
                "!!!!!! the two directed edges are equal "
                "but this operation is not supposed to happen"
            )
            return True
        if (
            self.nodes == other.nodes[::-1]
            and (self.info["image"] == -1 * other.info["image"]).all()
        ):
            return True
        return False

    def __str__(self):
        return f"DirectedEdge{self.index}, Nodes{self.nodes}, info={self.info}"

    def __repr__(self):
        return str(self)


class Graph:
    def __init__(self, nodes: list):
        self.nodes = nodes
        self.directed_edges = {}
        self.directed_edges_list = []
        self.undirected_edges = {}
        self.undirected_edges_list = []

    def add_edge(self, center_index, neighbor_index, image, distance):
        """Add an directed edge to the graph
        Args:
            center_index: center node index
            neighbor_index: neighbor node index
            image: the periodic cell image the neighbor is from
            distance: distance between center and neighbor.
        """
        directed_edge_index = len(self.directed_edges_list)
        directed_edge = DirectedEdge(
            [center_index, neighbor_index],
            index=directed_edge_index,
            info={"image": image, "distance": distance},
        )

        tmp = frozenset([center_index, neighbor_index])
        if tmp not in self.undirected_edges:
            directed_edge.info["undirected_edge_index"] = len(
                self.undirected_edges_list
            )
            undirected_edge = directed_edge.make_undirected(
                index=len(self.undirected_edges_list),
                info={"directed_edge_index": [directed_edge_index]},
            )
            self.undirected_edges[tmp] = [undirected_edge]
            self.undirected_edges_list.append(undirected_edge)
            self.nodes[center_index].add_neighbor(neighbor_index, directed_edge)
            self.directed_edges_list.append(directed_edge)
            return
        else:
            # this pair of nodes has been added before, we need to see if this time,
            # it's the other directed edge of the same undirected edge or it's another
            # totally different undirected edge that has different image and distance
            for undirected_edge in self.undirected_edges[tmp]:
                if abs(undirected_edge.info["distance"] - distance) < 1e-6:
                    if len(undirected_edge.info["directed_edge_index"]) == 1:
                        e = self.directed_edges_list[
                            undirected_edge.info["directed_edge_index"][0]
                        ]
                        if e == directed_edge:
                            directed_edge.info["undirected_edge_index"] = e.info[
                                "undirected_edge_index"
                            ]
                            self.nodes[center_index].add_neighbor(
                                neighbor_index, directed_edge
                            )
                            self.directed_edges_list.append(directed_edge)
                            undirected_edge.info["directed_edge_index"].append(
                                directed_edge_index
                            )
                            return

            # no undirected_edge matches to this directed edge
            directed_edge.info["undirected_edge_index"] = len(
                self.undirected_edges_list
            )
            undirected_edge = directed_edge.make_undirected(
                index=len(self.undirected_edges_list),
                info={"directed_edge_index": [directed_edge_index]},
            )
            self.undirected_edges[tmp].append(undirected_edge)
            self.undirected_edges_list.append(undirected_edge)
            self.nodes[center_index].add_neighbor(neighbor_index, directed_edge)
            self.directed_edges_list.append(directed_edge)
            return

    def adjacency_list(self):
        """Return:
        graph: the adjacency list
        [[0, 1, 0],
        [0, 2, 1],
        ... ...  ]
        the fist column specifies center/source node,
        the second column specifies neighbor/destination node,
        the third column specifies the undirected edge index
        (this is essentially the directed2undirected mapping)
        of the directed edge on this row.
        """
        graph = [edge.nodes for edge in self.directed_edges_list]
        directed2undirected = [
            edge.info["undirected_edge_index"] for edge in self.directed_edges_list
        ]
        return graph, directed2undirected

    def line_graph_adjacency_list(self, cutoff):
        """Return: line graph adjacency list
        [[0, 1, 1, 2, 2],
        [0, 1, 1, 4, 23],
        [1, 4, 23, 5, 66],
        ... ...  ]
        the fist column specifies node index at this angle,
        the second column specifies 1st undirected edge index,
        the third column specifies 1st directed edge index,
        the fourth column specifies 2nd undirected edge index,
        the fifth column specifies 2snd directed edge index,.
        """
        assert len(self.directed_edges_list) == 2 * len(
            self.undirected_edges_list
        ), f"Error: number of directed edge{len(self.directed_edges_list)} != 2 * number of undirected edge{len(self.directed_edges_list)}!"
        line_graph = []
        undirected2directed = []
        for u_edge in self.undirected_edges_list:
            undirected2directed.append(u_edge.info["directed_edge_index"][0])
            if u_edge.info["distance"] > cutoff:
                continue
            center1, center2 = list(u_edge.nodes)
            try:
                directed_edge1, directed_edge2 = u_edge.info["directed_edge_index"]
            except:
                print("Did not find 2 Directed_edges !!!")
                print(u_edge)
                print(
                    "edge.info['directed_edge_index'] = ",
                    u_edge.info["directed_edge_index"],
                )
                print()
                print("len directed_edges_list = ", len(self.directed_edges_list))
                print("len undirected_edges_list = ", len(self.undirected_edges_list))
            # find directed edges starting at center1
            for directed_edges in self.nodes[center1].neighbors.values():
                for directed_edge in directed_edges:
                    if directed_edge.index == directed_edge1:
                        continue
                    if directed_edge.info["distance"] < cutoff:
                        # print('Forming angles1:')
                        # print(self.directed_edges[directed_edge1])
                        # print(directed_edge)
                        # print()
                        line_graph.append(
                            [
                                center1,
                                u_edge.index,
                                directed_edge1,
                                directed_edge.info["undirected_edge_index"],
                                directed_edge.index,
                            ]
                        )
            for directed_edges in self.nodes[center2].neighbors.values():
                for directed_edge in directed_edges:
                    if directed_edge.index == directed_edge2:
                        continue
                    if directed_edge.info["distance"] < cutoff:
                        # print('Forming angles2:')
                        # print(self.directed_edges[directed_edge2])
                        # print(directed_edge)
                        # print()
                        line_graph.append(
                            [
                                center2,
                                u_edge.index,
                                directed_edge2,
                                directed_edge.info["undirected_edge_index"],
                                directed_edge.index,
                            ]
                        )
        return line_graph, undirected2directed

    def undirected2directed(self):
        """The index map from undirected_edge index to one of its directed_edge index."""
        out = []
        for undirected_edge in self.undirected_edges_list:
            out.append(undirected_edge.info["directed_edge_index"][0])
        return out

    def as_dict(self):
        """Return dictionary serialization of a Graph."""
        return {
            "nodes": self.nodes,
            "directed_edges": self.directed_edges,
            "directed_edges_list": self.directed_edges_list,
            "undirected_edges": self.undirected_edges,
            "undirected_edges_list": self.undirected_edges_list,
        }

    def to(self, filename="graph.json"):
        """Save graph dictionary to file."""
        utils.write_json(self.as_dict(), filename)
        return

    def __str__(self):
        return (
            f"Graph(num_nodes={len(self.nodes)}, num_directed_edges={len(self.directed_edges_list)}, "
            f"num_undirected_edges={len(self.undirected_edges_list)})"
        )

    def __repr__(self):
        return str(self)
