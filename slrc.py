import networkx as nx
import torch_geometric
from torch_geometric.utils import to_networkx, from_networkx
import matplotlib.pyplot as plt
import numpy as np
import itertools


def create_k_hop_graph(graph, k):
    """
    Creates a new graph from the original graph where there is an edge between two nodes
    if they are at least k hops apart in the original graph.

    Parameters
    ----------
    graph: networkx.Graph or torch_geometric.data.Data
        The graph
    k: int
        The minimum initial distance for edges to be included

    Returns
    -------
    networkx.Graph or torch_geometric.data.Data
        The new graph
    """
    is_torch = False
    if isinstance(graph, torch_geometric.data.Data):
        is_torch = True
        graph = to_networkx(graph, node_attrs=['x']).to_undirected()
        graph.remove_edges_from(nx.selfloop_edges(graph))

    new_graph = nx.Graph()
    new_graph.add_nodes_from(graph.nodes(data=True))

    for A, B in itertools.combinations(graph.nodes(), 2):
        try:
            if nx.shortest_path_length(graph, A, B) >= k:
                new_graph.add_edge(A, B)
        except nx.NetworkXNoPath:
            # No path exists between node1 and node2 in the original graph
            pass

    if is_torch:
        new_graph = from_networkx(new_graph)

    return new_graph
