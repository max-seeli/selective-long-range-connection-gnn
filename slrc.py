import networkx as nx
import torch_geometric
from torch_geometric.utils import to_networkx, from_networkx
import matplotlib.pyplot as plt
import numpy as np
import itertools


def create_k_hop_graph(graph, k, target=0):
    """
    Creates a k-hop graph from a given graph and target node (default: 0).

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

    new_graph = nx.DiGraph()
    new_graph.add_nodes_from(graph.nodes)

    target_distances = nx.single_source_shortest_path_length(graph, target)
    for node in graph.nodes:
        if target_distances[node] >= k:
            new_graph.add_edge(node, target)

    if is_torch:
        new_graph = from_networkx(new_graph)

    return new_graph


if __name__ == '__main__':

    G = nx.path_graph(5)
    k_hop_graph = create_k_hop_graph(G, 2)

    
    def create_polygon_layout(n):
        pos = {}
        for i in range(n):
            pos[i] = (np.cos(2*np.pi*i/n), np.sin(2*np.pi*i/n))
        return pos

    pos = create_polygon_layout(5)

    plt.figure(figsize=(12, 6))
    plt.subplot(121)
    nx.draw(G, with_labels=True, font_weight='bold', node_color='lightblue', pos=pos)
    plt.title('Original Graph')


    plt.subplot(122)
    nx.draw(k_hop_graph, with_labels=True, font_weight='bold', node_color='lightgreen', pos=pos)
    plt.title('K-Hop Graph (k=2)')

    plt.show()

    # Create a graph (torch_geometric)
