import networkx as nx
from itertools import product, permutations
from string import ascii_uppercase
import math
from tqdm import tqdm
import torch
import random

"""
In the NeighborsMatch graph learning task, two primary node types are utilized:
green and blue. Green nodes are labeled alphabetically and possess a variable
count of blue neighboring nodes. The task's core objective is to identify the
green node whose blue neighbor count matches that of a designated green target
node.

The Tree-NeighborsMatch graph introduces a variation to this setup. Here, the
green target node forms the root of a tree structure, and all other green nodes
function as leaf nodes in a binary tree configuration, distinct from the blue
nodes. This variant emphasizes a hierarchical organization in the graph structure.
"""

def create_all_tree_neighbors_match_graph(d):
    """
    Creates all possible Tree-NeighborsMatch trees of depth :math:`d`.
    
    The number of possible Tree-NeighborsMatch trees of depth :math:`d` is:
    
    :math:`(2^d) * (2^d)!`
    
    Where :math:`(2^d)` is the different possibilities of the number of blue nodes 
    for the target and :math:`(2^d)!` is the number of possible permutations of the 
    blue nodes for the leaves. The number of possible blue nodes is between
    :math:`0` and :math:`2^d - 1` (inclusive).

    The target node is always labeled with a question mark (?). The leaf nodes
    are labeled with the first :math:`2^d` letters of the alphabet. If there are more
    than 26 leaf nodes, the alphabet is extended with AA, AB, AC, etc.

    Parameters
    ----------
    d: int
        The depth of the trees

    Yields
    ------
    networkx.Graph
        The next Tree-NeighborsMatch tree
    """
    num_leaf_nodes = 2**d
    possible_blue_nodes = list(range(0, num_leaf_nodes))

    target = '?'
    leaves = generate_alphabetic_labels(num_leaf_nodes)

    all_combinations = list(product(possible_blue_nodes, permutations(possible_blue_nodes)))
    random.shuffle(all_combinations)
    for blue in tqdm(all_combinations, 
                     total=num_leaf_nodes * math.factorial(num_leaf_nodes),
                     desc="Generating Trees"):
        num_target_blue_nodes = blue[0]
        num_per_leaf_blue_nodes = blue[1]

        G = create_balanced_binary_tree(target, leaves)

        # Add blue nodes
        for i in range(num_target_blue_nodes):
            node_name = f"{target}_{i}"
            G.add_node(node_name, color='blue')
            G.add_edge(target, f"{target}_{i}")

        for i, leaf in enumerate(leaves):
            for j in range(num_per_leaf_blue_nodes[i]):
                node_name = f"{leaf}_{j}"
                G.add_node(node_name, color='blue')
                G.add_edge(leaf, node_name)

        for node in G.nodes:
            G.nodes[node]['x'] = get_tensor_encoding(G.nodes[node]['color'])

        # Add graph y-label
        G.graph['y'] = torch.Tensor([[num_target_blue_nodes == leaf for leaf in num_per_leaf_blue_nodes]])

        yield G

def num_graphs(d):
    """
    Returns the number of possible Tree-NeighborsMatch graphs of depth `d`.
    With the restriction that the number of blue nodes is between `0` and 
    :math:`2^d - 1` (inclusive).
    
    Parameters
    ----------
    d: int
        The depth of the trees

    Returns
    -------
    int
        The number of possible Tree-NeighborsMatch graphs of depth `d`.
    """
    num_leaf_nodes = 2**d
    return num_leaf_nodes * math.factorial(num_leaf_nodes)

# Utility functions
# -----------------
def create_balanced_binary_tree(root_name, leaf_names):
    """
    Creates a balanced binary tree with the given root and leaf nodes.

    The root node and the leaf nodes are colored green. All other nodes are
    colored white.
    
    Parameters
    ----------
    root_name: str
        The name of the root node
    leaf_names: list
        The names of the leaf nodes

    Returns
    -------
    networkx.Graph
        The balanced binary tree
    """
    n = len(leaf_names)
    depth = math.log2(n)
    
    if not depth.is_integer():
        raise ValueError("The number of leaf nodes must be a power of 2")
    

    G = nx.DiGraph()
    G.add_node(root_name, color='green')

    def add_nodes(current_node, current_depth):
        if current_depth <= depth:
            left_child = f"{current_node}_L"
            right_child = f"{current_node}_R"
            G.add_node(left_child, color='white')
            G.add_node(right_child, color='white')
            G.add_edge(current_node, left_child)
            G.add_edge(current_node, right_child)
            add_nodes(left_child, current_depth + 1)
            add_nodes(right_child, current_depth + 1)

    # Build the tree recursively
    add_nodes(root_name, 1)

    # Assign names to the leaf nodes
    leaf_nodes = [node for node, degree in G.out_degree() if degree == 0]
    for i, leaf_node in enumerate(leaf_nodes):
        nx.relabel_nodes(G, {leaf_node: leaf_names[i]}, copy=False)
        G.nodes[leaf_names[i]]['color'] = 'green'

    return G.to_undirected()


def generate_alphabetic_labels(n):
    """
    Generates n alphabetic labels starting with A-Z, then AA-AZ, BA-BZ, etc.

    Parameters
    ----------
    n: int
        The number of labels to generate

    Returns
    -------
    list
        The generated labels
    """
    labels = list(ascii_uppercase)

    current_length = 1
    while len(labels) < n:
        current_length += 1
        for label in product(ascii_uppercase, repeat=current_length):
            labels.append(''.join(label))
            if len(labels) == n:
                break

    return labels[:n]

def get_tensor_encoding(color):
    """
    Returns the tensor encoding of the given color.

    Parameters
    ----------
    color: str
        The color to encode

    Returns
    -------
    torch.Tensor
        The tensor encoding
    """
    if color == 'green':
        return torch.Tensor([1])
    elif color == 'blue':
        return torch.Tensor([2])
    elif color == 'white':
        return torch.Tensor([0])
    else:
        raise ValueError(f"Invalid color: {color}")
