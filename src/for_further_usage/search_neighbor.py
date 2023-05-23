"""
This source code provides three functions for an easy neighbor search. It's a working code if it get's
implemented in the right surrounding and can be used for further marking stretgies.

F.e. you could use this code to provide all neighbors of a marked element. Compare the neighbors thickness
with the marked neighbor and mark the neighbors if the thickness difference is too big.
F.e. you could calculate all normal vectors of neighbots of marked elements and further calculate the
angle difference between the vectors.
"""

import numpy as np


def nodes_array(ele):
    """
    Returns an array of nodes which are marked.

    @return: nodes_array
    """

    nodes = ele_undeformed[:, 0:3]
    nodes = nodes[ele].astype(np.int)
    nodes_array = np.asarray(nodes).reshape(-1, 3)

    return nodes_array


def get_all_edges(nodes_array):
    """
    Create a ndarray with three edge tuple.

    @param nodes_array:
    @return: edges
    """

    edges = [nodes_array[:, [0, 1]], nodes_array[:, [1, 2]], nodes_array[:, [2, 0]]]
    return edges


def long_stacked_edges_array(ele):
    """
    Creats an extended version of tuple nodes with a corresponcing column for element numbers.

    @param ele:
    @return: all_neighbor_long
    """

    index = np.repeat(ele, 3)
    all_edges = nodes_array(ele)
    all_edges = get_all_edges(all_edges)

    edges = []
    for i in range(len(all_edges[0])):
        edges.append(all_edges[0][i])
        edges.append(all_edges[1][i])
        edges.append(all_edges[2][i])

    all_neighbor_long = np.c_[index, edges]

    return all_neighbor_long
