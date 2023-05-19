"""
This source code provides three functions for an easy neighbor search. It's a working code if it get's
implemented in the right surrounding and can be used for further marking stretgies.

F.e. you could use this code to provide all neighbors of a marked element. Compare the neighbors thickness
with the marked neighbor and mark the neighbors if the thickness difference is too big.
F.e. you could calculate all normal vectors of neighbots of marked elements and further calculate the
angle difference between the vectors.
"""

import numpy as np


def direct_neighbor(self, ele):
    """
    See the find_intersection functions docstring for more information.

    After determining the direct neighbor, a loop checks whether the direct neighbor of the marked elements
    is somewhere in the list of marked elements or if it's an element that has toi be green/blue refined.
    @return:direct_neighbor, marked_ele
    """

    nodes = self.nodes_array(ele)
    check_nodes = self.ele_undeformed[:, 0:3]

    neighbor_collection = []
    marked_neighbor = []
    three_unmarked = []

    for i, row in enumerate(nodes):

        all_neighbor = np.asarray(find_intersection(row, check_nodes))

        all_neighbor = swap_neighbor(all_neighbor)

        try:
            for idx, list_counter in enumerate(all_neighbor):
                if list_counter[1] not in ele:
                    neighbor_collection.append(list_counter)
                else:
                        marked_neighbor.append(list_counter)
        except ValueError:
            raise "Something went wrong while determining the direct neighbor..."

    direct_neighbor = np.asarray(neighbor_collection)[:, 1]
    marked_ele = np.asarray(neighbor_collection)[:, 0]
    marked_neighbor = np.asarray(marked_neighbor)
    three_unmarked = np.unique(three_unmarked)
    return direct_neighbor, marked_ele, nodes, marked_neighbor


def find_intersection(row, ele_mesh):
    """
    Creates 3 different tuples based on the first and second node, the first and last node and the second and last
    node. The np.isin function checks whether the tuple nodes are somewhere in the check_nodes. If so, they occur
    two times in the match collection. There are 3 different lists of results, which are stored in the match collection
    (Three different results for three different edges).
    A list comprehension determines if a number occurs more than 1 time. If so, this number is the direct neighbor
    bacause it shares one edge with one of the tuples.
    @param row:
    @param ele_mesh:
    @return:
    """
    check_nodes = np.array(ele_mesh)
    templates = [tuple(row[[0, 1]]), tuple(row[[0, 2]]), tuple(row[[1, 2]])]
    match_collection = []
    for idx, tuple_ele in enumerate(templates):
        match_collection.append(np.where(
            np.isin(check_nodes, tuple_ele)
        )
        )
    all_neighbor = []
    for match in match_collection:
        match = match[0].tolist()
        all_neighbor.append(
            list(set([x for x in match if match.count(x) > 1]))
        )

    return all_neighbor


def swap_neighbor(all_neighbor):
    """
    Swap the axes, that the neighbor is in column 1 and the marked element in column 0
    @param all_neighbor:
    @return:
    """
    find_marked_ele = np.concatenate(all_neighbor).copy().tolist()
    sort = list(
        set([i for i in find_marked_ele if find_marked_ele.count(i) == 3])
    )
    indices = np.where(all_neighbor == sort)
    col_index = indices[1]
    for swap in range(3):
        if col_index[swap] == 1:
            all_neighbor[swap, [0, 1]] = all_neighbor[swap, [1, 0]]

    return all_neighbor
