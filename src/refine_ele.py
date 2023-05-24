"""
Fabian Kind
Hochschule Bonn-Rhein-Sieg
Institut für Elektrotechnik, Maschinenbau und Technikjournalismus
Masterprojekt 1
Adaptive Mesh Refinement
"""
import time

import numpy as np
from marking_ele import marking_ele
from bcs_read import bcs_read


class AMR(marking_ele):
    """
    This is the main class for executing the adaptive mesh refinement based on the RGB refinement strategy
    """

    def __init__(self, path, out_path, thickness):
        self.ele_list = []
        self.marked_ele = []

        super().__init__(path, out_path, thickness)

        self.all_edges = None
        self.green_neighbor = []

        self.for_red_ref = []
        self.for_green_ref = []
        self.for_blue_ref_one_neighbor = []
        self.for_blue_ref_two_neighbor = []
        self.green_marked_neighbor = []
        self.blue_marked_neighbor = []
        self.two_blue_marked_neighbor = []
        self.nodes_along_second_neighbor = []

        self.red_ele = []
        self.green_ele = []
        self.blue_ele = []

        self.blue_ele_edges = []
        self.green_ele_edges = []
        self.blue_longest_edge = []

        self.all_ele = []

        self.bcs_mesh = None

    def run_marking(self):
        """
        Run marking algorithm

        @return:
        """
        super().run_marking()

    def nodes_array(self, ele):
        """
        Returns an array of nodes which are marked.

        @return: nodes_array
        """

        nodes = self.ele_undeformed[:, 0:3]
        nodes = nodes[ele].astype(np.int)
        nodes_array = np.asarray(nodes).reshape(-1, 3)

        return nodes_array

    @staticmethod
    def get_all_edges(nodes_array):
        """
        Create a ndarray with three edge tuple.

        @param nodes_array:
        @return: edges
        """

        edges = [nodes_array[:, [0, 1]], nodes_array[:, [1, 2]], nodes_array[:, [2, 0]]]
        return edges

    @staticmethod
    def create_stacked_edges_array(edges):
        """
        Create a stacked array with corresponding element numbers.

        @param edges:
        @return: all_neighbor
        """

        all_neighbor = np.stack([edges[0], edges[1], edges[2]], axis=1)

        return all_neighbor

    def long_stacked_edges_array(self, ele):
        """
        Creats an extended version of tuple nodes with a corresponcing column for element numbers.

        @param ele:
        @return: all_neighbor_long
        """

        index = np.repeat(ele, 3)
        all_edges = self.nodes_array(ele)
        all_edges = self.get_all_edges(all_edges)

        edges = []
        for i in range(len(all_edges[0])):
            edges.append(all_edges[0][i])
            edges.append(all_edges[1][i])
            edges.append(all_edges[2][i])

        all_neighbor_long = np.c_[index, edges]

        return all_neighbor_long

    def get_ele_length(self, marked_ele):
        """
        This function calls all function to calculate the longest edge and checks where it is.

        @return:nodes_where_longest
        """
        nodes_mesh = self.neighbor_coordinates(marked_ele)
        longest_edge = AMR.ele_edge_length(nodes_mesh)
        nodes_where_longest = self.find_longest_edge(longest_edge)

        return nodes_where_longest

    def neighbor_coordinates(self, marked_ele):
        """
        Get all the coordinates of the nodes and split the array in a stack of 3 lists.

        @param marked_ele:
        @return: nodes_mesh
        """

        neighbor_nodes = self.ele_undeformed[marked_ele, 0:3].astype(np.int)

        for idx, add in enumerate(neighbor_nodes):
            neighbor_nodes[idx] = np.array(list(map(lambda x: x - 1, add)))
        mesh = self.mesh_undeformed[:, 0:3]
        nodes_mesh = []

        for nodes in neighbor_nodes:
            nodes_mesh.append(
                [mesh[nodes[0], 0:3], mesh[nodes[1], 0:3], mesh[nodes[2], 0:3]]
            )

        nodes_mesh = np.asarray(nodes_mesh)
        nodes_mesh = np.split(nodes_mesh, 3, axis=1)

        for flatten in range(3):
            nodes_mesh[flatten] = np.concatenate(nodes_mesh[flatten])

        return nodes_mesh

    @staticmethod
    def ele_edge_length(nodes_mesh):
        """
        Calculating the maximum side length of an element. The euclidian norm is used to perform this operation.
        Afterwards the maximum side length with the corresponding index are calculated. It's possible to determine
        the longest side by comparing the index with the calculation of the euclidian norm.

        @param nodes_mesh:
        @return: longest_edge
        """

        euc_dist = [np.array([]), np.array([]), np.array([])]

        # euc_dist[0] = np.linalg.norm((nodes_mesh[0] - nodes_mesh[1]), axis=1)
        # euc_dist[1] = np.linalg.norm((nodes_mesh[1] - nodes_mesh[2]), axis=1)
        # euc_dist[2] = np.linalg.norm((nodes_mesh[2] - nodes_mesh[0]), axis=1)
        euc_dist[0] = np.linalg.norm((nodes_mesh[0] - nodes_mesh[1]), axis=1)
        euc_dist[1] = np.linalg.norm((nodes_mesh[0] - nodes_mesh[2]), axis=1)
        euc_dist[2] = np.linalg.norm((nodes_mesh[1] - nodes_mesh[2]), axis=1)

        euc_dist = np.c_[euc_dist[0], euc_dist[1], euc_dist[2]]

        longest_edge = []
        for edges in euc_dist:
            longest_edge.append(np.argmax(edges, axis=0))
        longest_edge = [np.argmax(euc_dist[i], axis=0) for i in range(len(euc_dist))]

        return longest_edge

    def find_longest_edge(self, longest_edge):
        """
        This function checks if the longest edge is along the marked neighbors edge or not.

        @param longest_edge:
        @return: nodes_where_longest
        """

        nodes = self.ele_undeformed[:, 0:3]
        nodes_where_longest = []
        for n, le in zip(nodes, longest_edge):
            try:
                if le == 0:
                    nodes_where_longest.append(n[[0, 1]])
                if le == 1:
                    nodes_where_longest.append(n[[0, 2]])
                if le == 2:
                    nodes_where_longest.append(n[[1, 2]])

            except ValueError:
                print("Something went wrong while checking for the longest edge...")
                raise

        nodes_where_longest = np.asarray(nodes_where_longest)
        return nodes_where_longest

    @staticmethod
    def neighbor_intersection(row, check_nodes):
        """
        Creates 3 different tuples based on the first and second node, the first and last node and the second and last
        node. The np.isin function checks whether the tuple nodes are somewhere in the check_nodes. If so, they occur
        two times in the match collection. There are 3 different lists of results, which are stored in the match
        collection (three different results for three different edges).
        A list comprehension determines if a number occurs more than 1 time. If so, this number is the direct neighbor
        bacause it shares one edge with one of the tuples.

        @param row:
        @param check_nodes:
        @return: all_neighbor
        """

        templates = [tuple(row[[0, 1]]), tuple(row[[0, 2]]), tuple(row[[1, 2]])]

        match_collection = []
        for idx, tuple_ele in enumerate(templates):
            match_collection.append(np.where(np.isin(check_nodes, tuple_ele)))
        all_neighbor = []
        for match in match_collection:
            match = match[0].tolist()
            all_neighbor.append(list(set([x for x in match if match.count(x) > 1])))

        return all_neighbor

    @staticmethod
    def swap_neighbor(all_neighbor):
        """
        Swap the axes, that the neighbor is in column 1 and the marked element in column 0.

        @param all_neighbor:
        @return: all_neighbor
        """

        find_marked_ele = np.concatenate(all_neighbor).copy().tolist()
        sort = list(set([i for i in find_marked_ele if find_marked_ele.count(i) == 3]))
        indices = np.where(all_neighbor == sort)
        print(indices)
        col_index = indices[1]

        for swap in range(3):
            if col_index[swap] == 1:
                all_neighbor[swap, [0, 1]] = all_neighbor[swap, [1, 0]]

        return all_neighbor

    def all_marked_elements(self):
        """
        Concatenate all marked elements.
        """
        if self.for_blue_ref_one_neighbor:
            if self.for_blue_ref_two_neighbor:
                if self.for_green_ref:
                    self.all_ele = (
                            self.for_blue_ref_one_neighbor
                            + self.for_blue_ref_two_neighbor
                            + self.for_red_ref
                            + self.for_green_ref
                    )

        else:
            self.all_ele = self.for_red_ref

        self.all_ele = np.asarray(self.all_ele).reshape(-1, 1)

    def get_ele_dictionary(self, hanging_edges, all_edges):

        ele_dict = {}
        ele_num = 0
        for i, edge in enumerate(all_edges[:, 1::]):
            element_val = tuple(edge)
            ele_dict[element_val] = {"Edge": element_val, "Ele_number": ele_num, "Marked": False, "Longest_edge":False}

            if (i + 1) % 3 == 0:
                ele_num += 1

        for marked_lst in hanging_edges[:, 1::]:
            marked_lst = tuple(marked_lst)
            if marked_lst in ele_dict:
                ele_dict[marked_lst]["Marked"] = True
        filtered_dict = {key: value for key, value in ele_dict.items() if value["Marked"]}
        return ele_dict

    def elements_to_refine(self, ele_dict, nodes_where_longest, all_edges, hanging_edges):
        """
        This is the main function for determining the elements which have to be refined. The algorithm needs the
        marked elements from the class marking_ele and their corresponding edges. These edges are hanging edges, because
        they all have middle nodes which can have hanging nodes. In the first iteration we check, if unmarked
        elementes are adjacent to marked elements. If so, we mark all adjacent edges to marked elements.
        There are 5 possible outcomes:

        1. 3 edges are marked -> Red element
        2. 2 edges are marked and one edge is along the longest edge of the element -> Blue element
        3. 2 edges are marked and no edge is along the longest edge of the element -> Red element
        4. 1 edge is marked which is the longest edge of the element -> Green element
        5. 1 edge is marked which is not the longest edge of the element -> Blue element

        Option 3 and 5 create new hanging nodes which have to be eliminated before ending the algorithm.
        There are no more refinements if an element is a green marked element.

        @param marked_edges:
        @param all_edges:
        @param nodes_where_longest:
        @return:
        """
        marked_counts = {}
        for it, edges in enumerate(hanging_edges[:, 1::]):
            #print(it)
            ele_neighbor = np.where(
                (all_edges[:, 1::] == edges[::-1]).all(axis=1)
            )[0]

            if ele_neighbor:
                #print(ele_dict[tuple(all_edges[ele_neighbor[0], 1::])])
                if not ele_dict[tuple(all_edges[ele_neighbor[0], 1::])]["Marked"]:
                        longest_neighbor_edge = tuple(nodes_where_longest[all_edges[ele_neighbor[0], 0]])
                        edge_chain = edges
                        while True:
                            if np.isin(longest_neighbor_edge, edge_chain).all():
                                ele_dict[longest_neighbor_edge]["Marked"] = True
                                ele_dict[longest_neighbor_edge]["Longest_edge"] = True
                                #print(ele_dict[longest_neighbor_edge])
                                break
                            else:
                                #print(ele_dict[longest_neighbor_edge])
                                if not ele_dict[longest_neighbor_edge]["Marked"]:
                                    ele_dict[longest_neighbor_edge]["Marked"] = True

                                    edge_chain = longest_neighbor_edge
                                    ele_neighbor = np.where(
                                        (all_edges[:, 1::] == longest_neighbor_edge[::-1]).all(axis=1)
                                    )[0]
                                    longest_neighbor_edge = tuple(nodes_where_longest[all_edges[ele_neighbor[0], 0]])
                                else:
                                    break
                        """        
                        edge_chain = edges[::-1]
                        while True:
                            if not np.array_equal(longest_neighbor_edge, edge_chain):
                                if not ele_dict[longest_neighbor_edge]["Marked"]:
                                    print(ele_dict[longest_neighbor_edge])
                                    ele_dict[longest_neighbor_edge]["Marked"] = True
                                    print(ele_dict[longest_neighbor_edge])

                                    edge_chain = longest_neighbor_edge[::-1]
                                    ele_neighbor = np.where(
                                        (all_edges[:, 1::] == longest_neighbor_edge[::-1]).all(axis=1)
                                    )[0]
                                    longest_neighbor_edge = tuple(nodes_where_longest[all_edges[ele_neighbor[0], 0]])
                                else:
                                    break
                            else:
                                ele_dict[longest_neighbor_edge]["Marked"] = True
                                ele_dict[longest_neighbor_edge]["Longest_edge"] = True
                                print(ele_dict[longest_neighbor_edge])
                                break
                        """

        for val in ele_dict.values():
            ele_number = val["Ele_number"]
            marked = val["Marked"]
            longest_edge = val["Longest_edge"]
            if longest_edge:
                pass

            if ele_number not in marked_counts:
                marked_counts[ele_number] = {"Ele_number": ele_number, "Count": 0, "Longest_edge": longest_edge}
            if marked:
                marked_counts[ele_number]["Count"] += 1
            if longest_edge:
                marked_counts[ele_number]["Longest_edge"] = True

        print(
            "Marked {} blue elements with only one neighbor".format(
                len(self.for_blue_ref_one_neighbor)
            )
        )
        print("Marked {} green elements".format(len(self.for_green_ref)))
        print(
            "Marked {} blue elements with two neighbor".format(
                len(self.for_blue_ref_two_neighbor)
            )
        )
        print("Marked {} red elements".format(len(self.for_red_ref)))
        print('----------------------------------------------------')

    def check_for_wrong_assignement(self,
                                    longest_edge,
                                    ele_num,
                                    blue_longest_edges,
                                    long_ele

                                    ):
        """

        @return:
        """
        tic = time.perf_counter()
        for idx, arr_blue in enumerate(blue_longest_edges):
            # match = np.where((long_ele[:, 1::] == arr_blue).all(axis=1))[0]
            # entry_match = np.any(np.all(np.sort(long_ele[:, 1::], axis=1) == np.sort(arr_blue), axis=1))
            matches = np.isin(long_ele[:, 1::], arr_blue).all(axis=1)
            index = np.where(matches)[0]

            for col_index in index:
                if long_ele[col_index, 2] == arr_blue[1]:
                    if col_index in self.for_blue_ref_one_neighbor:
                        print('False')

        print(time.perf_counter() - tic)
        """
        for idx, arr_green in enumerate(self.green_ele_edges):
            for row_green in arr_green:
                if np.array_equal(row_green, longest_edge):
                    print('green', idx)
                    self.for_blue_ref_two_neighbor.append(
                        self.for_green_ref[idx]
                    )

                    self.two_blue_marked_neighbor.append(
                        [
                            self.green_marked_neighbor[idx],
                            ele_num

                        ]
                    )

                    del self.for_green_ref[idx]
                    del self.green_marked_neighbor[idx]
                    del self.green_ele_edges[idx]

        """

    @staticmethod
    def get_marked_neighbor(edge_index, check_edges, long_ele):
        """
        Get the neighbor of the freshly marked element. edge_index is the index where the tuple match occured.
        It can be 0,1 or 2. Check edges are the edges we checked, which are 3 tuples. Long_ele is a list
        of all mesh tuples with their corresponding node numbers. Neighboring edges swap their node rotation,
        meaning a neighbor tuple matches f.e. if Element 1: [523, 87]; Element 2: [87. 523].
        Swaping the nodes and using np.where on the long_ele array returns a index list where a match occurs.
        If we knoe the index, we also know the element numbers because they are along the first column of the
        long_ele array.

        @param edge_index:
        @param check_edges:
        @param long_ele:
        @return:
        """

        match_index = []
        for row in edge_index:
            swap_nodes = [check_edges[row][1], check_edges[row][0]]
            match_index.append(
                np.where((long_ele[:, 1::] == swap_nodes).all(axis=1))[0]
            )
        return match_index

    def get_edges_along_blue_elements(self):
        """
        Get all edges along the already marked elements of a blue element with two neighbors.
        """

        self.two_blue_marked_neighbor = np.asarray(self.two_blue_marked_neighbor)
        first_neighbor = self.nodes_array(self.two_blue_marked_neighbor[:, 0])
        second_neighbor = self.nodes_array(self.two_blue_marked_neighbor[:, 1])
        marked_ele = self.nodes_array(self.for_blue_ref_two_neighbor)

        for (fn, sn, me,) in zip(first_neighbor, second_neighbor, marked_ele, ):
            self.nodes_along_second_neighbor.append(np.intersect1d(fn, me))
            self.nodes_along_second_neighbor.append(np.intersect1d(sn, me))

        self.two_blue_marked_neighbor = self.two_blue_marked_neighbor.tolist()

    def mid_nodes(self, ele):
        """
        Calculation of all mid nodes. Afterwards a template will be created with new mid node numbers and their
        corresponding coordinates.
        @param ele:
        @return: mid_node_coor
        """

        coors = self.neighbor_coordinates(ele)
        mid_node = [np.array([]), np.array([]), np.array([])]

        mid_node[0] = np.divide((np.add(coors[0], coors[1])), 2).round(decimals=6)

        mid_node[1] = np.divide((np.add(coors[1], coors[2])), 2).round(decimals=6)

        mid_node[2] = np.divide((np.add(coors[2], coors[0])), 2).round(decimals=6)

        for col in range(3):
            mid_node[col] = np.split(np.concatenate(mid_node[col]), len(mid_node[col]))

        mid_node_coor = np.hstack((mid_node[0], mid_node[1], mid_node[2])).reshape(
            len(mid_node[0]) * 3, 3
        )

        unique_mesh, idx = np.unique(mid_node_coor, axis=0, return_index=True)
        node_axis = np.arange(
            len(self.mesh_undeformed) + 1,
            len(self.mesh_undeformed) + len(unique_mesh) + 1,
        ).astype(np.int)

        # Keeping the ordner because np.unique sorts by size (Not necessary but
        # keeps the chronologically order)
        bcs_mesh = unique_mesh[np.argsort(idx)]
        self.bcs_mesh = np.hstack((node_axis[:, np.newaxis], bcs_mesh))

        return mid_node_coor

    def find_matching_mid_node(self, mid_nodes_coors, shape):
        """
        This function works like a template. self.bcs_mesh contains all new generated mid node numbers and their
        corresponding coordinates. The mid_nodes_coors variable contains all coordinates of the mid nodes which
        are needed for the green, red or blue refinement. This allows to check where the coordinates of the template
        and the mid_nodes_coors match. F.e.:

        Template (self.bcs_mesh)| mid_node_coors
        ------------------------|--------------
        5241 x1 y1 z1           | x1 y1 z2 (matches column 1 to 3)
        5242 x2 y2 z2           | x3 y3 z3 (no match)

        ---> Take node number 5241 and continue

        @param mid_nodes_coors:
        @param shape:
        @return: mid_nodes, no_match
        """

        idx_cluster = []
        no_match = []
        for idx, coors in enumerate(mid_nodes_coors):
            matching_c = np.where((self.bcs_mesh[:, 1:4] == coors).all(axis=1))[0]

            if len(matching_c) > 0:
                idx_cluster.append(matching_c[0])
            elif matching_c or matching_c == 0:
                idx_cluster.append(matching_c)
            else:
                no_match.append(idx)
        if shape:
            if np.mod(len(idx_cluster), shape) == 0:
                mid_nodes = np.asarray(
                    [self.bcs_mesh[idx, 0].astype(np.int) for idx in idx_cluster]
                ).reshape(-1, shape)
            else:
                raise ValueError(
                    "Something went wrong while trying to find the mid nodes for the refinement"
                )
        else:
            mid_nodes = np.asarray(
                [
                    self.bcs_mesh[idx, 0].astype(np.int)
                    for idx in idx_cluster
                    if idx or idx == 0
                ]
            )

        return mid_nodes, no_match

    def calculate_mid_node(self, match, container_len):
        """
        This function calculates the mid node coordinate at the edges where the longest edge is.

        @param match:
        @param container_len:
        @return: mid_node
        """

        c_container = [
            np.array(np.empty((container_len, 3))),
            np.array(np.empty((container_len, 3))),
        ]

        match = list(map(lambda x: x - 1, match))

        for i in range(2):
            for idx, nodes in enumerate(match):
                c_container[i][idx] = self.mesh_undeformed[nodes[i].astype(np.int), 0:3]

        mid_node = np.divide((np.add(c_container[0], c_container[1])), 2).round(
            decimals=6
        )

        return mid_node

    @staticmethod
    def keep_rotation_direction(nodes_neighbor, nodes, nodes_where_longest, ele):
        """
        Neighboring nodes change their order to keep the rotation direction. Therefore it's very important
        to place the nodes at the right position, because they differ depending on the neighbor node position.

        1. Get the index of the intersection between the nodes neighbors (which are the nodes of a marked elements
           neighbors) and nodes (which are the nodes of the newly marked element f.e. a green element).
        2. There is always one node in the nodes variable which has no intersection, because there's only an
           intersection between an adjacent edge. This node is the "node to keep" because it's the vertex node
           opposite of the marked edge.
        3. Get the correct nodes rotation by calling the function nodes_rotation. Follow the docstrings of
           nodes_rotation for more insights.
        4. Only for blue elements with one neighbor: The vertex node is the node which is on the opposite of
           edge, where the adjacent marked element is.

        @param nodes_neighbor:
        @param nodes:
        @param nodes_where_longest:
        @param ele:
        @return: keep_node, index, keep_node_index, node_to_close_element, node_rotation
        """

        idx1 = []
        idx2 = []
        keep_node = []
        keep_node_index = []
        nodes = np.asarray(nodes)
        for index, row in enumerate(zip(nodes_neighbor, nodes)):
            intersection, _, indices = np.intersect1d(
                row[0], row[1], return_indices=True
            )
            idx1.append(np.where(row[0] == intersection[0]))
            idx2.append(np.where(row[0] == intersection[1]))

            keep_node.append(np.setxor1d(intersection, row[1])[0])

            keep_node_index.append(
                np.where(keep_node[index] == row[1])[0][0].astype(np.int)
            )

        index = np.concatenate(np.c_[idx1, idx2])

        # Get the nodes rotation by calling the function nodes_roation
        node_rotation = AMR.nodes_rotation(keep_node_index, nodes)

        vertex_node = []
        le = [nodes_where_longest[idx] for idx in ele]
        for idx, elements in enumerate(le):
            vertex_node.append(int(np.setxor1d(elements, nodes[idx])[0]))

        return keep_node, index, keep_node_index, vertex_node, node_rotation

    @staticmethod
    def nodes_rotation(keep_node_index, nodes):
        """
        Right-hand side rule for shell elements is responsible for the nodes rotation. If we know the index
        of the node to keep and the original order of the elements node, we know how the nodes rotation is.

        @param keep_node_index:
        @param nodes:
        return: node_rotation
        """

        node_rotation = []
        for idx, row in enumerate(keep_node_index):
            if row == 1:
                node_rotation.append(np.array((nodes[idx, 2], nodes[idx, 0])))
            elif row == 2:
                node_rotation.append(np.array((nodes[idx, 0], nodes[idx, 1])))
            else:
                node_rotation.append(np.array((nodes[idx, 1], nodes[idx, 2])))

        return node_rotation

    def find_vertex_and_mid_node(
            self, nodes, neighbor_one, neighbor_two, two_neighbor, mid_node_with_le
    ):
        """
        Find the node that closes the element and the corresponding vertex node. This is important because
        otherwise it's unclear which mid node should be used to create the blue element with two neighbors.
        This function extends the nodes_rotation and keep_rotation_direction function because we need to keep
        two neighboring elements into account.

        1.Get the nodes which are NOT along one of the marked neighbors of the blue element.
        2.Get the node which connects the edges which are along the marked neighbors. This node is the vertex
          node of the opposite side if the edge in 1.
        3.Get the nodes rotation with the connecting node
        4.Check whether the unmarked edge intersects with the node which is in the longest edge and get
        the nodes which bisect both edges where the neighbors are. One of them is the vertex node of the connecting
        node and the other one is the bisecting node of the second eedge where a neighbor is.

        @param nodes:
        @param neighbor_one:
        @param neighbor_two:
        @param two_neighbor:
        @param mid_node_with_le:
        @return: rotation_direction, node_to_close, vertex_node
        """

        unmarked_edge = []
        connecting_node = []
        connecting_node_index = []
        for row in range(len(nodes)):
            unmarked_edge.append(
                np.setxor1d(neighbor_one[row], neighbor_two[row]).astype(np.int)
            )

            connecting_node.append(np.setxor1d(unmarked_edge[row], nodes[row])[0])

            connecting_node_index.append(
                np.where(connecting_node[row] == nodes[row])[0][0]
            )

        rotation_direction = self.nodes_rotation(connecting_node_index, nodes)

        bisect_node = []
        vertex_node = []
        for edge in range(len(connecting_node_index)):
            longest_edge_to_node = np.intersect1d(
                rotation_direction[edge][0], mid_node_with_le[1][edge]
            )

            if longest_edge_to_node:
                bisect_node.append(two_neighbor[edge][0])
                vertex_node.append(two_neighbor[edge][1])
            else:
                bisect_node.append(two_neighbor[edge][1])
                vertex_node.append(two_neighbor[edge][0])

        return rotation_direction, bisect_node, vertex_node, connecting_node

    def mid_node_one_neighbor(self, neighbor, nodes_where_longest, marked_ele):
        """
        Get the nodes along the longest edge and the nodes along the marked_neighbor.

        @param neighbor:
        @param nodes_where_longest:
        @param marked_ele:
        @return:
        """

        longest_edge = [nodes_where_longest[index] for index in neighbor]

        neighbor_ele_edge = self.nodes_array(neighbor)
        marked_ele_edge = self.nodes_array(marked_ele)
        nodes_along_neighbor = []
        for ne, me in zip(neighbor_ele_edge, marked_ele_edge):
            nodes_along_neighbor.append(np.intersect1d(ne, me))

        return nodes_along_neighbor, longest_edge

    def get_mid_nodes(self, *args, shape):
        """
        Call function calculate_mid_node and find_matching_mid_node and return the matching mid nodes.

        *args can hold:
        1. longest edge
        2. mid_node_coors

        @param shape:
        @return:
        """

        if args[1] is None:
            mid_node_coors = self.calculate_mid_node(args[0], len(args[0]))
        else:
            mid_node_coors = args[1]

        mid_node, no_match = self.find_matching_mid_node(mid_node_coors, shape=shape)

        if no_match:
            raise ValueError(
                "Could not reference all marked elements to a corresponding mid node"
            )

        return mid_node

    def red_pattern(self, mid_nodes_coor, ele):
        """
        Creates a pattern for the red refined elements. First of all we use the list of unique elements (bcs_mesh)
        as a reference, because it includes a fresh generated axis of element numbers and the
        corresponding coordinates. It is not necessary to define new node numbers
        because the middle nodes of green and blue elements are connecting middle nodes of other elements.
        It is also necessary to implement the new pattern of the element.

        @param ele:
        @param mid_nodes_coor:
        @return:
        """

        mid_nodes = self.get_mid_nodes(None, mid_nodes_coor, shape=3)

        nodes = self.nodes_array(ele)

        for enum, (mn, ne) in enumerate(zip(mid_nodes, nodes)):
            self.red_ele.append(np.array((mn[0], mn[1], mn[2])))
            self.red_ele.append(np.array((ne[2], mn[2], mn[1])))
            self.red_ele.append(np.array((mn[2], ne[0], mn[0])))
            self.red_ele.append(np.array((mn[0], ne[1], mn[1])))

    def green_pattern(self, nodes_where_longest, ele):
        """
        There are two main operations in this function. The first loop searches the two connected nodes with the longest
        edge in the element. The function call self.find_matching_mid_nodes checks whether the mid node of the longest
        edge is present in the bcs_mesh template. If so, the green element is a neighbor of a red element. If not, it
        is the neighbor of a blue element.

        @param nodes_where_longest:
        @param ele:
        @return:green_ele
        """

        longest_edge = [nodes_where_longest[index] for index in self.for_green_ref]

        mid_node = self.get_mid_nodes(longest_edge, None, shape=None)

        nodes = self.nodes_array(ele)
        nodes_neighbor = self.nodes_array(self.green_marked_neighbor)
        keep_node, _, _, _, nodes_longest_edge = AMR.keep_rotation_direction(
            nodes_neighbor, nodes, nodes_where_longest, ele
        )

        for count, (nle, mn) in enumerate(zip(nodes_longest_edge, mid_node)):
            self.green_ele.append(np.array((mn, keep_node[count], nle[0])))
            self.green_ele.append(np.array((mn, nle[1], keep_node[count])))

    def stack_mid_nodes(self, longest_edge, nodes_along_neighbor):
        """
        This function stacks the mid nodes for blue elements.

        @param longest_edge:
        @param nodes_along_neighbor:
        @return:
        """

        mid_node_le = self.get_mid_nodes(longest_edge, None, shape=None)
        mid_node_nle = self.get_mid_nodes(nodes_along_neighbor, None, shape=None)

        try:
            mid_nodes = np.c_[mid_node_le, mid_node_nle]
        except ValueError:
            print(
                "Shape mismatch in longest edge and not longest edge in the blue element cluster"
            )
            raise

        return mid_nodes

    def blue_pattern_one_neighbor(self, marked_ele, neighbor_ele, nodes_where_longest):
        """
        Similar approach than the green pattern function. Here it is important to split the blue refinement because
        it can base on one marked neighbors or two.

        @param: not_longest_edge
        @param: neighbor
        @return:
        """

        nodes_along_neighbor, longest_edge = self.mid_node_one_neighbor(
            marked_ele, nodes_where_longest, neighbor_ele
        )[:5]

        mid_nodes = self.stack_mid_nodes(longest_edge, nodes_along_neighbor)

        self.create_blue_pattern_one_neighbor(
            mid_nodes, marked_ele, self.blue_marked_neighbor, nodes_where_longest
        )

    def blue_pattern_two_neighbor(self, longest_edge, ele_two_neighbor):
        """
        This function gets the mid nodes for blue elements with two neighbors.

        @param longest_edge:
        @param ele_two_neighbor:
        @return:
        """

        self.get_edges_along_blue_elements()

        nodes_nle = []
        longest_edge = [longest_edge[index] for index in self.for_blue_ref_two_neighbor]
        for idx, nodes in enumerate(self.nodes_along_second_neighbor):
            result = np.isin(longest_edge, nodes).all(axis=1)
            if not any(result):
                nodes_nle.append(nodes)

        mid_nodes = self.stack_mid_nodes(longest_edge, nodes_nle)
        mid_node_with_le = [mid_nodes[:, 0], longest_edge]

        self.create_blue_pattern_two_neighbor(
            mid_nodes,
            ele_two_neighbor,
            self.nodes_along_second_neighbor,
            mid_node_with_le,
        )

    def create_blue_pattern_one_neighbor(
            self, mid_nodes, ele, neighbor, nodes_where_longest
    ):
        """
        This function creates the blue pattern for elements which have one or two neighbors.

        @param mid_nodes:
        @param ele:
        @param neighbor:
        @param nodes_where_longest:
        @return:
        """
        nodes = self.nodes_array(ele)
        nodes_neighbor = self.nodes_array(neighbor)
        (
            keep_node,
            index,
            keep_node_index,
            vertex_node,
            node_rotation,
        ) = AMR.keep_rotation_direction(nodes_neighbor, nodes, nodes_where_longest, ele)
        keep_node = keep_node
        for count, (nr, mn) in enumerate(zip(node_rotation, mid_nodes)):
            if vertex_node[count] == nr[0]:
                self.blue_ele.append(
                    np.array(
                        (
                            keep_node[count],
                            vertex_node[count],
                            mn[1],
                        )
                    )
                )
                self.blue_ele.append(np.array((mn[1], nr[1], mn[0])))
                self.blue_ele.append(np.array((mn[0], keep_node[count], mn[1])))
            else:
                self.blue_ele.append(
                    np.array(
                        (
                            keep_node[count],
                            mn[1],
                            vertex_node[count],
                        )
                    )
                )
                self.blue_ele.append(np.array((mn[1], mn[0], nr[0])))
                self.blue_ele.append(np.array((mn[0], mn[1], keep_node[count])))

    def create_blue_pattern_two_neighbor(
            self, two_neighbor, ele, nodes_along_second_neighbor, mid_node_with_le
    ):
        """
        This function creates the blue pattern for elements which have one or two neighbors.
        @param two_neighbor:
        @param nodes_along_second_neighbor:
        @param ele:
        @param mid_node_with_le:
        @return:
        """

        nodes = self.ele_undeformed[ele, 0:3]
        neighbor_one = nodes_along_second_neighbor[0: len(ele) * 2: 2]
        neighbor_two = nodes_along_second_neighbor[1: len(ele) * 2: 2]

        (
            unmarked_edge,
            node_to_close,
            vertex_node,
            keep_node,
        ) = self.find_vertex_and_mid_node(
            nodes, neighbor_one, neighbor_two, two_neighbor, mid_node_with_le
        )

        for count, (node_to_close, vertex_node) in enumerate(
                zip(node_to_close, vertex_node)
        ):
            self.blue_ele.append(
                np.array(
                    (unmarked_edge[count][0], unmarked_edge[count][1], node_to_close)
                )
            )
            self.blue_ele.append(
                np.array((node_to_close, unmarked_edge[count][1], vertex_node))
            )
            self.blue_ele.append(
                np.array((keep_node[count], node_to_close, vertex_node))
            )

    def get_longest_edge(self):
        """
        Calculate the longest edge of all elements.
        @return: nodes_where_longest, all_edges, marked_edges
        """

        all_ele = np.arange(0, len(self.ele_undeformed))
        nodes_array = self.nodes_array(self.marked_ele)
        marked_edges = self.get_all_edges(nodes_array)
        all_edges = self.long_stacked_edges_array(all_ele)
        nodes_where_longest = self.get_ele_length(all_ele)

        return nodes_where_longest, all_edges, marked_edges

    def find_elements_to_refine(self, marked_edges, all_edges, nodes_where_longest):
        """
        Main function for the RGB-refinement and to determine the new mid nodes of
        red and blue elements.

        @param marked_edges:
        @param all_edges:
        @param nodes_where_longest:
        @return:
        """
        hanging_edges = self.long_stacked_edges_array(marked_edges)
        ele_dict = self.get_ele_dictionary(hanging_edges, all_edges)
        self.elements_to_refine(ele_dict, nodes_where_longest, all_edges, hanging_edges)
        mid_node_coors = self.mid_nodes(
            self.for_red_ref + self.for_blue_ref_one_neighbor
        )

        return mid_node_coors

    def create_all_pattern(self, mid_node_coors, nodes_where_longest):
        """
        This function concatenates all pattern creations

        @param mid_node_coors:
        @param nodes_where_longest:
        @return:
        """

        self.red_pattern(mid_node_coors, self.for_red_ref)
        self.green_pattern(nodes_where_longest, self.for_green_ref)
        self.blue_pattern_one_neighbor(
            self.for_blue_ref_one_neighbor,
            self.blue_marked_neighbor,
            nodes_where_longest,
        )

        self.blue_pattern_two_neighbor(
            nodes_where_longest, self.for_blue_ref_two_neighbor
        )

    def main_amr(self):
        """
        Main function
        @return:
        """

        self.run_marking()
        nodes_where_longest, all_edges, marked_edges = self.get_longest_edge()
        mid_node_coors = self.find_elements_to_refine(
            marked_edges, all_edges, nodes_where_longest
        )
        self.create_all_pattern(mid_node_coors, nodes_where_longest)
