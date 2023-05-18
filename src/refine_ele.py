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

    def create_stacked_edges_array(self, edges):
        """
        Create a stacked array with corresponding element numbers.

        @param ele:
        @return: all_neighbor
        """

        all_neighbor = np.stack([edges[0], edges[1], edges[2]], axis=1)

        return all_neighbor

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

        for i, row in enumerate(nodes):

            all_neighbor = np.asarray(AMR.neighbor_intersection(row, check_nodes))

            all_neighbor = AMR.swap_neighbor(all_neighbor)

            try:
                for idx, list_counter in enumerate(all_neighbor):
                    neighbor_collection.append(list_counter)
            except ValueError:
                raise "Something went wrong while determining the direct neighbor..."

        direct_neighbor = np.asarray(neighbor_collection)[:, 1]

        return direct_neighbor

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

    def elements_to_refine(self, marked_edges, all_edges, nodes_where_longest):
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

        hanging_edges = []
        for i in range(len(marked_edges[0])):
            hanging_edges.append(marked_edges[0][i])
            hanging_edges.append(marked_edges[1][i])
            hanging_edges.append(marked_edges[2][i])

        marked_ele = self.marked_ele
        all_edges = self.create_stacked_edges_array(all_edges)
        hanging_edges = np.asarray(hanging_edges)
        marked_ele = np.repeat(np.array(marked_ele).copy(), 3, axis=0)
        ele_num = np.arange(0, len(self.ele_undeformed))

        while len(hanging_edges) > 0:
            blue_hanging_edges = []
            red_hanging_edges = []
            hold_back_green = []
            hold_back_green_neighbor = []
            blue_marked_ele = []
            red_marked_ele = []
            blue = []
            for idx, check_edges in enumerate(all_edges):
                counter = 0
                edges = []
                check = []
                for edge_counter, bounding_edges in enumerate(check_edges):

                    check.append(
                        np.isin(
                            hanging_edges, bounding_edges
                        ).all(axis=1)
                    )

                    edges.append(bounding_edges)

                    if check[edge_counter].any():
                        counter += 1

                if counter == 3:
                    if ele_num[idx] not in self.for_red_ref:
                        self.for_red_ref.append(ele_num[idx])

                if counter == 2:
                    le = nodes_where_longest[ele_num[idx]]
                    check_for_le = np.isin(hanging_edges, le).all(axis=1)
                    if any(check_for_le):

                        self.get_second_blue_element_neighbor(
                            marked_ele, ele_num, idx, check
                        )

                    else:
                        if ele_num[idx] not in self.all_ele:
                            if ele_num[idx] not in self.for_red_ref:
                                self.for_red_ref.append(ele_num[idx])
                                longest_edge = nodes_where_longest[ele_num[idx]]

                                if ele_num[idx] not in red_marked_ele:
                                    red_marked_ele.append(ele_num[idx])

                                red_hanging_edges.append(longest_edge)

                if counter == 1:
                    le = nodes_where_longest[ele_num[idx]]
                    check_for_le = np.isin(hanging_edges, le).all(axis=1)
                    if any(check_for_le):
                        if ele_num[idx] not in self.all_ele:
                            hold_back_green.append(ele_num[idx])

                            for find_match in check:
                                true_indice = np.where(find_match)[0]
                                if len(true_indice) > 1:
                                    if marked_ele[true_indice[0]] == ele_num[idx]:
                                        hold_back_green_neighbor.append(
                                            marked_ele[true_indice[1]]
                                        )
                                    else:
                                        hold_back_green_neighbor.append(
                                            marked_ele[true_indice[0]]
                                        )

                                else:
                                    if true_indice:
                                        hold_back_green_neighbor.append(
                                            marked_ele[true_indice[0]]
                                        )

                    else:
                        if ele_num[idx] not in self.for_blue_ref_one_neighbor:
                            if ele_num[idx] not in self.all_ele:
                                self.for_blue_ref_one_neighbor.append(ele_num[idx])
                                longest_edge = nodes_where_longest[ele_num[idx]]

                                if ele_num[idx] not in blue_marked_ele:
                                    blue_marked_ele.append(ele_num[idx])

                                blue_hanging_edges.append(longest_edge)

                                for find_match in check:
                                    true_indice = np.where(find_match)[0]
                                    for index in true_indice:
                                        self.blue_marked_neighbor.append(marked_ele[index])
                                    blue.append(ele_num[idx])

            green_edges = self.nodes_array(hold_back_green)
            green_edges = self.get_all_edges(green_edges)

            wrong_index = []

            for tuple_ele in green_edges:
                for index_ele, edges in enumerate(tuple_ele):
                    if blue_hanging_edges:
                        wrong_blue = np.isin(blue_hanging_edges, edges).all(axis=1)
                        if wrong_blue.any():
                            index = np.where(wrong_blue)[0]
                            self.two_blue_marked_neighbor.append(
                                [hold_back_green_neighbor[index_ele],
                                 blue_marked_ele[index[0]]
                                 ]
                            )

                            self.for_blue_ref_two_neighbor.append(
                                hold_back_green[index_ele]
                            )
                    if red_hanging_edges:
                        wrong_red = np.isin(red_hanging_edges, edges).all(axis=1)
                        if wrong_red.any():
                            index = np.where(wrong_red)[0]
                            self.two_blue_marked_neighbor.append(
                                [hold_back_green_neighbor[index_ele],
                                 red_marked_ele[index[0]]
                                 ]
                            )
                            self.for_blue_ref_two_neighbor.append(
                                hold_back_green[index_ele]
                            )

                for green_ele in range(len(hold_back_green_neighbor)):
                    if green_ele not in wrong_index:
                        self.for_green_ref.append(hold_back_green[green_ele])
                        self.green_marked_neighbor.append(hold_back_green_neighbor[green_ele])

            hanging_edges = np.asarray(blue_hanging_edges + red_hanging_edges)
            marked_ele = np.array(blue_marked_ele + red_marked_ele)

            self.all_marked_elements()

        return all_edges

    def neighbors_along_longest_edge(self, all_edges, longest_edges):
        """
        Returns the element along the longest edge of the marked element.
        @param all_edges:
        @param longest_edges:
        @return: stackes_neighbor, ele_num, blue_ele
        """
        ele_num = []
        blue_ele = []
        for index, edges in enumerate(longest_edges):
            for column, col_index in enumerate(range(3)):
                edge_search = np.isin(all_edges[:, col_index], edges).all(axis=1)

                if edge_search.any():
                    ele_index = np.where(edge_search)[0]
                    for ele in ele_index:
                        if not np.array_equal(all_edges[ele, column], edges):
                            ele_num.append(ele_index[0])
                            blue_ele.append(index)

        longest_edge_neighbor_edge = self.nodes_array(ele_num)
        get_edges = AMR.get_all_edges(longest_edge_neighbor_edge)
        stacked_neighbor = self.create_stacked_edges_array(get_edges)

        return stacked_neighbor, ele_num, blue_ele

    def get_second_blue_element_neighbor(self, marked_ele, ele_num, idx, check):
        """
        If an element is blue marked and has two neighbors we need the element numbers of both neighbors
        for further operations. Therefore we get the index of the match of the unmarked edge and the marked
        edge. The marked_ele contains all marked elements and is the same shape as the check variable used in
        the count_occurence function. Therefore we can directly but the index into the marked_ele function
        and get both neighbors.

        @param marked_ele:
        @param ele_num:
        @param idx:
        @param check:
        @return:
        """

        # Append the marked element number to the instance variable "self.for_blue_ref_two_neighbor"
        self.for_blue_ref_two_neighbor.append(ele_num[idx])

        temp_blue_neighbor = []

        for find_match in check:
            true_indice = np.where(find_match)[0]

            if true_indice is not None or true_indice is True or len(true_indice) > 0:
                for index in true_indice:
                    temp_blue_neighbor.append(marked_ele[index])

        self.two_blue_marked_neighbor.append(temp_blue_neighbor)

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

        """
        red_coors = self.neighbor_coordinates(ele)
        mid_node = [np.array([]), np.array([]), np.array([])]

        mid_node[0] = np.divide((np.add(red_coors[0], red_coors[1])), 2).round(
            decimals=6
        )

        mid_node[1] = np.divide((np.add(red_coors[1], red_coors[2])), 2).round(
            decimals=6
        )

        mid_node[2] = np.divide((np.add(red_coors[2], red_coors[0])), 2).round(
            decimals=6
        )

        for col in range(3):
            mid_node[col] = np.split(np.concatenate(mid_node[col]), len(mid_node[col]))

        mid_node_coor = np.stack((mid_node[0], mid_node[1], mid_node[2])).reshape(
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
        """
        red_coors = self.neighbor_coordinates(ele)
        mid_node_1 = np.divide(
            (np.add(red_coors[0], red_coors[1])), 2
        ).round(decimals=6)
        mid_node_2 = np.divide(
            (np.add(red_coors[1], red_coors[2])), 2
        ).round(decimals=6)
        mid_node_3 = np.divide(
            (np.add(red_coors[2], red_coors[0])), 2
        ).round(decimals=6)

        split_size = len(mid_node_1)
        mid_node_1 = np.split(np.concatenate(mid_node_1), split_size)
        mid_node_2 = np.split(np.concatenate(mid_node_2), split_size)
        mid_node_3 = np.split(np.concatenate(mid_node_3), split_size)

        mid_node_coor = np.hstack(
            (mid_node_1, mid_node_2, mid_node_3)
        ).reshape(
            len(mid_node_1) * 3, 3
        )

        unique_mesh, idx = np.unique(mid_node_coor, axis=0, return_index=True)
        node_axis = np.arange(len(self.mesh_undeformed) + 1, len(
            self.mesh_undeformed
        ) + len(unique_mesh) + 1
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

    def matching_mid_nodes(self, edge_match, nodes_where_longest):
        """
        This function returns the mid node coordinate at the edges where the longest edge is.
        @param: edge_match
        @param: nodes_where_longest
        @return: mid_node
        """

        match = [nodes_where_longest[index] for index in edge_match]
        mid_node = self.calculate_mid_node(match, len(match))

        return mid_node

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
    def search_matching_mid_nodes(longest_edge_nodes, nodes_where_longest):
        """
        @param: longest_edge_nodes
        @param: nodes_where_longest
        @return: edge_match
        """

        new_nodes = AMR.neighbor_intersection(longest_edge_nodes, nodes_where_longest)
        for idx, result in enumerate(new_nodes):
            if len(result) == 2:
                return new_nodes[idx][0]
            elif new_nodes[idx]:
                return new_nodes[idx][0]

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
        nodes_ele = self.nodes_array(ele)

        neighbors = []
        for i, row in enumerate(nodes_ele):
            all_neighbor = AMR.neighbor_intersection(row, self.ele_undeformed[:, 0:3])
            all_neighbor = np.asarray(all_neighbor)
            neighbor = AMR.swap_neighbor(all_neighbor)
            neighbors.append(neighbor[0])

        neighbors = np.asarray(neighbors)
        nodes = self.nodes_array(neighbors[:, 0])

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

    def blue_pattern_one_neighbor(self, ele_one_neighbor, nodes_where_longest):
        """
        Similar approach than the green pattern function. Here it is important to split the blue refinement because
        it can base on one marked neighbors or two.

        @param: not_longest_edge
        @param: neighbor
        @return:
        """

        nodes_along_neighbor, longest_edge = self.mid_node_one_neighbor(
            self.blue_marked_neighbor, nodes_where_longest, ele_one_neighbor
        )

        mid_nodes = self.stack_mid_nodes(longest_edge, nodes_along_neighbor)

        self.create_blue_pattern_one_neighbor(
            mid_nodes, ele_one_neighbor, self.blue_marked_neighbor, nodes_where_longest
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
            self, one_neighbor, ele, neighbor, nodes_where_longest
    ):
        """
        This function creates the blue pattern for elements which have one or two neighbors.

        @param one_neighbor:
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
            node_to_close_element,
            node_rotation,
        ) = AMR.keep_rotation_direction(nodes_neighbor, nodes, nodes_where_longest, ele)
        keep_node = keep_node
        node_to_close_element = node_to_close_element
        for count, row_nodes in enumerate(zip(node_rotation, one_neighbor)):
            if node_to_close_element[count] == row_nodes[0][0]:
                self.blue_ele.append(
                    np.array(
                        (
                            keep_node[count],
                            node_to_close_element[count],
                            row_nodes[1][1],
                        )
                    )
                )
            else:
                self.blue_ele.append(
                    np.array(
                        (
                            keep_node[count],
                            row_nodes[1][1],
                            node_to_close_element[count],
                        )
                    )
                )
            self.blue_ele.append(
                np.array((row_nodes[1][0], row_nodes[1][1], row_nodes[0][0]))
            )
            self.blue_ele.append(
                np.array((row_nodes[0][1], row_nodes[1][1], row_nodes[1][0]))
            )

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
        all_edges = self.get_all_edges(self.nodes_array(all_ele))
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
        self.elements_to_refine(marked_edges, all_edges, nodes_where_longest)
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
            self.for_blue_ref_one_neighbor, nodes_where_longest
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
