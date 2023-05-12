from bcs_read import bcs_data
import numpy as np


class recursive_AMR:
    """
    This is the main class for executing the adaptive mesh refinement based on the RGB refinement strategy
    """

    def __init__(self, thickness):

        self.all_edges = None
        self.green_neighbor = []
        self.Bcs = bcs_data()

        self.ele_undeformed = None
        self.ele_deformed = None
        self.mesh_undeformed = None
        self.mesh_deformed = None
        self.bc = None

        self.__thickness = thickness
        self.ele_list = []
        self.marked_ele = []

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

    def thickness_diff_calc(self):
        """
        Caclulation of the thickness difference of deformed and undeformed element.
        @return: diff_calc
        """
        x, y = self.ele_deformed[:, 3], self.ele_undeformed[:, 3]
        diff_calc = []
        for ele in range(len(self.ele_undeformed)):
            diff_calc.append((1 - (y[ele] - x[ele]) / y[ele])
                             * 100)
        diff_calc = np.asarray(diff_calc)

        return diff_calc

    @property
    def set_thickness_diff(self):
        """
        Seting the thickness
        @return:
        """
        return self.__thickness

    @set_thickness_diff.setter
    def set_thickness_diff(self, min_val):
        if min_val <= 0:
            raise ValueError(
                "Input of thickness difference must be greater than 0")
        else:
            self.__thickness = min_val

    def thickness_diff(self):
        """
        Marks all elements whose thickness difference is in a sepcific range.
        @return:
        """
        thickness_diff = self.thickness_diff_calc()
        arg_list = np.where(
            (thickness_diff > self.__thickness) & (thickness_diff < 80)
        )

        ele_list = [arg_list[0].tolist()]
        for sublist in ele_list:
            for val in sublist:
                self.ele_list.append(val)
                self.marked_ele.append(val)

    def nodes_array(self, ele):
        """
        Returns a list and an array of nodes which are marked
        @return: lost of nodes, array of nodes
        """

        nodes = self.ele_undeformed[:, 0:3]
        nodes = nodes[ele].astype(np.int)
        nodes_array = np.asarray(nodes).reshape(len(ele), 3)

        return nodes_array

    @staticmethod
    def get_all_edges(nodes_array):
        """
        Create a ndarray with three edge tuple.
        @param nodes_array:
        @return: edges
        """

        edges = [
            nodes_array[:, [0, 1]],
            nodes_array[:, [1, 2]],
            nodes_array[:, [2, 0]]
        ]
        return edges

    @staticmethod
    def create_stacked_edges_array(edges):
        """
        Create a stacked array with corresponding element numbers
        @param edges:
        @return:
        """

        all_neighbor = np.stack(
            [edges[0], edges[1], edges[2]],
            axis=1
        )

        return all_neighbor

    def get_ele_length(self, marked_ele):
        """
        This function calls all function to calculate the longest edge and check where it is.
        @return:nodes_where_longest
        """
        nodes_mesh = self.neighbor_coordinates(marked_ele)
        longest_edge = recursive_AMR.ele_edge_length(nodes_mesh)
        nodes_where_longest = self.find_longest_edge(longest_edge)

        return nodes_where_longest

    def neighbor_coordinates(self, marked_ele):
        """
        Get all the coordinates of the nodes and split the array in a stack of 3 lists.
        @param all_ele:
        @return:
        """

        neighbor_nodes = self.ele_undeformed[marked_ele, 0:3].astype(
            np.int)
        for idx, add in enumerate(neighbor_nodes):
            neighbor_nodes[idx] = np.array(list(map(lambda x: x - 1, add)))
        mesh = self.mesh_undeformed[:, 0:3]
        nodes_mesh = []
        for nodes in neighbor_nodes:
            nodes_mesh.append([mesh[nodes[0], 0:3],
                               mesh[nodes[1], 0:3],
                               mesh[nodes[2], 0:3]])

        nodes_mesh = np.asarray(nodes_mesh)
        nodes_mesh = np.split(nodes_mesh, 3, axis=1)

        for flatten in range(3):
            nodes_mesh[flatten] = np.concatenate(nodes_mesh[flatten])

        return nodes_mesh

    @staticmethod
    def ele_edge_length(nodes_mesh):
        """
        Calculating the maximum side length of the elements which are next to a marked element. The euclidian norm is
        used to perform this operation. Afterwards the maximum side length with the corresponding index are calculated.
        It's possible to determine the longest side by comparing the index with the calculation of the euclidian norm.
        @param nodes_mesh:
        @return: longest_edge
        """

        euc_dist = [np.array([]), np.array([]), np.array([])]

        euc_dist[0] = np.linalg.norm((nodes_mesh[0] - nodes_mesh[1]), axis=1)
        euc_dist[1] = np.linalg.norm((nodes_mesh[0] - nodes_mesh[2]), axis=1)
        euc_dist[2] = np.linalg.norm((nodes_mesh[1] - nodes_mesh[2]), axis=1)

        euc_dist = np.c_[euc_dist[0], euc_dist[1], euc_dist[2]]
        longest_edge = [
            np.argmax(
                euc_dist[i],
                axis=0) for i in range(
                len(euc_dist))]

        return longest_edge

    def find_longest_edge(self, longest_edge):
        """
        This function checks if the longest edge is along the marked neighbors edge or not.
        @param longest_edge:
        @return: nodes_where_longest
        """

        nodes = self.ele_undeformed[:, 0:3]
        nodes_where_longest = []
        for le in zip(nodes, longest_edge):
            try:
                if le[1] == 0:
                    nodes_where_longest.append(le[0][[0, 1]])
                if le[1] == 1:
                    nodes_where_longest.append(le[0][[0, 2]])
                if le[1] == 2:
                    nodes_where_longest.append(le[0][[1, 2]])

            except BaseException:
                raise ValueError(
                    "Something went wrong while checking for the longest edge...")
        nodes_where_longest = np.asarray(nodes_where_longest)
        return nodes_where_longest

    @staticmethod
    def neighbor_intersection(row, ele_mesh):
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
        templates = [tuple(row[[0, 1]]), tuple(
            row[[0, 2]]), tuple(row[[1, 2]])]
        match_collection = []
        for idx, tuple_ele in enumerate(templates):
            match_collection.append(np.where(
                np.isin(check_nodes, tuple_ele)))
        all_neighbor = []
        for match in match_collection:
            match = match[0].tolist()
            all_neighbor.append(
                list(set([x for x in match if match.count(x) > 1])))

        return all_neighbor

    @staticmethod
    def swap_neighbor(all_neighbor):
        """
        Swap the axes, that the neighbor is in column 1 and the marked element in column 0
        @param all_neighbor:
        @return:
        """
        find_marked_ele = np.concatenate(all_neighbor).copy().tolist()
        sort = list(
            set([i for i in find_marked_ele if find_marked_ele.count(i) == 3]))
        indices = np.where(all_neighbor == sort)
        col_index = indices[1]
        for swap in range(3):
            if col_index[swap] == 1:
                all_neighbor[swap, [0, 1]] = all_neighbor[swap, [1, 0]]

        return all_neighbor

    def all_marked_elements(self):
        """
        Concatenate all marked elements.
        @return:
        """
        if self.for_blue_ref_one_neighbor and self.for_blue_ref_two_neighbor and self.for_green_ref:
            self.all_ele = self.for_blue_ref_one_neighbor + \
                           self.for_blue_ref_two_neighbor + \
                           self.for_red_ref + \
                           self.for_green_ref

            self.all_ele = np.asarray(
                self.all_ele).reshape(len(self.all_ele), 1)
        else:
            self.all_ele = self.for_red_ref
            self.all_ele = np.asarray(
                self.all_ele).reshape(len(self.all_ele), 1)

    def count_occurence(self, marked_edges, all_edges, nodes_where_longest):

        marked_edge = []

        for i in range(len(marked_edges[0])):
            marked_edge.append(marked_edges[0][i])
            marked_edge.append(marked_edges[1][i])
            marked_edge.append(marked_edges[2][i])

        marked_ele = self.marked_ele
        marked_edge = np.asarray(marked_edge)
        all_edges = self.create_stacked_edges_array(all_edges)

        hanging_edges = marked_edge
        green_longest_edge = []

        marked_ele = np.repeat(
            np.array(marked_ele).copy(),
            3,
            axis=0
        )

        ele_num = np.arange(0, len(self.ele_undeformed))
        while True:
            red = []
            blue = []
            longest_edges = []
            new_marked_ele = []
            idx_to_delete = []
            red_idx_to_delete = []
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

                elif counter == 2:
                    le = nodes_where_longest[ele_num[idx]]
                    check_for_le = np.isin(hanging_edges, le).all(axis=1)

                    if any(check_for_le):
                        if ele_num[idx] in self.for_green_ref:
                            idx_to_delete.append(ele_num[idx])
                        elif ele_num[idx] in self.for_red_ref:
                            red_idx_to_delete.append(ele_num[idx])
                            temp_blue_neighbor = []
                            self.for_blue_ref_two_neighbor.append(ele_num[idx])
                            for find_match in check:
                                true_indice = np.where(find_match)[0]

                                if len(true_indice) > 1:
                                    temp_blue_neighbor.append(
                                        marked_ele[true_indice[0]]
                                    )
                                elif true_indice or true_indice == 0:
                                    temp_blue_neighbor.append(
                                        marked_ele[true_indice[0]]
                                    )

                            self.two_blue_marked_neighbor.append(
                                temp_blue_neighbor
                            )

                        if ele_num[idx] not in self.all_ele:
                            temp_blue_neighbor = []
                            self.for_blue_ref_two_neighbor.append(ele_num[idx])
                            for find_match in check:
                                true_indice = np.where(find_match)[0]

                                if len(true_indice) > 1:
                                    temp_blue_neighbor.append(
                                        marked_ele[true_indice[0]]
                                    )
                                elif true_indice or true_indice == 0:
                                    temp_blue_neighbor.append(
                                        marked_ele[true_indice[0]]
                                    )

                            self.two_blue_marked_neighbor.append(
                                temp_blue_neighbor
                            )


                    else:
                        if ele_num[idx] in self.for_green_ref:
                            idx_to_delete.append(ele_num[idx])
                        if ele_num[idx] not in self.all_ele:
                            self.for_red_ref.append(ele_num[idx])
                            longest_edges.append(nodes_where_longest[ele_num[idx]])
                            red.append(ele_num[idx])
                            new_marked_ele.append(ele_num[idx])

                elif counter == 1:
                    le = nodes_where_longest[ele_num[idx]]
                    check_for_le = np.isin(hanging_edges, le).all(axis=1)
                    if any(check_for_le):
                        if ele_num[idx] not in self.all_ele:

                            self.for_green_ref.append(ele_num[idx])
                            green_longest_edge.append(
                                nodes_where_longest[ele_num[idx]]
                            )

                            for find_match in check:
                                true_indice = np.where(find_match)[0]
                                if true_indice or true_indice == 0:
                                    self.green_marked_neighbor.append(
                                        marked_ele[true_indice[0]]
                                    )
                    else:
                        if ele_num[idx] in self.for_red_ref:
                            red_idx_to_delete.append(ele_num[idx])
                        if ele_num[idx] not in self.all_ele:
                            self.for_blue_ref_one_neighbor.append(ele_num[idx])
                            longest_edges.append(nodes_where_longest[ele_num[idx]])
                            new_marked_ele.append(ele_num[idx])

                            for find_match in check:
                                true_indice = np.where(find_match)[0]
                                if true_indice or true_indice == 0:
                                    self.blue_marked_neighbor.append(
                                        marked_ele[true_indice[0]]
                                    )
                            blue.append(ele_num[idx])

            print(len(new_marked_ele))
            print('blue one', len(self.for_blue_ref_one_neighbor))
            print('green', len(self.for_green_ref))
            print('blue two', len(self.for_blue_ref_two_neighbor))
            print('red', len(self.for_red_ref))
            print('new red', len(red))
            print('new blue', len(blue))

            if len(new_marked_ele) > 0:
                self.all_marked_elements()
                # all_edges, ele_num = self.neighbors_along_longest_edge(all_edges, longest_edges)
                hanging_edges = np.append(hanging_edges,
                                          longest_edges,
                                          axis=0)
                marked_ele = np.append(marked_ele, new_marked_ele)
            else:
                break

        inter_blue_one_neighbor_red = np.intersect1d(self.for_blue_ref_one_neighbor, self.for_red_ref).tolist()
        inter_blue_two_neighbor_red = np.intersect1d(self.for_blue_ref_two_neighbor, self.for_red_ref).tolist()
        inter_green_red = np.intersect1d(self.for_green_ref, self.for_red_ref).tolist()
        inter_green_blue_two_neighbor = np.intersect1d(self.for_green_ref, self.for_blue_ref_two_neighbor).tolist()
        inter_green_blue_one_neighbor = np.intersect1d(self.for_green_ref, self.for_blue_ref_one_neighbor).tolist()
        inter_blue = np.intersect1d(self.for_blue_ref_two_neighbor, self.for_blue_ref_one_neighbor).tolist()

        if inter_green_blue_two_neighbor:
            for val in sorted(inter_green_blue_two_neighbor + inter_green_red, reverse=True):
                indice = np.where(
                    np.array(self.for_green_ref) == val
                )[0]

                del self.for_green_ref[indice[0]]
                del self.green_marked_neighbor[indice[0]]

        if inter_blue_one_neighbor_red:
            for val in sorted(inter_blue_one_neighbor_red + inter_blue_two_neighbor_red, reverse=True):
                indice = np.where(
                    np.array(self.for_red_ref) == val
                )[0]

                del self.for_red_ref[indice[0]]

        inter_blue_one_neighbor_red = np.intersect1d(self.for_blue_ref_one_neighbor, self.for_red_ref).tolist()
        inter_blue_two_neighbor_red = np.intersect1d(self.for_blue_ref_two_neighbor, self.for_red_ref).tolist()
        inter_green_red = np.intersect1d(self.for_green_ref, self.for_red_ref).tolist()
        inter_green_blue_two_neighbor = np.intersect1d(self.for_green_ref, self.for_blue_ref_two_neighbor).tolist()
        inter_green_blue_one_neighbor = np.intersect1d(self.for_green_ref, self.for_blue_ref_one_neighbor).tolist()
        inter_blue = np.intersect1d(self.for_blue_ref_two_neighbor, self.for_blue_ref_one_neighbor).tolist()
        return all_edges

    def neighbors_along_longest_edge(self, all_edges, longest_edges):
        """
        Returns the element along the longest edge of the marked element.
        @param all_edges:
        @param longest_edges:
        @return:
        """
        ele_num = []
        for edges in longest_edges:
            for column, col_index in enumerate(range(3)):
                edge_search = np.isin(
                    all_edges[:, col_index], edges
                ).all(axis=1)

                if edge_search.any():
                    ele_index = np.where(edge_search)[0]
                    for ele in ele_index:
                        if not np.array_equal(all_edges[ele, column], edges):
                            ele_num.append(ele_index[0])

        longest_edge_neighbor_edge = self.nodes_array(ele_num)
        get_edges = recursive_AMR.get_all_edges(longest_edge_neighbor_edge)
        stacked_neighbor = recursive_AMR.create_stacked_edges_array(get_edges)

        return stacked_neighbor, ele_num

    def get_edges_along_blue_elements(self):
        """
        Get all edges along the marked already marked elements of a blue element with two neighbor
        """

        self.two_blue_marked_neighbor = np.asarray(self.two_blue_marked_neighbor)
        first_neighbor = self.nodes_array(
            self.two_blue_marked_neighbor[:, 0]
        )
        second_neighbor = self.nodes_array(
            self.two_blue_marked_neighbor[:, 1]
        )
        marked_ele = self.nodes_array(
            self.for_blue_ref_two_neighbor
        )

        for first_neighbor, second_neighbor, marked_ele, in zip(first_neighbor,
                                                                second_neighbor,
                                                                marked_ele,
                                                                ):
            self.nodes_along_second_neighbor.append(
                np.intersect1d(first_neighbor, marked_ele)
            )
            self.nodes_along_second_neighbor.append(
                np.intersect1d(second_neighbor, marked_ele)
            )
        self.two_blue_marked_neighbor = self.two_blue_marked_neighbor.tolist()

    def mid_nodes(self, ele):
        """
        Calculation of all mid nodes. Afterwards a template will be created with new mid node numbers and their
        corresponding coordinates.

        @return:
        """
        red_coors = self.neighbor_coordinates(ele)
        mid_node_1 = np.divide(
            (np.add(red_coors[0], red_coors[1])), 2).round(decimals=6)
        mid_node_2 = np.divide(
            (np.add(red_coors[1], red_coors[2])), 2).round(decimals=6)
        mid_node_3 = np.divide(
            (np.add(red_coors[2], red_coors[0])), 2).round(decimals=6)

        split_size = len(mid_node_1)
        mid_node_1 = np.split(np.concatenate(mid_node_1), split_size)
        mid_node_2 = np.split(np.concatenate(mid_node_2), split_size)
        mid_node_3 = np.split(np.concatenate(mid_node_3), split_size)

        mid_node_coor = np.hstack(
            (mid_node_1, mid_node_2, mid_node_3)).reshape(
            len(mid_node_1) * 3, 3)

        # unique_mesh, idx = np.unique(mid_node_coor, axis=0, return_index=True)
        node_axis = np.arange(len(self.mesh_undeformed) + 1, len(
            self.mesh_undeformed) + len(mid_node_coor) + 1).astype(np.int)

        # Keeping the ordner because np.unique sorts by size (Not necessary but
        # keeps the chronologically order)
        # bcs_mesh = unique_mesh[np.argsort(idx)]
        self.bcs_mesh = np.hstack((node_axis[:, np.newaxis], mid_node_coor))
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
            matching_c = np.where(
                (self.bcs_mesh[:, 1:4] == coors).all(axis=1))[0]
            if len(matching_c) > 0:

                idx_cluster.append(matching_c[0])
            elif matching_c or matching_c == 0:
                idx_cluster.append(matching_c)
            else:
                no_match.append(idx)
        if shape:
            if np.mod(len(idx_cluster), shape) == 0:
                mid_nodes = np.asarray([self.bcs_mesh[idx, 0].astype(
                    np.int) for idx in idx_cluster]).reshape(-1, shape)
                # int(len(idx_cluster) / shape)
            else:
                raise ValueError(
                    "Somenthing went wrong while trying to find the mid nodes for the refinement")
        else:
            mid_nodes = np.asarray([self.bcs_mesh[idx, 0].astype(np.int)
                                    for idx in idx_cluster if idx or idx == 0])
        return mid_nodes, no_match

    def matching_mid_nodes(self, edge_match, nodes_where_longest):
        """
        This function calculates the mid node coordinate at the edges where the longest edge is.
        @param: edge_match
        @param: nodes_where_longest
        @return: mid_node
        """

        match = [nodes_where_longest[index] for index in edge_match]
        mid_node = self.calculate_mid_node(match, len(match))

        return mid_node

    def calculate_mid_node(self, match, container_len):

        c_container = [np.array(np.empty((container_len, 3))),
                       np.array(np.empty((container_len, 3)))]

        match = list(map(lambda x: x - 1, match))

        for i in range(2):
            for idx, nodes in enumerate(match):
                c_container[i][idx] = self.mesh_undeformed[nodes[i].astype(
                    np.int), 0:3]

        mid_node = np.divide(
            (np.add(c_container[0], c_container[1])), 2).round(decimals=6)
        return mid_node

    def search_matching_mid_nodes(
            self, longest_edge_nodes, nodes_where_longest):
        """
        @param: longest_edge_nodes
        @param: nodes_where_longest
        @return: edge_match
        """

        new_nodes = recursive_AMR.neighbor_intersection(
            longest_edge_nodes, nodes_where_longest)
        for idx, result in enumerate(new_nodes):
            if len(result) == 2:
                return new_nodes[idx][0]
            elif new_nodes[idx]:
                return new_nodes[idx][0]

    @staticmethod
    def keep_rotation_direction(nodes_neighbor, nodes, nodes_where_longest, ele):
        """
        Neighboring nodes change their order to keep the rotation direction. Therefore it's very important
        to place the nodes at the right position, because they differ depending on the neighbor node position
        @param nodes_neighbor:
        @param nodes:
        @return:
        """

        idx1 = []
        idx2 = []
        keep_node = []
        keep_node_index = []
        nodes = np.asarray(nodes)
        for index, row in enumerate(zip(nodes_neighbor, nodes)):
            intersection, _, indices = np.intersect1d(
                row[0], row[1], return_indices=True)
            idx1.append(np.where(row[0] == intersection[0]))
            idx2.append(np.where(row[0] == intersection[1]))

            keep_node.append(
                np.setxor1d(intersection, row[1])[0]
            )

            keep_node_index.append(
                np.where(
                    keep_node[index] == row[1])[0][0].astype(np.int)
            )

        index = np.concatenate(
            np.c_[idx1, idx2]
        )
        nodes = np.array(nodes)

        node_rotation = recursive_AMR.nodes_rotation(keep_node_index, nodes)

        node_to_close_element = []
        le = [nodes_where_longest[idx] for idx in ele]
        for idx, elements in enumerate(le):
            node_to_close_element.append(
                int(np.setxor1d(elements, nodes[idx])[0])
            )

        return keep_node, index, keep_node_index, node_to_close_element, node_rotation

    @staticmethod
    def nodes_rotation(keep_node_index, nodes):
        """
        Check the nodes rotation
        @param keep_node_index:
        @param nodes:
        return: node_rotation
        """

        node_rotation = []
        for idx, row in enumerate(keep_node_index):
            if row == 1:
                node_rotation.append(
                    np.array((nodes[idx, 2], nodes[idx, 0]))
                )
            elif row == 2:
                node_rotation.append(
                    np.array((nodes[idx, 0], nodes[idx, 1]))
                )
            else:
                node_rotation.append(
                    np.array((nodes[idx, 1], nodes[idx, 2]))
                )
        return node_rotation

    def search_mid_point(self, longest_edge, nodes, shape):
        """

        @param longest_edge:
        @param nodes:
        @param shape:
        @return:
        """

        edge_match = []
        for gp in nodes:
            edge_match.append(
                self.search_matching_mid_nodes(gp, longest_edge)
            )

        mid_node_c = self.matching_mid_nodes(edge_match, longest_edge)
        mid_node, no_match = self.find_matching_mid_node(mid_node_c, shape)

        return mid_node, no_match, edge_match

    def find_vertex_and_mid_node(self, nodes,
                                 neighbor_one,
                                 neighbor_two,
                                 two_neighbor,
                                 mid_node_with_le):
        """
        Find the node that closes the element and the corresponding vertex node. This is important because
        otherwise it's unclear which mid node should be used to create the blue element with two neighbots.
        @param nodes:
        @param neighbor_one:
        @param neighbor_two:
        @param two_neighbor:
        @param mid_node_with_le:
        @return: unmarked_edge, node_to_close, vertex_node
        """

        node_rotation = []
        keep_node = []
        keep_node_index = []
        node_to_close = []
        vertex_node = []

        for row in range(len(nodes)):
            node_rotation.append(np.setxor1d(
                neighbor_one[row], neighbor_two[row]
            ).astype(np.int)
                                 )

            keep_node.append(
                np.setxor1d(
                    node_rotation[row], nodes[row]
                )[0]
            )

            keep_node_index.append(
                np.where(
                    keep_node[row] == nodes[row]
                )[0][0]

            )

        unmarked_edge = self.nodes_rotation(keep_node_index, nodes)

        for edge in range(len(keep_node_index)):
            longest_edge_to_node = np.intersect1d(
                unmarked_edge[edge][0], mid_node_with_le[1][edge]
            )

            if longest_edge_to_node:
                node_to_close.append(
                    two_neighbor[edge][0]
                )
                vertex_node.append(
                    two_neighbor[edge][1]
                )
            else:
                node_to_close.append(
                    two_neighbor[edge][1]
                )
                vertex_node.append(
                    two_neighbor[edge][0]
                )

        return unmarked_edge, node_to_close, vertex_node, keep_node

    def red_pattern(self, mid_nodes_coor, ele):
        """
        Creates a pattern for the red refined elements. First of all we use the list of unique elements (bcs_mesh)
        as a reference, because it includes a fresh generated axis of element numbers (self.new_nodes) and the
        corresponding coordinates.
        It is also necessary to implement the new pattern of the element and even more important to keep the correct
        rotating direction.
        The pattern can be used for the blue and green refinement. It is not necessary to define new node numbers
        because the middle nodes of green and blue elements are connecting middle nodes of other elements


        @param new_nodes:
        @param mid_nodes_coor:
        @return:
        """

        mid_nodes, no_match = self.find_matching_mid_node(
            mid_nodes_coor, shape=3)

        neighbors = []
        for row in self.nodes_array(ele):
            all_neighbor = np.asarray(recursive_AMR.neighbor_intersection(
                row, self.ele_undeformed[:, 0:3]))
            neighbor = recursive_AMR.swap_neighbor(all_neighbor)
            neighbors.append(neighbor[0])

        neighbors = np.asarray(neighbors)
        nodes = self.nodes_array(neighbors[:, 0])
        nodes_neighbor = self.nodes_array(np.asarray(neighbors)[:, 1])
        # neighbor_nodes_rotation, rotation_index, keep_nodes = self.rotation_direction(nodes_neighbor, nodes)

        for count, row_nodes in enumerate(zip(nodes_neighbor,
                                              mid_nodes,
                                              nodes)):
            self.red_ele.append(np.array(
                (row_nodes[1][0], row_nodes[1][1], row_nodes[1][2])
            ))
            self.red_ele.append(np.array(
                (row_nodes[2][2], row_nodes[1][2], row_nodes[1][1])
            ))
            self.red_ele.append(np.array(
                (row_nodes[1][2], row_nodes[2][0], row_nodes[1][0])
            ))
            self.red_ele.append(np.array(
                (row_nodes[1][0], row_nodes[2][1], row_nodes[1][1])
            ))

        #              x3
        #            /  |  \
        #           /   |   \
        #          /    |    \
        #         /     |     \
        #        /      |      \
        #       x3*------------ x2*
        #      / \     |      / \
        #     /   \    |     /   \
        #    /     \   |    /     \
        #   /       \  |   /       \
        #  /         \ |  /         \
        # x1----------x1*------------x2
        #          longeset edge

    def green_pattern(self, nodes_where_longest, ele):
        """
        There are two main operations in this function. The first loop searches the two connected nodes with the longest
        edge in the element. The function call self.find_matching_mid_nodes checks whether the mid node of the longest
        edge is present in the bcs_mesh template. If so, the green element is a neighbor of a red element. If not, it
        is the neighbor of a blue element.
        @return:green_ele
        """
        nodes = self.nodes_array(ele)

        edge_match = []
        for gp in nodes:
            edge_match.append(
                self.search_matching_mid_nodes(gp, nodes_where_longest)
            )
        res = [nodes_where_longest[index] for index in self.for_green_ref]
        mid_node_le = self.calculate_mid_node(res, len(res))
        mid_node, no_match = self.find_matching_mid_node(mid_node_le,
                                                         shape=None)

        nodes = list(nodes)
        nodes_neighbor = self.nodes_array(self.green_marked_neighbor)
        keep_node, index, _, _, nodes_longest_edge = self.keep_rotation_direction(
            nodes_neighbor, nodes, nodes_where_longest, ele)

        for count, row_nodes in enumerate(zip(nodes_longest_edge,
                                              mid_node
                                              )
                                          ):
            self.green_ele.append(np.array(
                (row_nodes[1], keep_node[count], row_nodes[0][0])
            ))
            self.green_ele.append(np.array(
                (row_nodes[1], row_nodes[0][1], keep_node[count])
            ))

    #              x3
    #            /  |  \
    #           /   |   \
    #          /    |    \
    #         /     |     \
    #        /      |      \
    #       /       |       \
    #      /        |        \
    #     /         |         \
    #    /          |          \
    #   /           |           \
    #  /            |            \
    # x1----------x1*------------x2
    #          longeset edge

    def blue_pattern_one_neighbor(self, longest_edge, ele_one_neighbor, nodes_where_longest):
        """
        Similar approach than the green pattern function. Here it is important to split the blue refinement because
        it can base on one marked neighbors or two.
        @param: longest_edge
        @param: not_longest_edge
        @param: neighbor
        @return:
        """

        nodes_one_neighbor = self.nodes_array(ele_one_neighbor)

        nodes_along_neighbor = []
        for row_blue in zip(self.ele_undeformed[self.blue_marked_neighbor, 0:3],
                            self.ele_undeformed[ele_one_neighbor, 0:3]):
            nodes_along_neighbor.append(
                np.intersect1d(row_blue[0], row_blue[1])
            )

        le = [longest_edge[index] for index in self.for_blue_ref_one_neighbor]
        try:
            mid_node_le = self.calculate_mid_node(le, len(le))
            match_one_le, no_match_le = self.find_matching_mid_node(
                mid_node_le, shape=None
            )

            mid_node_c = self.calculate_mid_node(
                nodes_along_neighbor, len(nodes_along_neighbor))
            match_one_nle, no_match = self.find_matching_mid_node(
                mid_node_c, shape=None)

        except ValueError:
            raise "Blue elements can not be assigned"

        try:
            one_neighbor = np.c_[match_one_nle, match_one_le]

        except ValueError:
            raise 'Shape mismatch in longest edge and not longest edge in the blue element cluster'

        self.create_blue_pattern_one_neighbor(
            one_neighbor,
            ele_one_neighbor,
            self.blue_marked_neighbor,
            nodes_where_longest)

    def blue_pattern_two_neighbor(
            self, longest_edge, ele_two_neighbor):
        """

        @param longest_edge:
        @param ele_two_neighbor:
        @return:
        """

        nodes_two_neighbor = self.nodes_array(ele_two_neighbor)
        self.get_edges_along_blue_elements()

        nodes_nle = []
        res = [longest_edge[index] for index in self.for_blue_ref_two_neighbor]
        other = []
        for idx, nodes in enumerate(self.nodes_along_second_neighbor):
            result = np.isin(
                res, nodes
            ).all(axis=1)
            if not any(result):
                nodes_nle.append(nodes)
            else:
                other.append(nodes)

        try:
            # match_two, no_match_two, edge_match = self.search_mid_point(longest_edge,
            #                                                            nodes_two_neighbor,
            #                                                            shape=None)
            # le_to_mid_node = [longest_edge[edge] for edge in edge_match]

            mid_node_le = self.calculate_mid_node(res, len(res))
            match_two, no_match_two_le = self.find_matching_mid_node(mid_node_le,
                                                                     shape=None)
            mid_node_c = self.calculate_mid_node(nodes_nle, len(nodes_nle))

            match_two_nle, no_match_two_nle = self.find_matching_mid_node(mid_node_c,
                                                                          shape=None)
        except ValueError:
            raise "Blue elements can not be assigned"

        try:
            two_neighbor = np.c_[match_two, match_two_nle]
            mid_node_with_le = np.array((match_two, res))
        except ValueError:
            raise 'Shape mismatch in longest edge and not longest edge in the blue element cluster'

        self.create_blue_pattern_two_neighbor(two_neighbor,
                                              ele_two_neighbor,
                                              self.nodes_along_second_neighbor,
                                              mid_node_with_le)

    def create_blue_pattern_one_neighbor(self, one_neighbor, ele, neighbor, nodes_where_longest):
        """
        This function creates the blue pattern for elements which have one or two neighbors.
        @param neighbor_stack:
        @param ele:
        @return:
        """
        nodes = self.nodes_array(ele)
        nodes_neighbor = self.nodes_array(neighbor)
        keep_node, index, keep_node_index, node_to_close_element, node_rotation = self.keep_rotation_direction(
            nodes_neighbor, nodes, nodes_where_longest, ele)
        keep_node = keep_node
        node_to_close_element = node_to_close_element
        for count, row_nodes in enumerate(zip(node_rotation,
                                              one_neighbor)):
            if node_to_close_element[count] == row_nodes[0][0]:
                self.blue_ele.append(
                    np.array(
                        (keep_node[count], node_to_close_element[count], row_nodes[1][1])
                    )
                )
            else:
                self.blue_ele.append(
                    np.array(
                        (keep_node[count], row_nodes[1][1], node_to_close_element[count])
                    )
                )
            self.blue_ele.append(
                np.array(
                    (row_nodes[1][0], row_nodes[1][1], row_nodes[0][0])
                )
            )
            self.blue_ele.append(
                np.array(
                    (row_nodes[0][1], row_nodes[1][1], row_nodes[1][0])
                )
            )

    def create_blue_pattern_two_neighbor(self,
                                         two_neighbor,
                                         ele,
                                         nodes_along_second_neighbor,
                                         mid_node_with_le):

        """
        This function creates the blue pattern for elements which have one or two neighbors.
        @param two_neighbor:
        @param nodes_along_second_neighbor:
        @param ele:
        @param mid_node_with_le:
        @return:
        """

        nodes = self.ele_undeformed[ele, 0:3]
        neighbor_one = nodes_along_second_neighbor[0:len(ele) * 2:2]
        neighbor_two = nodes_along_second_neighbor[1:len(ele) * 2:2]

        unmarked_edge, node_to_close, vertex_node, keep_node = self.find_vertex_and_mid_node(nodes,
                                                                                             neighbor_one,
                                                                                             neighbor_two,
                                                                                             two_neighbor,
                                                                                             mid_node_with_le)

        for count, (node_to_close, vertex_node) in enumerate(zip(node_to_close, vertex_node)):
            self.blue_ele.append(
                np.array(
                    (unmarked_edge[count][0], unmarked_edge[count][1], node_to_close)
                )
            )
            self.blue_ele.append(
                np.array(
                    (node_to_close, unmarked_edge[count][1], vertex_node)
                )
            )
            self.blue_ele.append(
                np.array(
                    (keep_node[count], node_to_close, vertex_node)
                )
            )

    def main_recursive_amr(self):
        self.thickness_diff()
        all_ele = np.arange(0, len(self.ele_undeformed))
        nodes_array = self.nodes_array(self.marked_ele)
        marked_edges = self.get_all_edges(nodes_array)
        all_edges = self.get_all_edges(self.nodes_array(all_ele))
        nodes_where_longest = self.get_ele_length(all_ele)
        self.count_occurence(marked_edges, all_edges, nodes_where_longest)
        mid_node_coors = self.mid_nodes(self.for_red_ref + self.for_blue_ref_one_neighbor)
        self.red_pattern(mid_node_coors, self.for_red_ref)
        self.green_pattern(nodes_where_longest, self.for_green_ref)
        self.blue_pattern_one_neighbor(
            nodes_where_longest, self.for_blue_ref_one_neighbor, nodes_where_longest)

        self.blue_pattern_two_neighbor(
            nodes_where_longest, self.for_blue_ref_two_neighbor)
