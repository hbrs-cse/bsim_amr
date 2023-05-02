from bcs_read import bcs_data
import numpy as np


class recursive_AMR:
    """
    This is the main class for executing the adaptive mesh refinement based on the RGB refinement strategy
    """

    def __init__(self, thickness):

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
        self.second_blue_marked_neighbor = []

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

    def get_all_edges(self, nodes_array):

        edges = [
            nodes_array[:, [0, 1]],
            nodes_array[:, [1, 2]],
            nodes_array[:, [2, 0]]
        ]
        return edges

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

    def count_occurence(self, marked_edges, all_edges, nodes_where_longest):

        marked_edges = np.vstack((marked_edges[0], marked_edges[1], marked_edges[2]))
        all_edges = np.stack((all_edges[0], all_edges[1], all_edges[2]), axis=1)

        for idx, check_edges in enumerate(all_edges):
            counter = 0
            edges = []

            for bounding_edges in check_edges:

                check = np.isin(
                        marked_edges, bounding_edges
                ).all(axis=1)
                edges.append(bounding_edges)

                if check.any():
                    counter += 1

            if counter == 3:
                self.for_red_ref.append(idx)
            elif counter == 2:
                longest_edge = nodes_where_longest[idx]
                if all(np.isin(longest_edge, edges)):
                    self.for_blue_ref_two_neighbor.append(idx)
                else:
                    self.for_red_ref.append(idx)
            elif counter == 1:
                longest_edge = nodes_where_longest[idx]
                print(longest_edge)
                if all(np.isin(longest_edge, edges)):
                    self.for_green_ref.append(idx)
                else:
                    self.for_blue_ref_one_neighbor.append(idx)

        return all_edges


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

        unique_mesh, idx = np.unique(mid_node_coor, axis=0, return_index=True)
        node_axis = np.arange(len(self.mesh_undeformed) + 1, len(
            self.mesh_undeformed) + len(unique_mesh) + 1).astype(np.int)

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
            matching_c = np.where(
                (self.bcs_mesh[:, 1:4] == coors).all(axis=1))[0]
            if matching_c or matching_c == 0:
                idx_cluster.append(matching_c)
            else:
                no_match.append(idx)
        if shape:
            if np.mod(len(idx_cluster), shape) == 0:
                mid_nodes = np.asarray([self.bcs_mesh[idx, 0].astype(
                    np.int) for idx in idx_cluster]).reshape(int(len(idx_cluster) / shape), shape)
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

        new_nodes = recursive_AMR.find_intersection(
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
        for index, row in enumerate(zip(nodes_neighbor, nodes)):
            intersection, _, indices = np.intersect1d(
                row[0], row[1], return_indices=True)
            idx1.append(np.where(row[0] == intersection[0])[0][0])
            idx2.append(np.where(row[0] == intersection[1])[0][0])

            keep_node.append(
                np.setxor1d(intersection, row[1])[0]
            )

            keep_node_index.append(
                np.where(
                    keep_node[index] == row[1])[0][0].astype(np.int)
            )

        index = np.c_[idx1, idx2]
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
            all_neighbor = np.asarray(recursive_AMR.find_intersection(
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

    def green_pattern(self, nodes_where_longest, ele, iteration):
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

        mid_node_c = self.matching_mid_nodes(edge_match, nodes_where_longest)
        mid_node, no_match = self.find_matching_mid_node(
            mid_node_c, shape=None)
        nodes = list(nodes)
        if no_match:
            if iteration == 0:
                for idx in sorted(no_match, reverse=True):
                    del self.for_green_ref[idx]
                    del self.green_marked_neighbor[idx]
                    del nodes[idx]
            else:
                for idx in sorted(no_match, reverse=True):
                    del self.for_green_ref[iteration][idx]
                    del self.green_marked_neighbor[iteration][idx]
                    del nodes[idx]

        if iteration == 0:
            nodes_neighbor = self.nodes_array(self.green_marked_neighbor)
            keep_node, index, _, _, _ = self.keep_rotation_direction(
                nodes_neighbor, nodes, nodes_where_longest, ele)
        else:
            nodes_neighbor = self.nodes_array(
                self.green_marked_neighbor[iteration])
            keep_node, index, _, _, _ = self.keep_rotation_direction(
                nodes_neighbor, nodes, nodes_where_longest, ele)

        for count, row_nodes in enumerate(zip(nodes_neighbor,
                                              np.concatenate(mid_node)
                                              )):
            self.green_ele.append(np.array(
                (row_nodes[0][index[count, 0]], row_nodes[1], keep_node[count])
            ))
            self.green_ele.append(np.array(
                (row_nodes[0][index[count, 1]], keep_node[count], row_nodes[1])
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

        try:
            match_one, no_match_one, edge_match = self.search_mid_point(
                longest_edge, nodes_one_neighbor, shape=None)
            mid_node_c = self.calculate_mid_node(
                nodes_along_neighbor, len(nodes_along_neighbor))
            match_one_nle, no_match = self.find_matching_mid_node(
                mid_node_c, shape=None)

        except ValueError:
            raise "Blue elements can not be assigned"

        try:
            one_neighbor = np.c_[match_one_nle, match_one]

        except ValueError:
            raise 'Shape mismatch in longest edge and not longest edge in the blue element cluster'

        self.create_blue_pattern_one_neighbor(
            one_neighbor,
            ele_one_neighbor,
            self.blue_marked_neighbor,
            nodes_where_longest)

    def blue_pattern_two_neighbor(
            self, longest_edge, ele_two_neighbor, iteration):
        """

        @param longest_edge:
        @param ele_two_neighbor:
        @return:
        """
        if iteration == 0:
            first_blue_neighbors = self.second_blue_marked_neighbor[:, 0]
            second_blue_neighbors = self.second_blue_marked_neighbor[:, 1]
        else:
            first_blue_neighbors = self.second_blue_marked_neighbor[iteration][:, 0]
            second_blue_neighbors = self.second_blue_marked_neighbor[iteration][:, 1]

        nodes_two_neighbor = self.nodes_array(ele_two_neighbor)
        nodes_along_second_neighbor = []
        for row_sec_blue in zip(self.ele_undeformed[first_blue_neighbors, 0:3],
                                self.ele_undeformed[second_blue_neighbors, 0:3],
                                self.ele_undeformed[ele_two_neighbor, 0:3],
                                ):
            nodes_along_second_neighbor.append(
                np.intersect1d(row_sec_blue[0], row_sec_blue[2])
            )
            nodes_along_second_neighbor.append(
                np.intersect1d(row_sec_blue[1], row_sec_blue[2])
            )
        nodes_nle = []
        for nodes in nodes_along_second_neighbor:
            result = np.where(
                (longest_edge == nodes).all(axis=1)
            )[0]
            if not result:
                nodes_nle.append(nodes)

        try:
            match_two, no_match_two, edge_match = self.search_mid_point(longest_edge,
                                                                        nodes_two_neighbor,
                                                                        shape=None)
            mid_node_c = self.calculate_mid_node(nodes_nle, len(nodes_nle))

            match_two_nle, no_match_two_nle = self.find_matching_mid_node(mid_node_c,
                                                                          shape=None)
        except ValueError:
            raise "Blue elements can not be assigned"

        try:
            two_neighbor = np.c_[match_two, match_two_nle]
        except ValueError:
            raise 'Shape mismatch in longest edge and not longest edge in the blue element cluster'

        if iteration == 0:
            self.create_blue_pattern_two_neighbor(two_neighbor,
                                                  ele_two_neighbor,
                                                  nodes_along_second_neighbor,
                                                  longest_edge)
        else:
            self.create_blue_pattern_two_neighbor(two_neighbor,
                                                  ele_two_neighbor,
                                                  nodes_along_second_neighbor,
                                                  longest_edge)

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

    def create_blue_pattern_two_neighbor(self, two_neighbor, ele, nodes_along_second_neighbor, nodes_where_longest):

        """
        This function creates the blue pattern for elements which have one or two neighbors.
        @param neighbor:
        @param two_neighbor:
        @param nodes_along_seconf_neighbor:
        @param ele:
        @return:
        """

        nodes = self.ele_undeformed[ele, 0:3]

        neighbor_one = nodes_along_second_neighbor[0:len(ele) * 2:2]
        neighbor_two = nodes_along_second_neighbor[1:len(ele) * 2:2]

        unmarked_edge = []
        keep_node = []
        keep_node_index = []
        along_longest_edge = []
        index_longest_edge_node = []
        le = [nodes_where_longest[idx] for idx in ele]
        for row in range(len(nodes)):
            node_rotation = np.setxor1d(
                neighbor_one[row], neighbor_two[row]
            ).astype(np.int)

            unmarked_edge.append(
                sorted(node_rotation, key=lambda x: nodes[row].tolist().index(x))
            )

            keep_node.append(
                np.setxor1d(
                    unmarked_edge[row], nodes[row]
                )[0]
            )

            keep_node_index.append(
                np.where(
                    keep_node[row] == nodes[row]
                )[0][0]
            )

            along_longest_edge.append(
                np.intersect1d(
                    le[row], unmarked_edge[row]
                ).astype(np.int)[0]
            )

            index_longest_edge_node.append(
                np.where(
                    along_longest_edge[row] == unmarked_edge[row]
                )[0][0]
            )
        for count, row_nodes in enumerate(two_neighbor):
            if keep_node_index[count] == 1:
                self.blue_ele.append(
                    np.array(
                        (unmarked_edge[count][0], row_nodes[index_longest_edge_node[count]], unmarked_edge[count][1])
                    )
                )
            else:
                self.blue_ele.append(
                    np.array(
                        (unmarked_edge[count][0], unmarked_edge[count][1], row_nodes[index_longest_edge_node[count]])
                    )
                )
            self.blue_ele.append(
                np.array(
                    (row_nodes[0], unmarked_edge[count][1], row_nodes[1])
                )
            )
            self.blue_ele.append(
                np.array(
                    (keep_node[count], row_nodes[0], row_nodes[1])
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


