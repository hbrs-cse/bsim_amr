"""
Fabian Kind
Hochschule Bonn-Rhein-Sieg
Institut für Elektrotechnik, Maschinenbau und Technikjournalismus
Masterprojekt 1
Adaptive Mesh Refinement
"""

from bcs_read import bcs_data
import os
import numpy as np


class AMR:
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

    def append_nodes(self, ele):
        if self.marked_ele:
            for check in ele:
                if check not in self.marked_ele:
                    self.marked_ele.append(ele)
                    if isinstance(self.marked_ele[-1], list):
                        self.marked_ele = self.marked_ele[0:- \
                            2] + self.marked_ele[-1]

    @staticmethod
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

            all_neighbor = np.asarray(AMR.find_intersection(row, check_nodes))

            all_neighbor = AMR.swap_neighbor(all_neighbor)

            try:
                three_neighbor = 0
                for idx, list_counter in enumerate(all_neighbor):
                    if list_counter[1] not in ele:
                        if list_counter[1] not in self.all_ele:
                            neighbor_collection.append(list_counter)
                            three_neighbor += 1
                            if three_neighbor == 3:
                                three_unmarked.append(i)
                        else:
                            marked_neighbor.append(list_counter)
            except ValueError:
                raise "Something went wrong while determining the direct neighbor..."

        direct_neighbor = np.asarray(neighbor_collection)[:, 1] if neighbor_collection else None
        marked_ele = np.asarray(neighbor_collection)[:, 0] if neighbor_collection else None
        marked_neighbor = np.asarray(marked_neighbor) if marked_neighbor else None
        three_unmarked = np.unique(three_unmarked)
        return direct_neighbor, marked_ele, nodes, marked_neighbor, three_unmarked

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
        longest_edge = [np.argmax(euc_dist[i], axis=0) for i in range(len(euc_dist))]

        return longest_edge

    def neighbor_coordinates(self, all_ele):
        """
        Get all the coordinates of the nodes and split the array in a stack of 3 lists.
        @param all_ele:
        @return:
        """

        neighbor_nodes = self.ele_undeformed[all_ele, 0:3].astype(
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

    def find_longest_edge(self, longest_edge):
        """
        This function checks if the longest edge is along the marked neighbors edge or not.
        @param longest_edge:
        @param direct_neighbor:
        @return: nodes_where_longest
        """

        nodes = self.ele_undeformed[:, 0:3]
        nodes_where_longest = []
        not_longest = []
        for le in zip(nodes, longest_edge):
            try:
                if le[1] == 0:
                    nodes_where_longest.append(le[0][[0, 1]])
                else:
                    not_longest.append(le[0][[0, 1]])
                if le[1] == 1:
                    nodes_where_longest.append(le[0][[0, 2]])
                else:
                    not_longest.append(le[0][[0, 2]])
                if le[1] == 2:
                    nodes_where_longest.append(le[0][[1, 2]])
                else:
                    not_longest.append(le[0][[1, 2]])

            except BaseException:
                raise ValueError(
                    "Something went wrong while checking for the longest edge...")

        return nodes_where_longest, not_longest

    def check_for_green_blue(self, marked_ele, nodes_where_longest, direct_neighbor):
        """
        This is the loop for determining if a neighbor of the marked element is along the longest edge or not.
        This is necessary for the decision for a blue or green refinement.
        @param direct_neighbor:
        @param marked_ele:
        @param nodes_where_longest:
        @return:green_check, blue_check
        """
        check_nodes = self.ele_undeformed[marked_ele, 0:3]
        nodes_where_longest = [nodes_where_longest[marked] for marked in direct_neighbor]
        green_check = []
        for ge in check_nodes:
            intersection = self.find_intersection(ge, nodes_where_longest)
            for result in intersection:
                if result:
                    for result_iter in result:
                        if result_iter not in marked_ele:
                            green_check.append(result_iter)

        return green_check

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

            self.all_ele = np.asarray(self.all_ele).reshape(len(self.all_ele), 1)
        else:
            self.all_ele = self.for_red_ref
            self.all_ele = np.asarray(self.all_ele).reshape(len(self.all_ele), 1)

    def get_ele_length(self, all_ele):
        """
        This function calls all function to calculate the longest edge and check where it is.
        @return:nodes_where_longest
        """
        nodes_mesh = self.neighbor_coordinates(all_ele)
        longest_edge = self.ele_edge_length(nodes_mesh)
        nodes_where_longest, not_longest = self.find_longest_edge(
            longest_edge)

        return nodes_where_longest, not_longest

    def second_neighbor(self, marked_ele, direct_neighbor):
        """

        @param marked_ele:
        @param direct_neighbor:
        @return:
        """

        _, inverse, count = np.unique(direct_neighbor,
                                      return_inverse=True,
                                      return_counts=True)
        duplicates = np.where(count[inverse] == 2)[0].tolist()
        surrounded_ele = np.where(count[inverse] == 3)[0].tolist()
        second_neighbor = np.c_[marked_ele[duplicates], direct_neighbor[duplicates]]
        second_neighbor = second_neighbor[second_neighbor[:, 1].argsort()]
        return second_neighbor, surrounded_ele

    def get_green_blue_elements(
            self,
            direct_neighbor,
            marked_ele,
            second_neighbor,
            surrounded_ele,
            nodes_where_longest):
        """
        @param marked_ele
        @param nodes_where_longest:
        @param direct_neighbor:
        @param second_neighbor:
        @param surrounded_ele:
        """

        hanging_nodes_red = []
        hanging_nodes_blue = []
        # If an element is surrounded by three marked elements it's a red element
        for sur in surrounded_ele:
            self.for_red_ref.append(direct_neighbor[sur])

        direct_neighbor = np.delete(direct_neighbor, surrounded_ele)
        marked_ele = np.delete(marked_ele, surrounded_ele)

        # If an element is in the marked set of elements and is a marked neighbor of a unmarked element which has
        # two neighbors and if it's not present in the green refinement than both marked elements are not along
        # the longest side and the unmarked element must be red refined.

        sec_neighbor_idx = []
        for idx, sec_blue in enumerate(direct_neighbor):
            neighbor = AMR.find_intersection(self.ele_undeformed[sec_blue, 0:3], nodes_where_longest)
            for not_empty in neighbor:
                if not_empty:
                    for ele in not_empty:
                        if ele != sec_blue:
                            if ele in self.for_red_ref:
                                if sec_blue in second_neighbor[:, 1]:
                                    if sec_blue not in self.for_blue_ref_two_neighbor:
                                        self.for_blue_ref_two_neighbor.append(sec_blue)
                                        sec_neighbor_idx.append(np.where(ele == second_neighbor[:, 0])[0])
                                else:
                                    self.for_green_ref.append(sec_blue)
                                    self.green_marked_neighbor.append(marked_ele[idx])
                            else:
                                if sec_blue in second_neighbor[:, 1]:
                                    self.for_red_ref.append(sec_blue)
                                    hanging_nodes_red.append(sec_blue)
                                else:
                                    self.for_blue_ref_one_neighbor.append(sec_blue)
                                    self.blue_marked_neighbor.append(marked_ele[idx])
                                    hanging_nodes_blue.append(sec_blue)

        # Sometimes a marked element is the second neighbor of 2 neighboring elements which are blue refined.
        # It's necessary to determine where this marked element belongs to, because otherwise it get's assigned wrong
        # Caution! Assigning idx[1] could be wrong because it's possible that a marked element has 3 neighbors
        # which are blue refined!

        for ele, idx in enumerate(sec_neighbor_idx):
            if len(idx) > 1:
                if second_neighbor[idx[0], 1] == self.for_blue_ref_two_neighbor[ele]:
                    sec_neighbor_idx[ele] = np.asarray([idx[0]])
                else:
                    sec_neighbor_idx[ele] = np.asarray([idx[1]])

        for get_neighbor in sec_neighbor_idx:
            idx_neighbor = np.where(second_neighbor[get_neighbor, 1] == second_neighbor[:, 1])[0]
            neighbor_of_blue_marked = []
            for new_ele in idx_neighbor:
                neighbor_of_blue_marked.append(second_neighbor[new_ele, 0])
            self.second_blue_marked_neighbor.append(neighbor_of_blue_marked)

        self.all_marked_elements()

        hanging_nodes_red = np.unique(hanging_nodes_red).tolist()
        hanging_nodes_blue = np.unique(hanging_nodes_blue).tolist()

        _, _, _, marked_neighbor, _ = self.direct_neighbor(hanging_nodes_red + hanging_nodes_blue)
        direct_neighbor_red, marked_ele_red, _, marked_neighbor_red, _ = self.direct_neighbor(hanging_nodes_red)
        green_to_delete = []
        idx = []
        for index, checker in enumerate(marked_neighbor_red[:, 1]):
            if checker in self.for_green_ref:
                green_to_delete.append(checker)
                idx.append(index)

        green_idx = [np.where(ele == self.for_green_ref)[0][0] for ele in green_to_delete]

        for assign_blue in green_to_delete:
            self.for_blue_ref_two_neighbor.append(assign_blue)
        self.second_blue_marked_neighbor = self.second_blue_marked_neighbor

        for assign_blue_neighbor in zip(idx, green_idx):
            add_blue_neighbor = np.array((marked_neighbor_red[assign_blue_neighbor[0], 1],
                                          self.green_marked_neighbor[assign_blue_neighbor[1]]))
            self.second_blue_marked_neighbor.append(add_blue_neighbor)

        self.second_blue_marked_neighbor = np.asarray(self.second_blue_marked_neighbor)
        for delete_ele in sorted(green_idx, reverse=True):
            del self.for_green_ref[delete_ele]
            del self.green_marked_neighbor[delete_ele]

        for to_refine in hanging_nodes_red:
            self.for_red_ref.append(to_refine)

        return hanging_nodes_red, hanging_nodes_blue


    def second_iter_ele(self, second_neighbor, direct_neighbor, marked_ele, along_edge):
        """
        Similiar algorithm to function 'get_green_blue_elements' but this algorithm finds only
        blue elements with two neighbors and green elements because thats the only possible way
        for closing the iteration loop.

        @param second_neighbor:
        @param direct_neighbor:
        @param marked_ele:
        @param along_edge:
        @return:
        """

        for_green_ref = []
        for_blue_ref = []
        sec_neighbor_idx = []
        idx_to_delete = []
        green_marked_neighbor = []

        second_blue_marked_neighbor = []
        # Delete second neighbor if it's a green element or a blue element with two neighbors
        for idx, check_ele in enumerate(second_neighbor[:, 0]):
            if check_ele in np.concatenate(self.for_green_ref):
                if check_ele in np.concatenate(self.for_blue_ref_two_neighbor):
                    idx_to_delete.append(idx)

        second_neighbor = np.delete(second_neighbor, idx_to_delete, axis=0)

        # Find Intersection between the corresponding edge (for blue elements with one neoghbor the longest
        # edge and for red elements not the longest edge

        direct_neighbor_to_delete = []
        for idx, edge in enumerate(direct_neighbor):
            neighbor = AMR.find_intersection(self.ele_undeformed[edge, 0:3], along_edge)
            if not any(neighbor):
                direct_neighbor_to_delete.append(idx)
        direct_neighbor = np.delete(direct_neighbor, direct_neighbor_to_delete)
        marked_ele = np.delete(marked_ele, direct_neighbor_to_delete)

        # Check if the neighbor of the hanging nodes element has also a marked neighbr. If so, the
        # hanging marked neighbor is a blue element and otherwise a green element

        for idx, neighbor_ele in enumerate(direct_neighbor):
            neighbor = AMR.find_intersection(self.ele_undeformed[neighbor_ele, 0:3], self.ele_undeformed[:, 0:3])
            if neighbor_ele in second_neighbor[:, 1]:
                for not_empty in neighbor:
                    if not_empty:
                        for ele in not_empty:
                            if ele != neighbor_ele:
                                if neighbor_ele not in for_blue_ref:
                                    for_blue_ref.append(neighbor_ele)
                                    sec_neighbor_idx.append(np.where(neighbor_ele == second_neighbor[:, 1]))
            else:
                for_green_ref.append(neighbor_ele)
                green_marked_neighbor.append(marked_ele[idx])

        self.green_marked_neighbor.append(green_marked_neighbor)

        # Assign the new blue marked elements neighbor

        for marked_neighbor in sec_neighbor_idx:
            for marked_ele in marked_neighbor:
                second_blue_marked_neighbor.append(second_neighbor[marked_ele, 0])

        self.second_blue_marked_neighbor.append(np.asarray(second_blue_marked_neighbor))

        # Create a second list

        if for_blue_ref:
            self.for_blue_ref_two_neighbor.append(for_blue_ref)
        if for_green_ref:
            self.for_green_ref.append(for_green_ref)

    def remove_hanging_nodes(self, hanging_nodes_red, hanging_nodes_blue, nodes_where_longest):
        """
        Remove the hanging nodes which are along the longest side of a blue refined element which has one
        marked neighbor. Therefore the neighbor along the longest side has to be either a green element or
        another blue element if the neighbir along the longest side has another neighbor.

        The algorithm closes here.
        @param hanging_nodes_blue:
        @param hanging_nodes_red:
        @param nodes_where_longest:
        @return:
        """

        blue_longest_edge = [nodes_where_longest[edge] for edge in hanging_nodes_blue]
        red_longest_edge = [nodes_where_longest[edge] for edge in hanging_nodes_red]
        all_hanging_nodes = hanging_nodes_blue + hanging_nodes_red
        direct_neighbor, marked_ele, _, marked_neighbor, _ = self.direct_neighbor(all_hanging_nodes)

        second_neighbor = self.second_neighbor(marked_ele, direct_neighbor)[0]

        # Following lines check if a red element that had be assigned because an unmarked element has two marked
        # elements not along the longest edge has a green neighbor. This is not allowed because otherwise the green
        # element has 2 neighbors.



        self.for_green_ref = [self.for_green_ref]
        self.for_blue_ref_two_neighbor = [self.for_blue_ref_two_neighbor]
        self.green_marked_neighbor = [self.green_marked_neighbor]
        self.second_blue_marked_neighbor = [self.second_blue_marked_neighbor]

        self.second_iter_ele(second_neighbor, direct_neighbor, marked_ele, blue_longest_edge)
        self.second_iter_ele(second_neighbor, direct_neighbor, marked_ele, red_longest_edge)

        self.second_blue_marked_neighbor = np.asarray(self.second_blue_marked_neighbor)



    def check_elements(self, direct_neighbor,
                       marked_ele,
                       nodes_where_longest):
        """
        This function calls all the functions to determine the green and blue elements.

        @param nodes_where_longest:
        @param direct_neighbor:
        @param marked_ele:
        @return:
        """
        green_check = self.check_for_green_blue(marked_ele, nodes_where_longest, direct_neighbor)
        second_neighbor, surrounded_ele = self.second_neighbor(marked_ele, direct_neighbor)
        hanging_nodes_red, hanging_nodes_blue = self.get_green_blue_elements(
            direct_neighbor,
            marked_ele,
            second_neighbor,
            surrounded_ele,
            nodes_where_longest)
        # self.all_marked_elements()
        print("______________________________")
        print("Found {} green element".format(len(self.for_green_ref)))
        print("______________________________")
        print("Found {} blue element".format(
            len(self.for_blue_ref_two_neighbor + self.for_blue_ref_one_neighbor)))

        return hanging_nodes_red, hanging_nodes_blue

    def rgb_iter(self, hanging_nodes):
        """
        Iteration loop for determining all hanging nodes.
        @param hanging_nodes:
        """
        if hanging_nodes:
            print("-----------------------------------")
            print("Trying to find all hanging nodes...\n\n")
            print("Remaining hanging nodes:{}".format(len(hanging_nodes)))

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
                raise ValueError("Somenthing went wrong while trying to find the mid nodes for the refinement")
        else:
            mid_nodes = np.asarray(
                [self.bcs_mesh[idx, 0].astype(np.int) for idx in idx_cluster if idx or idx == 0])
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
                c_container[i][idx] = self.mesh_undeformed[nodes[i].astype(np.int), 0:3]

        mid_node = np.divide(
            (np.add(c_container[0], c_container[1])), 2).round(decimals=6)
        return mid_node

    def search_matching_mid_nodes(self, longest_edge_nodes, nodes_where_longest):
        """
        @param: longest_edge_nodes
        @param: nodes_where_longest
        @return: edge_match
        """

        new_nodes = AMR.find_intersection(longest_edge_nodes, nodes_where_longest)
        for idx, result in enumerate(new_nodes):
            if len(result) == 2:
                return new_nodes[idx][0]
            elif new_nodes[idx]:
                return new_nodes[idx][0]

    def search_mid_point(self, longest_edge, nodes, shape):
        """

        @param edge:
        @param nodes:
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

    def keep_rotation_direction(self, nodes_neighbor, nodes):
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
        for id, row in enumerate(zip(nodes_neighbor, nodes)):
            if id == 102:
                pass
            intersection, _, indices = np.intersect1d(row[0], row[1], return_indices=True)
            idx1.append(np.where(row[0] == intersection[0])[0][0])
            idx2.append(np.where(row[0] == intersection[1])[0][0])

            keep_node.append(
                np.setxor1d(intersection, row[1])[0]
            )

        index = np.c_[idx1, idx2]

        return keep_node, index

    def rotation_direction(self, nodes_neighbor, nodes):

        keep_nodes = []
        idx = []
        for index, row in enumerate(zip(nodes, nodes_neighbor)):
            keep_nodes.append(np.setxor1d(
                np.intersect1d(row[0], row[1]), row[0])
            )
            idx.append(
                np.where(keep_nodes[index] == row[0])
            )

        rotation_index = []
        neighbor_nodes_rotation = []

        for get_rotation in idx:

            if get_rotation == 0:
                rotation_index.append([[0, 2],
                                       [1, 0],
                                       [2, 1]])

                neighbor_nodes_rotation.append([1, 2])

            if get_rotation == 1:
                rotation_index.append([[1, 0],
                                       [2, 1],
                                       [0, 2]])

                neighbor_nodes_rotation.append([2, 0])

            else:
                rotation_index.append([[2, 1],
                                       [0, 2],
                                       [1, 0]])

                neighbor_nodes_rotation.append([0, 1])

        return neighbor_nodes_rotation, rotation_index, keep_nodes

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

        mid_nodes, no_match = self.find_matching_mid_node(mid_nodes_coor, shape=3)

        neighbors = []
        for row in self.nodes_array(ele):
            all_neighbor = np.asarray(AMR.find_intersection(row, self.ele_undeformed[:, 0:3]))
            neighbor = AMR.swap_neighbor(all_neighbor)
            neighbors.append(neighbor[0])

        neighbors = np.asarray(neighbors)
        nodes = self.nodes_array(neighbors[:, 0])
        nodes_neighbor = self.nodes_array(np.asarray(neighbors)[:, 1])
        #neighbor_nodes_rotation, rotation_index, keep_nodes = self.rotation_direction(nodes_neighbor, nodes)

        for count, row_nodes in enumerate(zip(nodes_neighbor,
                                              mid_nodes,
                                              nodes)):
            self.red_ele.append(np.array(
                (row_nodes[1][0], row_nodes[1][1], row_nodes[1][2])
            ))
            self.red_ele.append(np.array(
                (row_nodes[2][2], row_nodes[1][1], row_nodes[1][2])
            ))
            self.red_ele.append(np.array(
                (row_nodes[1][0], row_nodes[2][0], row_nodes[1][2])
            ))
            self.red_ele.append(np.array(
                (row_nodes[1][0], row_nodes[2][1], row_nodes[1][1])
            ))

        """
        for i in range(len(mid_nodes)):
            self.red_ele.append(np.array(
                # x1*,                 x2*,               x3*
                (mid_nodes[i, 0], mid_nodes[i, 1], mid_nodes[i, 2]), dtype=np.int
            ))
            self.red_ele.append(np.array(
                (new_nodes[i, 0], mid_nodes[i, 0], mid_nodes[i, 1]), dtype=np.int
            ))
            self.red_ele.append(np.array(
                (mid_nodes[i, 0], new_nodes[i, 1], mid_nodes[i, 2]), dtype=np.int
            ))
            self.red_ele.append(np.array(
                (mid_nodes[i, 1], mid_nodes[i, 2], new_nodes[i, 2]), dtype=np.int
            ))
        """
        #              x3
        #            /  |  \
        #           /   |   \
        #          /    |    \
        #         /     |     \
        #        /      |      \
        #       x3*------------ x2*
        #      / \     |      / \           right turn
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
        mid_node, no_match = self.find_matching_mid_node(mid_node_c, shape=None)
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
            keep_node, index = self.keep_rotation_direction(nodes_neighbor, nodes)
        else:
            nodes_neighbor = self.nodes_array(self.green_marked_neighbor[iteration])
            keep_node, index = self.keep_rotation_direction(nodes_neighbor, nodes)

        for count, row_nodes in enumerate(zip(nodes_neighbor,
                                              np.concatenate(mid_node)
                                              )):
            self.green_ele.append(np.array(
                (row_nodes[0][index[count, 0]], row_nodes[1], keep_node[count])
            ))
            self.green_ele.append(np.array(
                (row_nodes[0][index[count, 1]], row_nodes[1], keep_node[count])
            ))

    #              x3
    #            /  |  \
    #           /   |   \
    #          /    |    \
    #         /     |     \
    #        /      |      \
    #       /       |       \
    #      /        |        \           right turn
    #     /         |         \
    #    /          |          \
    #   /           |           \
    #  /            |            \
    # x1----------x1*------------x2
    #          longeset edge
    def blue_pattern_one_neighbor(self, longest_edge, ele_one_neighbor):
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
            match_one, no_match_one, edge_match = self.search_mid_point(longest_edge, nodes_one_neighbor, shape=None)
            mid_node_c = self.calculate_mid_node(nodes_along_neighbor, len(nodes_along_neighbor))
            match_one_nle, no_match = self.find_matching_mid_node(mid_node_c, shape=None)

        except ValueError:
            raise "Blue elements can not be assigned"

        try:
            one_neighbor = np.c_[match_one_nle, match_one]

        except ValueError:
            raise 'Shape mismatch in longest edge and not longest edge in the blue element cluster'

        self.blue_pattern(one_neighbor, ele_one_neighbor, self.blue_marked_neighbor)

    def blue_pattern_two_neighbor(self, longest_edge, ele_two_neighbor, iteration):
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
            result = np.where((longest_edge == nodes).all(axis=1))[0]
            if not result:
                nodes_nle.append(nodes)

        try:
            match_two, no_match_two, edge_match = self.search_mid_point(longest_edge, nodes_two_neighbor, shape=None)
            mid_node_c = self.calculate_mid_node(nodes_nle, len(nodes_nle))
            match_two_nle, no_match_two_nle = self.find_matching_mid_node(mid_node_c, shape=None)
        except ValueError:
            raise "Blue elements can not be assigned"

        try:
            two_neighbor = np.c_[match_two, match_two_nle]
        except ValueError:
            raise 'Shape mismatch in longest edge and not longest edge in the blue element cluster'

        if iteration == 0:
            self.blue_pattern(two_neighbor, ele_two_neighbor, self.second_blue_marked_neighbor[:, 1])
        else:
            self.blue_pattern(two_neighbor, ele_two_neighbor, self.second_blue_marked_neighbor[iteration][:, 1])


    def blue_pattern(self, neighbor_stack, ele, neighbor):
        """
        This function creates the blue pattern for elements which have one or two neighbors.
        @param neighbor_stack:
        @param ele:
        @return:
        """
        nodes = self.nodes_array(ele)
        nodes_neighbor = self.nodes_array(neighbor)
        keep_node_two, index_two = self.keep_rotation_direction(nodes_neighbor, nodes)

        for count, row_nodes in enumerate(zip(nodes_neighbor,
                                              neighbor_stack)):
            self.blue_ele.append(np.array(
                (row_nodes[0][index_two[count, 1]], row_nodes[1][0], row_nodes[1][1])
            ))
            self.blue_ele.append(np.array(
                (row_nodes[0][index_two[count, 0]], row_nodes[1][1], row_nodes[1][0])
            ))
            self.blue_ele.append(np.array(
                (row_nodes[0][index_two[count, 0]], keep_node_two[count], row_nodes[1][1])
            ))
            """
                                           index)):
            self.green_ele.append(np.array(
                (row_nodes[1][index[count, 0]], row_nodes[2], keep_node[count])
            ))
            self.green_ele.append(np.array(
                (row_nodes[1][index[count, 1]], row_nodes[2], keep_node[count])
            ))



                              for i in range(len(one_neighbor)):
            self.blue_ele.append(np.array(
                (nodes[i, 2], one_neighbor[i, 1], one_neighbor[i, 0] ), dtype=np.int
            ))
            self.blue_ele.append(np.array(
                (nodes[i, 0], one_neighbor[i, 0], nodes[i, 1]), dtype=np.int
            ))
            self.blue_ele.append(np.array(
                (one_neighbor[i, 1], nodes[i, 1], one_neighbor[i, 0]), dtype=np.int
            ))
            """

    def create_all_pattern(self, nodes_where_longest, iter_num, *argv):

        """
        This function creates the new mid nodes and a template which is used to determine the corresponding
        mid nodes to the edges. Afterwards the functions to create the red, green and blue pattern are called.

        @param nodes_where_longest:
        @param not_longest_edge:
        @param iter_num:
        @param argv:
        @return:
        """

        if iter_num == 1:
            first_iter = argv[0] + argv[2]
            mid_node_coors = self.mid_nodes(first_iter)
            self.red_pattern(mid_node_coors, argv[0])
            self.green_pattern(nodes_where_longest, argv[1], 0)
            self.blue_pattern_one_neighbor(nodes_where_longest, argv[2])
            self.blue_pattern_two_neighbor(nodes_where_longest, argv[3], 0)

        else:

            for iteration, green_ele in enumerate(argv[0][1::]):
                iteration += 1
                self.green_pattern(nodes_where_longest, green_ele, iteration)

            for iteration, blue_ele in enumerate(argv[1][1::]):
                iteration += 1
                if isinstance(argv[1], list):
                    self.blue_pattern_two_neighbor(nodes_where_longest, blue_ele, iteration)

    def main_amr(self):
        """
        Main Loop for the Mesh refinement
        @return:
        """
        self.thickness_diff()
        direct_neighbor, marked_neighbor, nodes, _, three_neighbor = self.direct_neighbor(
            self.ele_list)

        # This is the main loop to determine the remaining hanging nodes

        for ele in self.ele_list:
            self.for_red_ref.append(ele)

        for for_del in sorted(three_neighbor, reverse=True):
            idx = np.where(marked_neighbor == self.for_red_ref[for_del])[0]
            for idx_to_delete in sorted(idx, reverse=True):
                marked_neighbor = np.delete(marked_neighbor, idx_to_delete)
                direct_neighbor = np.delete(direct_neighbor, idx_to_delete)
            del self.for_red_ref[for_del]

        print("______________________________")
        print("Found {} red element".format(len(self.ele_list)))
        nodes_where_longest, not_longest_edge = self.get_ele_length(range(0, len(self.ele_undeformed)))

        hanging_nodes_red = []
        hanging_nodes_blue = []
        iter_num = 0
        while iter_num < 2:
            hnr, hnb = self.check_elements(direct_neighbor,
                                           marked_neighbor,
                                           nodes_where_longest)
            for nodes_r in hnr:
                hanging_nodes_red.append(nodes_r)
            for nodes_b in hnb:
                hanging_nodes_blue.append(nodes_b)
            self.rgb_iter(hanging_nodes_red + hanging_nodes_blue)

            if len(hanging_nodes_red + hanging_nodes_blue) != 0:
                print("______________________________")
                print("Found {} red element".format(len(self.for_red_ref)))
            else:
                print("-----------------------------------")
                print("Eliminated all hanging nodes!")
                break
            iter_num += 1
            self.create_all_pattern(nodes_where_longest,
                                    iter_num,
                                    self.for_red_ref,
                                    self.for_green_ref,
                                    self.for_blue_ref_one_neighbor,
                                    self.for_blue_ref_two_neighbor)
            self.all_marked_elements()
            self.remove_hanging_nodes(hanging_nodes_red,
                                      hanging_nodes_blue,
                                      nodes_where_longest)
            iter_num = 2
            self.create_all_pattern(nodes_where_longest,
                                    iter_num,
                                    self.for_green_ref,
                                    self.for_blue_ref_two_neighbor,
                                    self.for_blue_ref_one_neighbor)
            break


class write_file():
    def __init__(self, obj, out_path):
        self._out_path = out_path

        self.ele_undeformed = obj.ele_undeformed
        self.ele_deformed = obj.ele_deformed
        self.mesh_undeformed = obj.mesh_undeformed
        self.mesh_deformed = obj.ele_deformed
        self.bc = obj.bc

        self.for_red_ref = obj.for_red_ref
        self.for_green_ref = obj.for_green_ref
        self.for_blue_ref_one_neighbor = obj.for_blue_ref_one_neighbor
        self.for_blue_ref_two_neighbor = obj.for_blue_ref_two_neighbor

        self.blue_ele = obj.blue_ele
        self.green_ele = obj.green_ele
        self.red_ele = obj.red_ele

        self.bcs_mesh = obj.bcs_mesh

        self.check_out_path = out_path

    @property
    def check_out_path(self):
        """
        Assigning the output path
        @return: self.__check_out_path
        """
        return self.__check_out_path

    @check_out_path.setter
    def check_out_path(self, path):
        if isinstance(path, str):
            self.__check_out_path = path
        else:
            raise TypeError("Error while assigning the output folder")

    def check_path(self):
        """
        Check if the output path folder ends with /out
        @return:
        """
        if self.check_out_path.endswith("/out"):
            pass
        else:
            raise TypeError("Wrong output folder")

    def manipulate_ele(self):
        """
        Update the old ndarray with the unrefined elements. Also keep thickness and temperature of the marked and
        refined elements the same. Delete the elements from the mesh which are marked
        """
        for_green_ref = []
        for_blue_ref_two_neighbor = []

        for all_ele in self.for_green_ref:
            for_green_ref += all_ele
        for all_ele in self.for_blue_ref_two_neighbor:
            for_blue_ref_two_neighbor += all_ele

        # for_green_ref = self.for_green_ref
        # for_blue_ref_two_neighbor = self.for_blue_ref_two_neighbor

        thickness_temp_red = np.repeat(self.ele_undeformed[self.for_red_ref, 3::], 4, axis=0)
        thickness_temp_green = np.repeat(self.ele_undeformed[for_green_ref, 3::], 2, axis=0)
        thickness_temp_blue = np.repeat(
            self.ele_undeformed[self.for_blue_ref_one_neighbor + for_blue_ref_two_neighbor, 3::],
            3, axis=0)

        self.green_ele, self.red_ele = np.asarray(self.green_ele), np.asarray(self.red_ele)
        complete_red_cluster = np.hstack((self.red_ele, thickness_temp_red))
        complete_green_cluster = np.hstack((self.green_ele, thickness_temp_green))
        complete_blue_cluster = np.hstack((self.blue_ele, thickness_temp_blue))
        self.ele_undeformed = np.delete(self.ele_undeformed,
                                        [self.for_red_ref +
                                         for_green_ref
                                         ],
                                        axis=0)

        self.ele_undeformed = np.append(self.ele_undeformed, np.concatenate(
            (complete_green_cluster, complete_red_cluster), axis=0), axis=0)

    def append_mesh(self):
        """
        Append the new mid node coordinates to the mesh.
        @return:
        """

        complete_mesh_cluster = np.hstack((self.bcs_mesh[:, 1::], np.zeros((len(self.bcs_mesh), 2), dtype=np.int)))
        self.mesh_undeformed = np.append(self.mesh_undeformed, complete_mesh_cluster, axis=0)

    def write_file(self):
        """
        Write the new file.
        @return:
        """
        self.file_name = "Undeformed_refined_mesh2.bcs"
        file_length_ele = len(self.ele_undeformed)
        file_length_mesh = len(self.mesh_undeformed)
        file_length_bc = len(self.bc)
        filter = [
            "B-SIM - DATA OF THE SHEET\n",
            "FULL\n",
            "-111 1 1 1 1 1 1 END NOP\n",
            "200.0  Char. dist\n",
            "-111 1 1 1 1 1 1 END OF COORS\n",
            "-111 1 1 1 1 1 1 END OF BCs\n",
        ]
        with open(os.path.join(self._out_path, self.file_name), "w") as bcs_amf:
            bcs_amf.write(filter[0])
            bcs_amf.write(filter[1])

            # print(self.ele_undeformed)
            for ele in range(file_length_ele):
                bcs_amf.write(
                    "{:5d}{:7d}{:7d}{:7d}{:16.6f}{:16.6f}\n".format(
                        ele + 1,
                        int(self.ele_undeformed[ele, 0]),
                        int(self.ele_undeformed[ele, 1]),
                        int(self.ele_undeformed[ele, 2]),
                        self.ele_undeformed[ele, 3],
                        self.ele_undeformed[ele, 4],

                    )
                )

            bcs_amf.write(filter[2])
            bcs_amf.write(filter[3])

            for ele in range(file_length_mesh):
                bcs_amf.write(
                    "{:5d}{:16.6f}{:16.6f}{:16.6f}{:2d}{:2d}\n".format(
                        ele + 1,
                        self.mesh_undeformed[ele, 0],
                        self.mesh_undeformed[ele, 1],
                        self.mesh_undeformed[ele, 2],
                        self.mesh_undeformed[ele, 3].astype(np.int),
                        self.mesh_undeformed[ele, 4].astype(np.int),

                    )
                )

            bcs_amf.write(filter[4])
            for bc in range(file_length_bc):
                bcs_amf.write("{:5d}{:6d}\n".format(
                    self.bc[bc, 0].astype(np.int), self.bc[bc, 1].astype(np.int)))
            bcs_amf.write(filter[4])
            bcs_amf.close

    def check_success(self):
        """
        Checks if the file exisits.
        @return:
        """
        if os.path.exists(os.path.join(self.__check_out_path, self.file_name)):
            print("Success!!")
        else:
            raise RuntimeError("Did not complete")
