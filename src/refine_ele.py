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
import time


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
        self.for_blue_ref = []
        self.green_marked_neighbor = []
        self.blue_marked_neighbor = []

        self.red_ele = []
        self.green_ele = []
        self.blue_ele = []

        self.blue_ele_edges = []
        self.green_ele_edges = []
        self.blue_longest_edge = []

        self.all_ele = []
        self.red_mid_nodes = {}
        self.blue_mid_nodes = {}
        self.green_mid_nodes = {}

        self.bcs_mesh = None

    def run_marking(self):
        """
        Run marking algorithm

        @return:
        """
        super().run_marking()
        print("Found {} elements to refine".format(len(self.marked_ele)))

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

        edges = np.c_[index, edges]
        return edges

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

        neighbor_nodes = self.nodes_array(marked_ele)

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

    def get_ele_dictionary(self, hanging_edges, nodes_where_longest):
        """
        Creating a dictionary which stores the element edge end corresponding Element numbers. Set
        marked edges to true, otherwise false. A second dictionary stores the longest edge of each elements edge.

        @param hanging_edges:
        @param nodes_where_longest:
        @return:
        """

        longest_edge_dict = {}
        ele_num = 0
        fit_longest_edge = np.repeat(nodes_where_longest, 3, axis=0)

        for i, (element_val, edge) in enumerate(self.ele_dict.items()):
            if element_val not in longest_edge_dict:
                longest_edge_dict[element_val] = {"Longest_edge": fit_longest_edge[i]}

            if (i + 1) % 3 == 0:
                ele_num += 1

        for marked_lst in hanging_edges[:, 1::]:
            marked_edge = tuple(marked_lst)
            reversed_edge = tuple(sorted(marked_edge, reverse=True))
            if marked_edge in self.ele_dict:
                self.ele_dict[marked_edge]["Marked"] = True
            if reversed_edge in self.ele_dict:
                self.ele_dict[reversed_edge]["Marked"] = True

        return longest_edge_dict

    def elements_to_refine(self, hanging_edges, longest_edge_dict):
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

        @param hanging_edges:
        @param longest_edge_dict:
        @return:
        """


        for edges in hanging_edges[:, 1::]:
            edges = tuple(edges)
            reversed_edge = tuple(reversed(edges))
            if reversed_edge in self.ele_dict:
                self.ele_dict[reversed_edge]["Marked"] = True
                longest_edge = tuple(longest_edge_dict[reversed_edge]["Longest_edge"])
                adjacent_edge = edges
                if longest_edge:
                    while True:
                        if np.isin(adjacent_edge, longest_edge).all():
                            break

                        if longest_edge not in self.ele_dict:
                            raise KeyError(
                                "Hanging nodes {} at the boundary the clamping area. Please choose another threshold for "
                                "the refinement.".format(longest_edge)
                            )

                        if not self.ele_dict[longest_edge]["Marked"]:
                            self.ele_dict[longest_edge]["Marked"] = True
                            adjacent_edge = tuple(reversed(longest_edge))
                            if adjacent_edge in self.ele_dict:
                                self.ele_dict[adjacent_edge]["Marked"] = True
                            if adjacent_edge in longest_edge_dict:
                                longest_edge = tuple(
                                    longest_edge_dict[adjacent_edge]["Longest_edge"]
                                )
                        else:
                            break

    def count_marked_edges(self):
        """
        Counts the marked edges per element and assigns the number to corresponding instance variables

        @return: marked_dict
        """

        marked_dict = {}
        for index, (edge, val) in enumerate(self.ele_dict.items()):
            ele_number = val["Ele_num"]
            marked = val["Marked"]
            if ele_number not in marked_dict:
                marked_dict[ele_number] = {
                    "Ele_num": ele_number,
                    "Count": 0,
                    "Green_mark": False,
                    "Red_mark": False,
                    "Blue_mark": False,
                    "Edge": [],
                }

            if marked:
                marked_dict[ele_number]["Edge"].append(edge)
                marked_dict[ele_number]["Count"] += 1

        for _, val in marked_dict.items():
            count = val["Count"]
            ele_num = val["Ele_num"]

            if count == 1:
                self.for_green_ref.append(ele_num)
                marked_dict[ele_num]["Green_mark"] = True
            if count == 2:
                self.for_blue_ref.append(ele_num)
                marked_dict[ele_num]["Blue_mark"] = True

            if count == 3:
                self.for_red_ref.append(ele_num)
                marked_dict[ele_num]["Red_mark"] = True

        return marked_dict

    def get_marked_neighbor(self, marked_dict):
        """
        Assign the neighbors of marked elements. Neighboring edges are edges with a reversed edge order. The
        reversed tuple is present in the element dictionary and therefore easy to find.

        @param marked_dict:
        @return:
        """

        for _, val in marked_dict.items():
            count = val["Count"]
            edges = val["Edge"]
            if count and count < 3:
                neighbor = []
                for edge in edges:
                    neighbor.append(tuple(reversed(edge)))

                if count == 1:
                    if neighbor[0] in self.ele_dict:
                        self.green_marked_neighbor.append(
                            self.ele_dict[neighbor[0]]["Ele_num"]
                        )

                if count == 2:
                    for ne in neighbor:
                        if ne not in self.ele_dict:
                            raise KeyError(
                                "Hanging nodes at the boundary the clamping area. Please choose another threshold for "
                                "the refinement."
                            )
                    else:
                        self.blue_marked_neighbor.append(
                            [
                                self.ele_dict[neighbor[0]]["Ele_num"],
                                self.ele_dict[neighbor[1]]["Ele_num"],
                            ]
                        )
        self.blue_marked_neighbor = np.asarray(self.blue_marked_neighbor)

    def get_new_mid_nodes(self, marked_dict, longest_edge_dict):
        """
        Calculate new mid node coordinates for hanging edges and assign them to a dictionary.

        @param marked_dict:
        @param longest_edge_dict:
        @return:mid_node
        """

        node_num = len(self.mesh_undeformed)
        mid_node_dict = {}
        for _, val in marked_dict.items():
            count = val["Count"]
            ele_num = val["Ele_num"]
            edges = val["Edge"]

            if edges:
                longest_edge = longest_edge_dict[edges[0]]["Longest_edge"]
                if count == 2 or count == 3:
                    for edge in edges:
                        edge = tuple(sorted(edge))
                        if count == 3:
                            if edge not in mid_node_dict:
                                mid_node_coor, node_num = self.calculate_mid_nodes(
                                    edge, node_num
                                )
                                mid_node_dict[edge] = {
                                    "Mid_nodes": node_num,
                                    "Coordinates": tuple(mid_node_coor),
                                    "Ele_num": ele_num,
                                }

                        if count == 2:
                            if np.isin(longest_edge, edge).all():
                                if edge not in mid_node_dict:
                                    mid_node_coor, node_num = self.calculate_mid_nodes(
                                        edge, node_num
                                    )
                                    mid_node_dict[edge] = {
                                        "Mid_nodes": node_num,
                                        "Coordinates": tuple(mid_node_coor),
                                        "Ele_num": ele_num,
                                    }

            self.bcs_mesh = [entry["Coordinates"] for entry in mid_node_dict.values()]

        return mid_node_dict

    def calculate_mid_nodes(self, edge, node_num):
        """
        Actual calculation of the mid node.

        @param edge:
        @param node_num:
        @return: mid_node_coot, node_num
        """

        node_1 = self.mesh_undeformed[edge[0] - 1, 0:3]
        node_2 = self.mesh_undeformed[edge[1] - 1, 0:3]
        mid_node_coor = np.divide((np.add(node_1, node_2)), 2).round(decimals=6)
        node_num += 1

        return mid_node_coor, node_num

    def assign_mid_nodes(self, mid_node_dict, marked_dict, longest_edge_dict):
        """
        Assign the calculated mid nodes to the corresponfing adjacent edges.

        @param mid_node_dict:
        @param marked_dict:
        @param longest_edge_dict:
        @return:
        """

        green_dict = {
            key: value for key, value in marked_dict.items() if value["Green_mark"]
        }
        red_dict = {
            key: value for key, value in marked_dict.items() if value["Red_mark"]
        }
        blue_dict = {
            key: value for key, value in marked_dict.items() if value["Blue_mark"]
        }

        for _, row in green_dict.items():
            edges = tuple(row["Edge"][0])
            ele_num = row["Ele_num"]
            longest_edge = tuple(sorted(longest_edge_dict[edges]["Longest_edge"]))
            green_edges = tuple(sorted(edges))

            if ele_num not in self.green_mid_nodes:
                self.green_mid_nodes[ele_num] = {
                    "Mid_node": mid_node_dict[green_edges]["Mid_nodes"],
                    "Ele_num": ele_num,
                    "Edges": green_edges,
                    "Longest_edge": longest_edge,
                }

        for _, row in red_dict.items():
            red_edges = tuple(row["Edge"])
            ele_num = row["Ele_num"]

            red_edges = [tuple(sorted(edge)) for edge in red_edges]

            if ele_num not in self.red_mid_nodes:
                self.red_mid_nodes[ele_num] = {
                    "Mid_node": [
                        mid_node_dict[red_edges[0]]["Mid_nodes"],
                        mid_node_dict[red_edges[1]]["Mid_nodes"],
                        mid_node_dict[red_edges[2]]["Mid_nodes"],
                    ],
                    "Ele_num": ele_num,
                    "Edges": red_edges,
                }

        for _, row in blue_dict.items():
            blue_edges = tuple(row["Edge"])
            ele_num = row["Ele_num"]
            blue_edges = [tuple(edge) for edge in blue_edges]

            longest_edge = tuple(
                sorted(longest_edge_dict[blue_edges[0]]["Longest_edge"])
            )
            for edge in blue_edges:
                if edge != longest_edge and tuple(reversed(edge)) != longest_edge:
                    not_longest_edge = tuple(sorted(edge))

            self.blue_mid_nodes[ele_num] = {
                "Mid_node": [
                    mid_node_dict[longest_edge]["Mid_nodes"],
                    mid_node_dict[not_longest_edge]["Mid_nodes"],
                ],
                "Edges": blue_edges,
                "Longest_edge": longest_edge,
            }

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

    @staticmethod
    def keep_rotation_direction(nodes_neighbor, nodes, nodes_where_longest):
        """
        Neighboring nodes change their order to keep the rotation direction. Therefore it's very important
        to place the nodes at the right position, because they differ depending on the neighbor node position.

        1. Get the index of the intersection between the nodes neighbors (which are the nodes of a marked elements
           neighbors) and nodes (which are the nodes of the newly marked element f.e. a green element).
        2. There is always one node in the nodes variable which has no intersection, because there's only an
           intersection between an adjacent edge. This node is the "keep node" because it's the vertex node
           opposite of the marked edge.
        3. Get the correct nodes rotation by calling the function nodes_rotation. Follow the docstrings of
           nodes_rotation for more insights.

        @param nodes_neighbor:
        @param nodes:
        @param nodes_where_longest:
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
        for idx, elements in enumerate(nodes_where_longest):
            vertex_node.append(int(np.setxor1d(elements, nodes[idx])[0]))

        return keep_node, index, keep_node_index, vertex_node, node_rotation

    def find_vertex_and_mid_node(self, nodes, neighbor_one, neighbor_two):
        """
        Different approach than the function "keep_rotation_direction". For the blue refinement
        there are two possible way of creating the element. This is due to the location of the longest
        edge. There are multiple steps:

        1.Get the edge which is not marked
        2.Get the node which connects both of the marked edges.
        3.Get the node which is the vertex of the edge which is NOT the longest edge.
        4.Get the node which connects the unmarked edge and the edge which is not the longest.
        5.Function "nodes_rotation" checks whats the correct rotation of the nodes. The second node
        follows after the first node. Therefore it's possible to check if the vertex node occurs at the first
        or second position of the rotation_direction list. This is an indicator for the correct rotation direction
        of the new build blue element.

        @param nodes:
        @param neighbor_one:
        @param neighbor_two:
        @return: marked_edge_connecting_node, vertex_node, free_node, index_differ
        """

        vertex_node = []
        unmarked_edge = []
        free_node = []
        marked_edge_connecting_node = []
        marked_edge_connecting_node_index = []
        index_differ = []
        for row in range(len(nodes)):
            unmarked_edge.append(
                np.setxor1d(neighbor_one[row], neighbor_two[row]).astype(np.int)
            )

            marked_edge_connecting_node.append(
                np.setxor1d(unmarked_edge[row], nodes[row])[0]
            )

            marked_edge_connecting_node_index.append(
                np.where(marked_edge_connecting_node[row] == nodes[row])[0][0]
            )

            vertex_node.append(np.intersect1d(unmarked_edge[row], neighbor_one[row])[0])
            free_node.append(np.intersect1d(neighbor_two[row], unmarked_edge[row])[0])

        rotation_direction = self.nodes_rotation(
            marked_edge_connecting_node_index, nodes
        )

        for idx, rotation in enumerate(rotation_direction):
            index_differ.append(np.where(rotation == vertex_node[idx])[0][0])

        return marked_edge_connecting_node, vertex_node, free_node, index_differ

    def red_pattern(self):
        """
        This function creates the red pattern.
        """
        mid_nodes = [entry["Mid_node"] for entry in self.red_mid_nodes.values()]

        nodes = self.nodes_array(self.for_red_ref)

        for enum, (mn, ne) in enumerate(zip(mid_nodes, nodes)):
            self.red_ele.append(np.array((mn[0], mn[1], mn[2])))
            self.red_ele.append(np.array((ne[2], mn[2], mn[1])))
            self.red_ele.append(np.array((mn[2], ne[0], mn[0])))
            self.red_ele.append(np.array((mn[0], ne[1], mn[1])))

    def green_pattern(self):
        """
        This function creates the green pattern.

        """

        mid_node = [entry["Mid_node"] for entry in self.green_mid_nodes.values()]
        nodes_where_longest = [
            entry["Longest_edge"] for entry in self.green_mid_nodes.values()
        ]

        nodes = self.nodes_array(self.for_green_ref)
        nodes_neighbor = self.nodes_array(self.green_marked_neighbor)
        keep_node, _, _, _, nodes_longest_edge = AMR.keep_rotation_direction(
            nodes_neighbor, nodes, nodes_where_longest
        )

        for count, (nle, mn) in enumerate(zip(nodes_longest_edge, mid_node)):
            self.green_ele.append(np.array((mn, keep_node[count], nle[0])))
            self.green_ele.append(np.array((mn, nle[1], keep_node[count])))

    def blue_pattern(self):
        """
        This function creates the blue pattern.

        """

        nodes = self.ele_undeformed[self.for_blue_ref, 0:3]
        edges = [val["Edges"] for _, val in self.blue_mid_nodes.items()]

        longest_edge = [val["Longest_edge"] for _, val in self.blue_mid_nodes.items()]
        not_longest_edge = []

        for row, le in zip(edges, longest_edge):
            for edge in row:
                if edge != le and tuple(reversed(edge)) != le:
                    not_longest_edge.append(edge)

        mid_nodes = [val["Mid_node"] for _, val in self.blue_mid_nodes.items()]

        (
            marked_edge_connecting_node,
            vertex_node,
            free_node,
            index_differ,
        ) = self.find_vertex_and_mid_node(nodes, longest_edge, not_longest_edge)

        for count, (cn, fn, vn, index_diff) in enumerate(
            zip(marked_edge_connecting_node, free_node, vertex_node, index_differ)
        ):
            if not index_diff == 1:
                self.blue_ele.append(
                    np.array((cn, mid_nodes[count][0], mid_nodes[count][1]))
                )
                self.blue_ele.append(
                    np.array((mid_nodes[count][1], mid_nodes[count][0], vn))
                )
                self.blue_ele.append(np.array((fn, mid_nodes[count][1], vn)))
            else:
                self.blue_ele.append(
                    np.array((cn, mid_nodes[count][1], mid_nodes[count][0]))
                )
                self.blue_ele.append(
                    np.array((mid_nodes[count][1], vn, mid_nodes[count][0]))
                )
                self.blue_ele.append(np.array((fn, vn, mid_nodes[count][1])))

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

    def find_elements_to_refine(self, nodes_where_longest):
        """
        Main function for the RGB-refinement and to determine the new mid nodes of
        red and blue elements.

        @param marked_edges:
        @param nodes_where_longest:
        @return:
        """
        hanging_edges = self.long_stacked_edges_array(self.marked_ele)
        longest_edge_dict = self.get_ele_dictionary(hanging_edges, nodes_where_longest)
        self.elements_to_refine(hanging_edges, longest_edge_dict)
        marked_dict = self.count_marked_edges()
        self.get_marked_neighbor(marked_dict)

        mid_node_dict = self.get_new_mid_nodes(marked_dict, longest_edge_dict)
        self.assign_mid_nodes(mid_node_dict, marked_dict, longest_edge_dict)

        return marked_dict, longest_edge_dict

    def create_all_pattern(self):
        """
        This function concatenates all pattern creations.
        """

        self.red_pattern()
        self.green_pattern()
        self.blue_pattern()

    def print_information(self):
        """
        Prints information about the refinement.
        @return:
        """

        print("-------------------------------------------------------")
        print("Marked {} red elements".format(len(self.for_red_ref)))
        print("Marked {} blue elements".format(len(self.for_blue_ref)))
        print("Marked {} green_elements".format(len(self.for_green_ref)))
        print("-------------------------------------------------------")

    def main_amr(self):
        """
        Main function
        @return:
        """

        self.run_marking()
        nodes_where_longest, all_edges, marked_edges = self.get_longest_edge()
        self.find_elements_to_refine(nodes_where_longest)
        self.print_information()
        self.create_all_pattern()
