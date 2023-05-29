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
        self.for_blue_ref= []
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

    def get_ele_dictionary(self, hanging_edges, all_edges, nodes_where_longest):
        """
        Creating a dictionary which stored the element edge end corresponding Element numbers. Set
        marked edges to true, otherwise false. A second dictionary stores the longest edge of each elements edge.

        @param hanging_edges:
        @param all_edges:
        @param nodes_where_longest:
        @return:
        """

        ele_dict = {}
        longest_edge_dict = {}
        ele_num = 0
        fit_longest_edge = np.repeat(nodes_where_longest, 3, axis=0)

        for i, edge in enumerate(all_edges[:, 1:]):
            element_val = tuple(edge)

            if element_val not in ele_dict:
                ele_dict[element_val] = {"Edge": element_val,
                                         "Ele_num": ele_num,
                                         "Marked": False,
                                         }
            if element_val not in longest_edge_dict:
                longest_edge_dict[element_val] = {"Longest_edge": fit_longest_edge[i]}

            if (i + 1) % 3 == 0:
                ele_num += 1

        for marked_lst in hanging_edges[:, 1::]:
            marked_edge = tuple(marked_lst)
            reversed_edge = tuple(sorted(marked_edge, reverse=True))
            if marked_edge in ele_dict:
                ele_dict[marked_edge]["Marked"] = True
            if reversed_edge in ele_dict:
                ele_dict[reversed_edge]["Marked"] = True

        return ele_dict, longest_edge_dict


    def elements_to_refine(self, ele_dict, hanging_edges, longest_edge_dict):
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

        for edges in hanging_edges[:, 1::]:
            edges = tuple(edges)
            reversed_edge = tuple(reversed(edges))
            if reversed_edge in ele_dict:
                ele_dict[reversed_edge]["Marked"] = True
                longest_edge = tuple(longest_edge_dict[reversed_edge]["Longest_edge"])
                if np.isin(edges, (3,86)).all():
                    pass
                if np.isin(longest_edge, (3,86)).all():
                    pass
                if np.isin(longest_edge, (3,86)).all():
                    pass
                adjacent_edge = edges
                if longest_edge:
                    while True:
                        if np.isin(adjacent_edge, longest_edge).all():
                            break
                        if not ele_dict[longest_edge]["Marked"]:
                            ele_dict[longest_edge]["Marked"] = True
                            adjacent_edge = tuple(reversed(longest_edge))
                            if adjacent_edge in ele_dict:
                                ele_dict[adjacent_edge]["Marked"] = True
                            if adjacent_edge in longest_edge_dict:
                                longest_edge = tuple(longest_edge_dict[adjacent_edge]["Longest_edge"])
                                if np.isin(longest_edge, (3,86)).all():
                                    pass
                                if np.isin(longest_edge, (3,86)).all():
                                    pass
                        else:
                            break

        marked_dict = {}
        for index, val in enumerate(ele_dict.values()):
            ele_number = val["Ele_num"]
            marked = val["Marked"]
            edge = val["Edge"]
            if ele_number not in marked_dict:
                marked_dict[ele_number] = {"Ele_num": ele_number,
                                           "Count": 0,
                                           "Green_mark": False,
                                           "Red_mark": False,
                                           "Blue_mark": False,
                                           "Edge": []
                                        }

            if marked:
                marked_dict[ele_number]["Edge"].append(edge)
                marked_dict[ele_number]["Count"] += 1

        for _, val in marked_dict.items():
            count = val["Count"]
            ele_num = val["Ele_num"]

            if count == 1:
                self.for_green_ref.append(
                    ele_num
                )
                marked_dict[ele_num]["Green_mark"] = True
            if count == 2:
                self.for_blue_ref.append(
                    ele_num
                )
                marked_dict[ele_num]["Blue_mark"] = True

            if count == 3:
                self.for_red_ref.append(
                    ele_num
                )
                marked_dict[ele_num]["Red_mark"] = True

        return marked_dict

    def get_marked_neighbor(self, marked_dict, ele_dict):
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

        for key, val in marked_dict.items():
            count = val["Count"]
            edges = val["Edge"]

            if count and count < 3:
                neighbor = []
                for edge in edges:
                    neighbor.append(
                        tuple(reversed(edge))
                    )

                if count == 1:
                    if neighbor[0] in ele_dict:
                        self.green_marked_neighbor.append(
                            ele_dict[neighbor[0]]["Ele_num"]
                        )

                if count == 2:
                    if neighbor[0] and neighbor[1] in ele_dict:
                        print(ele_dict[neighbor[1]])
                        #print(neighbor[0], count)
                        self.blue_marked_neighbor.append(
                            [ele_dict[neighbor[0]]["Ele_num"],
                             ele_dict[neighbor[1]]["Ele_num"]]
                        )
        self.blue_marked_neighbor= np.asarray(self.blue_marked_neighbor)

    def get_new_mid_nodes(self, ele_dict, marked_dict, longest_edge_dict):
        """
        Calculate new mid node coordinates for hanging edges.
        @param ele_dict:
        @param marked_dict:
        @param longest_edge_dict:
        @return:
        """

        node_num = len(self.mesh_undeformed)
        mid_node_dict = {}
        for key, val in marked_dict.items():
            edges = val["Edge"]
            count = val["Count"]
            ele_num = val["Ele_num"]

            if edges:
                longest_edge = longest_edge_dict[edges[0]]["Longest_edge"]
                if count == 2 or count == 3:
                    for edge in edges:
                        edge = tuple(sorted(edge))
                        if count == 3:
                            if edge not in mid_node_dict:
                                mid_node_coor, node_num = self.calculate_mid_nodes(edge, node_num)
                                mid_node_dict[edge] = {"Mid_nodes": node_num,
                                                       "Coordinates": tuple(mid_node_coor),
                                                       "Ele_num": ele_num,
                                                       }

                        if count == 2:
                            if np.isin(longest_edge, edge).all():
                                if edge not in mid_node_dict:
                                    mid_node_coor, node_num = self.calculate_mid_nodes(edge, node_num)
                                    mid_node_dict[edge] = {"Mid_nodes": node_num,
                                                           "Coordinates": tuple(mid_node_coor),
                                                           "Ele_num": ele_num,
                                                           }

            self.bcs_mesh = [entry['Coordinates'] for entry in mid_node_dict.values()]
        return mid_node_dict

    def assign_mid_nodes(self, mid_node_dict, ele_dict, marked_dict, longest_edge_dict):
        """
        Assign the calculated mid nodes to the corresponfing adjacent edges.
        @param mid_node_dict:
        @param ele_dict:
        @param marked_dict:
        @param longest_edge_dict:
        @return:
        """

        green_dict = {key: value for key, value in marked_dict.items() if value["Green_mark"]}
        red_dict = {key: value for key, value in marked_dict.items() if value["Red_mark"]}
        blue_dict = {key: value for key, value in marked_dict.items() if value["Blue_mark"]}


        for _, row in green_dict.items():
            edges = tuple(row["Edge"][0])
            ele_num = row["Ele_num"]
            longest_edge = tuple(sorted(longest_edge_dict[edges]["Longest_edge"]))
            green_edges = tuple(sorted(edges))

            if ele_num not in self.green_mid_nodes:
                self.green_mid_nodes[ele_num] = {"Mid_node": mid_node_dict[green_edges]["Mid_nodes"],
                                                 "Ele_num": ele_num,
                                                 "Edges": green_edges,
                                                 "Longest_edge": longest_edge
                                                 }


        for _, row in red_dict.items():
            red_edges = row["Edge"]
            ele_num = row["Ele_num"]

            red_edges = [tuple(sorted(edge)) for edge in red_edges]

            if ele_num not in self.red_mid_nodes:
                self.red_mid_nodes[ele_num] = {"Mid_node": [mid_node_dict[red_edges[0]]["Mid_nodes"],
                                                            mid_node_dict[red_edges[1]]["Mid_nodes"],
                                                            mid_node_dict[red_edges[2]]["Mid_nodes"]
                                                            ],
                                               "Ele_num": ele_num,
                                               "Edges": red_edges
                                               }

        for index, (_, row) in enumerate(blue_dict.items()):
            if index == 45:
                pass
            blue_edges = row["Edge"]
            ele_num = row["Ele_num"]
            blue_edges = [tuple(edge) for edge in blue_edges]

            longest_edge = tuple(sorted(longest_edge_dict[blue_edges[0]]["Longest_edge"]))
            for edge in blue_edges:
                if edge != longest_edge and tuple(reversed(edge)) != longest_edge:
                    not_longest_edge = tuple(sorted(edge))

            self.blue_mid_nodes[ele_num] = {"Mid_node": [mid_node_dict[longest_edge]["Mid_nodes"],
                                                         mid_node_dict[not_longest_edge]["Mid_nodes"]],
                                            "Edges": blue_edges,
                                            "Longest_edge": longest_edge
                                            }


    def calculate_mid_nodes(self, edge, node_num):

        node_1 = self.mesh_undeformed[edge[0] -1 , 0:3]
        node_2 = self.mesh_undeformed[edge[1] -1, 0:3]
        mid_node_coor = np.divide((np.add(node_1, node_2)), 2).round(
            decimals=6
        )
        node_num += 1

        return mid_node_coor, node_num

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
        # self.bcs_mesh = np.hstack((node_axis[:, np.newaxis], bcs_mesh))

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

        # match = list(map(lambda x: x - 1, match))

        for i in range(2):
            for idx, nodes in enumerate(match):
                c_container[i][idx] = self.mesh_undeformed[nodes[i].astype(np.int), 0:3]

        mid_node = np.divide((np.add(c_container[0], c_container[1])), 2).round(
            decimals=6
        )

        return mid_node

    @staticmethod
    def keep_rotation_direction(nodes_neighbor, nodes, nodes_where_longest):
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
        for idx, elements in enumerate(nodes_where_longest):
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
            self, nodes, neighbor_one, neighbor_two, ele_dict
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

            marked_edge_connecting_node.append(np.setxor1d(unmarked_edge[row], nodes[row])[0])

            marked_edge_connecting_node_index.append(
                np.where(marked_edge_connecting_node[row] == nodes[row])[0][0]
            )

            vertex_node.append(
                np.intersect1d(unmarked_edge[row], neighbor_one[row])[0]
            )
            free_node.append(np.intersect1d(neighbor_two[row], unmarked_edge[row])[0])

        rotation_direction = self.nodes_rotation(marked_edge_connecting_node_index, nodes)

        for idx, rotation in enumerate(rotation_direction):
            index_differ.append(
                np.where(rotation == vertex_node[idx])[0][0]
            )


        """
        for row in range(len(nodes)):
            #unmarked_edge.append(
            #    np.setxor1d(neighbor_one[row], neighbor_two[row]).astype(np.int)
            #)

            free_node.append(np.setxor1d(neighbor_two[row], nodes[row])[0])

            marked_edge_connecting_node.append(
                np.intersect1d(neighbor_one[row], neighbor_two[row])[0]
            )

            vertex_node.append(
                np.setxor1d(neighbor_one[row], nodes[row])[0]
            )

            vertex_node_index.append(
                np.where(nodes[row] == vertex_node[-1])[0]
            )
        nodes_rotation = AMR.nodes_rotation(vertex_node_index, nodes)

        for idx, rotation in enumerate(nodes_rotation):
            index_differ.append(
                np.where(rotation == marked_edge_connecting_node[idx])[0]
            )

  
        
        for (ele_node, le, nle) in zip(nodes, neighbor_one, neighbor_two):
            if 516 in ele_node:
                pass
            tuple_container = [tuple([ele_node[0], ele_node[1]]),
                               tuple([ele_node[1], ele_node[2]]),
                               tuple([ele_node[2], ele_node[0]])
                               ]
            index_le = [index for index, item in enumerate(tuple_container) if np.isin(item, le).all()]
            index_nle = [index for index, item in enumerate(tuple_container) if np.isin(item, nle).all()]

            if index_le < index_nle:
                index_differ.append(2)
            else:
                if index_le[0] == 1 and index_nle[0] == 2:
                    index_le = [0]

                index_differ.append(
                    [abs(x - y) for x, y in zip(index_le, index_nle)][0]
                )
        
        """

        return marked_edge_connecting_node, vertex_node, free_node, index_differ

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

    def red_pattern(self):
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

        # mid_nodes = self.get_mid_nodes(None, mid_nodes_coor, shape=3)
        mid_nodes = [entry['Mid_node'] for entry in self.red_mid_nodes.values()]

        nodes = self.nodes_array(self.for_red_ref)

        for enum, (mn, ne) in enumerate(zip(mid_nodes, nodes)):
            self.red_ele.append(np.array((mn[0], mn[1], mn[2])))
            self.red_ele.append(np.array((ne[2], mn[2], mn[1])))
            self.red_ele.append(np.array((mn[2], ne[0], mn[0])))
            self.red_ele.append(np.array((mn[0], ne[1], mn[1])))

    def green_pattern(self):
        """
        There are two main operations in this function. The first loop searches the two connected nodes with the longest
        edge in the element. The function call self.find_matching_mid_nodes checks whether the mid node of the longest
        edge is present in the bcs_mesh template. If so, the green element is a neighbor of a red element. If not, it
        is the neighbor of a blue element.

        @param nodes_where_longest:
        @param ele:
        @return:green_ele
        """
        # mid_node = self.get_mid_nodes(nodes_where_longest, None, shape=None)

        mid_node = [entry["Mid_node"] for entry in self.green_mid_nodes.values()]
        nodes_where_longest = [entry["Longest_edge"] for entry in self.green_mid_nodes.values()]

        nodes = self.nodes_array(self.for_green_ref)
        nodes_neighbor = self.nodes_array(self.green_marked_neighbor)
        keep_node, _, _, _, nodes_longest_edge = AMR.keep_rotation_direction(
            nodes_neighbor, nodes, nodes_where_longest
        )

        for count, (nle, mn) in enumerate(zip(nodes_longest_edge, mid_node)):
            self.green_ele.append(np.array((mn, keep_node[count], nle[0])))
            self.green_ele.append(np.array((mn, nle[1], keep_node[count])))


    def blue_pattern(
            self, ele_dict
    ):
        """
        This function creates the blue pattern for elements which have one or two neighbors.
        @param two_neighbor:
        @param nodes_along_second_neighbor:
        @param ele:
        @param mid_node_with_le:
        @return:
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
            marked_edge_connecting_node, vertex_node, free_node, index_differ
        ) = self.find_vertex_and_mid_node(
            nodes, longest_edge, not_longest_edge, ele_dict
        )

        for count, (cn, fn, vn, index_diff) in enumerate(
                zip(marked_edge_connecting_node, free_node, vertex_node, index_differ)
        ):
            if count == 45:
                pass

            if not index_diff == 1:
                self.blue_ele.append(
                    np.array(
                        (cn, mid_nodes[count][0], mid_nodes[count][1])
                    )
                )
                self.blue_ele.append(
                    np.array((mid_nodes[count][1], mid_nodes[count][0], vn))
                )
                self.blue_ele.append(
                    np.array((fn, mid_nodes[count][1], vn))
                )
            else:
                self.blue_ele.append(
                    np.array(
                        (cn, mid_nodes[count][1], mid_nodes[count][0])
                    )
                )
                self.blue_ele.append(
                    np.array((mid_nodes[count][1], vn, mid_nodes[count][0]))
                )
                self.blue_ele.append(
                    np.array((fn, vn, mid_nodes[count][1]))
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
        ele_dict, longest_edge_dict = self.get_ele_dictionary(hanging_edges, all_edges, nodes_where_longest)
        marked_dict = self.elements_to_refine(ele_dict, hanging_edges, longest_edge_dict)
        self.get_marked_neighbor(marked_dict, ele_dict)
        mid_node_dict = self.get_new_mid_nodes(ele_dict, marked_dict, longest_edge_dict)
        self.assign_mid_nodes(mid_node_dict, ele_dict, marked_dict, longest_edge_dict)

        return ele_dict, marked_dict, longest_edge_dict

    def create_all_pattern(self, ele_dict):
        """
        This function concatenates all pattern creations

        @param mid_node_coors:
        @param nodes_where_longest:
        @return:
        """

        self.red_pattern()
        self.green_pattern()
        self.blue_pattern(ele_dict)

    def main_amr(self):
        """
        Main function
        @return:
        """

        self.run_marking()
        nodes_where_longest, all_edges, marked_edges = self.get_longest_edge()
        ele_dict, marked_dict, longest_edge_dict = self.find_elements_to_refine(
            marked_edges, all_edges, nodes_where_longest
        )
        self.create_all_pattern(ele_dict)
