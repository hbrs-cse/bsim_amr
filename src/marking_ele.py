"""
Fabian Kind
Hochschule Bonn-Rhein-Sieg
Institut für Elektrotechnik, Maschinenbau und Technikjournalismus
Masterprojekt 1
Adaptive Mesh Refinement
"""
from bcs_read import bcs_read
import numpy as np


class marking_ele(bcs_read):
    """
    Class for marking of elements
    """

    def __init__(self, path, out_path, thickness_lower_threshold, thickness_upper_threshold,
                 angular_deviation_threshold, filename_out
                 ):
        super().__init__(path, out_path, thickness_lower_threshold, thickness_upper_threshold,
                         angular_deviation_threshold, filename_out
                         )
        self.ele_dict = {}

    def get_ele(self):
        """
        Run the reading of the .bcs-Data

        @return:
        """
        super().run_reading()

    def thickness_diff_calc(self):
        """
        Caclulation of the thickness difference of deformed and undeformed element.
        @return: diff_calc
        """
        y, x = self.ele_deformed[:, 3], self.ele_undeformed[:, 3]

        diff_calc = np.asarray(abs((y - x) / y) * 100)

        return diff_calc

    def thickness_diff(self, thickness_diff):
        """
        Marks all elements whose thickness difference is in a sepcific range.
        """
        arg_list = np.where(
            (thickness_diff > self.thickness_lower_threshold) & (thickness_diff < self.thickness_upper_threshold)
            )[0]
        for val in arg_list:
            self.ele_list.append(val)
            self.marked_ele.append(val)

    def get_nodes_array(self, ele):
        """
        Returns an array of nodes which are marked.

        @return: nodes_array
        """

        nodes = self.ele_undeformed[:, 0:3]
        nodes = nodes[ele].astype(np.int)
        nodes_array = np.asarray(nodes).reshape(-1, 3)

        return nodes_array

    @staticmethod
    def edges(nodes_array):
        """
        Create a ndarray with three edge tuple.

        @param nodes_array:
        @return: edges
        """

        edges = [nodes_array[:, [0, 1]], nodes_array[:, [1, 2]], nodes_array[:, [2, 0]]]
        return edges

    def stacked_edges_array(self, ele):
        """
        Stack the edges array.

        @param ele:
        @return:
        """
        index = np.repeat(ele, 3)
        all_edges = self.get_nodes_array(ele)
        all_edges = marking_ele.edges(all_edges)

        edges = []
        for i in range(len(all_edges[0])):
            edges.append(all_edges[0][i])
            edges.append(all_edges[1][i])
            edges.append(all_edges[2][i])

        edges = np.c_[index, edges]
        return edges

    def get_back_edges(self):
        """
        Calls the below functions.

        @return:
        """
        all_ele = np.arange(0, len(self.ele_undeformed))
        all_edges = self.stacked_edges_array(all_ele)

        return all_edges

    def ele_dictionary(self, all_edges):
        """
        Creating a dictionary which stores the element edge end corresponding Element numbers. Set
        marked edges to true, otherwise false. A second dictionary stores the longest edge of each elements edge.

        @param all_edges:
        @return:
        """

        ele_dict = {}
        ele_num = 0
        for i, edge in enumerate(all_edges[:, 1:]):
            element_val = tuple(edge)

            if element_val not in ele_dict:
                self.ele_dict[element_val] = {
                    "Ele_num": ele_num,
                    "Marked": False,
                }

            if (i + 1) % 3 == 0:
                ele_num += 1

    def calc_normal_vector(self):
        """
        Calculates the normal vector per element
        @return:
        """

        mesh_coor = self.mesh_deformed[:, 0:3]
        normal_vec_dict = {}

        for _, val in self.ele_dict.items():
            ele_num = val["Ele_num"]
            edge = np.subtract(self.nodes_array(ele_num)[0], 1)

            direction_vector_container = [np.array([]), np.array([])]

            direction_vector_container[0] = np.subtract(
                mesh_coor[edge[1]], mesh_coor[edge[0]]
            )

            direction_vector_container[1] = np.subtract(
                mesh_coor[edge[0]], mesh_coor[edge[2]]
            )

            normal_vector = np.cross(
                direction_vector_container[0], direction_vector_container[1]
            )

            vector_length = np.linalg.norm(normal_vector)

            if ele_num not in normal_vec_dict:
                normal_vec_dict[ele_num] = {
                    "Normal_vector": normal_vector,
                    "Vector_length": vector_length,
                }

        return normal_vec_dict

    def calc_angular_deviation(self, normal_vec_dict):
        """
        Calculate the angular deviation between the neighboring elements normal vectors.

        @param normal_vec_dict:
        @return:
        """
        for edge, val in self.ele_dict.items():
            neighbor_ele = tuple(reversed(edge))
            ele_num = val["Ele_num"]
            if neighbor_ele in self.ele_dict:
                ele_num_neighbor = self.ele_dict[neighbor_ele]["Ele_num"]
                normal_vec = normal_vec_dict[ele_num]["Normal_vector"]
                vector_length = normal_vec_dict[ele_num]["Vector_length"]

                if ele_num in normal_vec_dict:
                    neighbor_normal_vec = normal_vec_dict[ele_num_neighbor][
                        "Normal_vector"
                    ]
                    neighbor_vector_length = normal_vec_dict[ele_num_neighbor][
                        "Vector_length"
                    ]
                    scalar_prod = np.dot(normal_vec, neighbor_normal_vec)
                    vector_length = np.multiply(vector_length, neighbor_vector_length)

                    if not scalar_prod > 0:
                        neighbor_normal_vec = np.multiply(neighbor_normal_vec, -1)
                        scalar_prod = np.dot(normal_vec, neighbor_normal_vec)

                    cosinus_of_angle = np.divide(scalar_prod, vector_length).round(decimals=3)

                    angular_deviation = np.degrees(
                        np.arccos(
                            cosinus_of_angle
                        )
                    )

                    if (
                            angular_deviation > self.angular_deviation_threshold
                            and ele_num not in self.marked_ele
                            and ele_num_neighbor not in self.marked_ele
                    ):
                        pass
                        self.marked_ele.append(ele_num)
                        self.marked_ele.append(ele_num_neighbor)

    def run_marking(self):
        """
        Run the main function for the marking strategy.

        """
        self.get_ele()
        thickness_diff = self.thickness_diff_calc()
        self.thickness_diff(thickness_diff)
        edges = self.get_back_edges()
        self.ele_dictionary(edges)
        normal_vector_dict = self.calc_normal_vector()
        self.calc_angular_deviation(normal_vector_dict)
