import numpy as np
import os


class write_file:
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

        blue_elements_one = self.ele_undeformed[
                            self.for_blue_ref_one_neighbor, 3::
                            ]
        blue_elements_two = self.ele_undeformed[
                            self.for_blue_ref_two_neighbor, 3::
                            ]
        green_elements = self.ele_undeformed[
                         self.for_green_ref, 3::
                         ]
        red_elements = self.ele_undeformed[
                       self.for_red_ref, 3::
                       ]

        thickness_temp_red = np.repeat(
            red_elements, 4, axis=0
        )
        thickness_temp_green = np.repeat(
            green_elements, 2, axis=0
        )
        thickness_temp_blue_one = np.repeat(
            blue_elements_one, 3, axis=0
        )
        thickness_temp_blue_two = np.repeat(
            blue_elements_two, 3, axis=0
        )
        thickness_temp_blue = thickness_temp_blue_one.tolist() + \
                              thickness_temp_blue_two.tolist()

        self.green_ele, self.red_ele = np.asarray(
            self.green_ele), np.asarray(self.red_ele)

        complete_red_cluster = np.hstack(
            (self.red_ele, thickness_temp_red)
        )
        complete_green_cluster = np.hstack(
            (self.green_ele, thickness_temp_green)
        )
        complete_blue_cluster = np.hstack(
            (self.blue_ele, thickness_temp_blue)
        )

        self.ele_undeformed = np.delete(self.ele_undeformed,
                                        [self.for_red_ref +
                                        self.for_blue_ref_one_neighbor +
                                        self.for_green_ref +
                                        self.for_blue_ref_two_neighbor
                                         ],
                                        axis=0)

        self.ele_undeformed = np.append(
            self.ele_undeformed,
            np.concatenate(
                (complete_red_cluster,
                 complete_green_cluster,
                 complete_blue_cluster),
                axis=0),
            axis=0)

    def append_mesh(self):
        """
        Append the new mid node coordinates to the mesh.
        @return:
        """

        complete_mesh_cluster = np.hstack(
            (self.bcs_mesh[:, 1::], np.zeros((len(self.bcs_mesh), 2), dtype=np.int)))
        self.mesh_undeformed = np.append(
            self.mesh_undeformed, complete_mesh_cluster, axis=0)

    def write_file(self):
        """
        Write the new file.
        @return:
        """
        self.file_name = "Undeformed_refined_mesh6.bcs"
        file_length_ele = len(self.ele_undeformed)
        file_length_mesh = len(self.mesh_undeformed)
        file_length_bc = len(self.bc)
        filtering = [
            "B-SIM - DATA OF THE SHEET\n",
            "FULL\n",
            "-111 1 1 1 1 1 1 END NOP\n",
            "200.0  Char. dist\n",
            "-111 1 1 1 1 1 1 END OF COORS\n",
            "-111 1 1 1 1 1 1 END OF BCs\n",
        ]
        with open(os.path.join(self._out_path, self.file_name), "w") as bcs_amf:
            bcs_amf.write(filtering[0])
            bcs_amf.write(filtering[1])

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

            bcs_amf.write(filtering[2])
            bcs_amf.write(filtering[3])

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

            bcs_amf.write(filtering[4])
            for bc in range(file_length_bc):
                bcs_amf.write("{:5d}{:6d}\n".format(
                    self.bc[bc, 0].astype(np.int), self.bc[bc, 1].astype(np.int)))
            bcs_amf.write(filtering[4])

    def check_success(self):
        """
        Checks if the file exisits.
        @return:
        """
        if os.path.exists(os.path.join(self.__check_out_path, self.file_name)):
            print("Success!!")
        else:
            raise RuntimeError("Did not complete")
