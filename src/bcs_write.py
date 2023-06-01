import numpy as np
import os
from refine_ele import AMR


class write_file(AMR):
    """
    Class for writing .bcs-files
    """

    def __init__(self, path, out_path, thickness):
        super().__init__(path, out_path, thickness)
        self.file_name = None

    def run_amr(self):
        """
        Run the whole AMR loop

        @return:
        """
        super().main_amr()

    def manipulate_ele(self):
        """
        Update the old ndarray with the unrefined elements. Also keep thickness and temperature of the marked and
        refined elements the same. Delete the elements from the mesh which are marked
        """

        blue_elements = self.ele_undeformed[self.for_blue_ref, 3::]
        green_elements = self.ele_undeformed[self.for_green_ref, 3::]
        red_elements = self.ele_undeformed[self.for_red_ref, 3::]

        thickness_temp_red = np.repeat(red_elements, 4, axis=0)
        thickness_temp_green = np.repeat(green_elements, 2, axis=0)
        thickness_temp_blue = np.repeat(blue_elements, 3, axis=0)

        complete_red_cluster = np.hstack((self.red_ele, thickness_temp_red))
        complete_green_cluster = np.hstack((self.green_ele, thickness_temp_green))
        complete_blue_cluster = np.hstack((self.blue_ele, thickness_temp_blue))

        self.ele_undeformed = np.delete(
            self.ele_undeformed,
            self.for_red_ref + self.for_green_ref + self.for_blue_ref,
            axis=0,
        )

        self.ele_undeformed = np.insert(
            self.ele_undeformed,
            0,
            np.concatenate(
                ([complete_red_cluster, complete_green_cluster, complete_blue_cluster]),
                axis=0,
            ),
            axis=0,
        )

    def append_mesh(self):
        """
        Append the new mid node coordinates to the mesh.
        @return:
        """

        complete_mesh_cluster = np.hstack(
            (self.bcs_mesh, np.zeros((len(self.bcs_mesh), 2), dtype=np.int))
        )

        self.mesh_undeformed = np.append(
            self.mesh_undeformed, complete_mesh_cluster, axis=0
        )

    def write_bcs(self):
        """
        Write the new file.
        @return:
        """
        self.file_name = "test_mesh_full_dict.bcs"
        file_length_ele = len(self.ele_undeformed)
        file_length_mesh = len(self.mesh_undeformed)
        file_length_bc = len(self.bc)
        filtering = [
            "B-SIM - DATA OF THE SHEET\n",
            self.symmetry_assignement,
            "-111 1 1 1 1 1 1 END NOP\n",
            "200.0  Char. dist\n",
            "-111 1 1 1 1 1 1 END OF COORS\n",
            "-111 1 1 1 1 1 1 END OF BCs\n",
            self.plane_coordinates,
        ]
        with open(os.path.join(self.out_path, self.file_name), "w") as bcs_amf:
            bcs_amf.write(filtering[0])
            bcs_amf.write(filtering[1])

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
            if file_length_bc:
                for bc in range(file_length_bc):
                    bcs_amf.write(
                        "{:5d}{:6d}\n".format(
                            self.bc[bc, 0].astype(np.int), self.bc[bc, 1].astype(np.int)
                        )
                    )
            bcs_amf.write(filtering[5])

            # if self.plane_coordinates:
            #    bcs_amf.write(filtering[6])

    def check_success(self):
        """
        Checks if the file exisits.
        @return:
        """
        if os.path.exists(os.path.join(self.out_path, self.file_name)):
            print("Success!")
        else:
            raise RuntimeError("Did not complete")

    def run_main(self):
        """
        Run the whole project

        """

        self.run_amr()
        # print("Writing new .bcs-file...")
        self.manipulate_ele()
        self.append_mesh()
        self.write_bcs()
        self.check_success()
