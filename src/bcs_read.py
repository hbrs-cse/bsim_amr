"""
Fabian Kind
Hochschule Bonn-Rhein-Sieg
Institut für Elektrotechnik, Maschinenbau und Technikjournalismus
Masterprojekt 1
Adaptive Mesh Refinement
"""

import os.path
import glob
import numpy as np
import pandas as pd
from bsim_amr import BSimAmr


class bcs_read(BSimAmr):
    def __init__(self, path, out_path, thickness):
        super().__init__(path, out_path, thickness)

        self.line = None
        self.path_lib = None
        self.path_undef = None
        self.path_def = None
        self.get_latest_deformed = []
        self.get_latest_undeformed = []
        self.ele_undeformed = None
        self.ele_deformed = None
        self.mesh_undeformed = None
        self.mesh_deformed = None
        self.bc = None
        self.coors = None

    def get_path_undeformed(self):
        """ "
        Here all undeformed .bcs files will be append to the List "get_undeformed_files". The Glob module finds all
        pathnames to files that are inside a given absolute path to a folder. Afterwards it's necessary to determine
        all undeformed .bcs-files from the folder with an if-statement. Only the latest folders entry will be picked
        because of th os.path.getctime function. Therefore the user is able to reduce possible errors caused by
        selecting the wrong input data for the adaptive mesh refinement. If the .bcs-files list is empty an error
        message appears.
        """
        get_undeformed_files = glob.glob(self.filepath)
        list_bcs_undeformed = []
        for index in get_undeformed_files:
            if "deformed" not in index and index.endswith(".bcs"):
                list_bcs_undeformed.append(index)
                self.get_latest_undeformed = max(
                    list_bcs_undeformed, key=os.path.getctime
                )
                lines = index.split("\\")
                self.path_undef = os.path.basename(lines[1])
            else:
                continue

        if list_bcs_undeformed:
            pass
            print("File '{0}' will be read...".format(self.path_undef))
        else:
            raise ValueError("No .bcs-files found")

    def get_path_deformed(self):
        """
        This function works the same way like the function before. It determines all the deformed .bcs-files in
        the given folders path.

        """

        get_path_deformed_files = glob.glob(self.filepath)
        list_bcs_deformed = []
        for index in get_path_deformed_files:
            if index.endswith("deformed.bcs"):
                list_bcs_deformed.append(index)
                self.get_latest_deformed = max(list_bcs_deformed, key=os.path.getctime)
                lines = index.split("\\")
                self.path_def = os.path.basename(lines[1])
            else:
                continue

        if list_bcs_deformed:
            pass
            print("File '{0}' will be read...\n".format(self.path_def))
        else:
            raise ValueError("No .bcs-files found")

    def read_ele(self):
        """
        After retrieving the raw files path of the undeformed .bcs-file the two backslashes seperating the filesname
        will be replaced with a slash. The function "open" opens the given filepath and writes all the lines from the
        files content in a list variable. Therefore it's possible to define exactly the start and end of the information
        that should be stored later in numpy ndarray. All the information stored in the file will be retrieved by
        the numpy function "genfromtxt". There are multiple parameters available to satisfy the users output conditions.
        """

        dir_path_und = self.get_latest_undeformed.replace("\\", "/")
        dir_path_def = self.get_latest_deformed.replace("\\", "/")

        self.path_lib = [dir_path_und, dir_path_def]
        self.line = []
        ele_data_container = [np.array([]), np.array([])]

        for i in range(len(ele_data_container)):
            with open(self.path_lib[i], "r") as bcs_file:
                for lines in bcs_file.readlines():
                    self.line.append(lines)

            start = self.line.index("FULL\n")
            end = self.line.index("-111 1 1 1 1 1 1 END NOP\n")

            lines_to_keep = np.arange(start + 1, end)

            ele_data_container[i] = pd.read_csv(
                self.path_lib[i],
                delim_whitespace=True,
                engine="python",
                skiprows=lambda x: x not in lines_to_keep,
                names=["Node_1", "Node_2", "Node_3", "Thickness", "Temperature"],
                usecols=[1, 2, 3, 4, 5],
            ).to_numpy()

        self.ele_undeformed, self.ele_deformed = (
            ele_data_container[0],
            ele_data_container[1],
        )

    def read_mesh(self):
        """
        This function returns the same dataframe as the function before but the deformed .bcs-file is used.
        """
        mesh_data_container = [np.array([]), np.array([])]

        for i in range(len(mesh_data_container)):
            start = self.line.index("200.0  Char. dist\n")
            end = self.line.index("-111 1 1 1 1 1 1 END OF COORS\n")

            lines_to_keep = np.arange(start + 1, end)

            mesh_data_container[i] = pd.read_csv(
                self.path_lib[i],
                delim_whitespace=True,
                engine="python",
                skiprows=lambda x: x not in lines_to_keep,
                names=["Mesh_x", "Mesh_y", "Mesh_z", "None_1", "None_2"],
                usecols=[1, 2, 3, 4, 5],
            ).to_numpy()

            self.mesh_undeformed, self.mesh_deformed = (
                mesh_data_container[0],
                mesh_data_container[1],
            )

    def read_bc(self):
        """
        Reads the .bcs-file
        """

        self.bc = np.array([])
        bc_index = self.line.index("-111 1 1 1 1 1 1 END OF BCs\n")
        bc_begin_index = self.line.index("-111 1 1 1 1 1 1 END OF COORS\n")

        if bc_index - bc_begin_index > 1:

            self.bc = pd.read_csv(
                self.path_lib[0],
                sep="\\s+",
                engine="python",
                skiprows=bc_begin_index + 1,
                nrows=bc_index - bc_begin_index - 2,
            ).to_numpy()

            self.coors = pd.read_csv(
                self.path_lib[0],
                sep="\\s+",
                engine="python",
                skiprows=bc_begin_index + (bc_index - bc_begin_index),
                nrows=bc_index + (bc_index - bc_begin_index)
            ).to_numpy()
        else:
            self.bc = []

    def run_reading(self):
        """
        Runs all functions
        """
        self.get_path_undeformed()
        self.get_path_deformed()
        self.read_ele()
        self.read_mesh()
        self.read_bc()
