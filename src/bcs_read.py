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
import re
import pandas as pd


class bcs_data:
    """
    The following source code is represented in an OOP-format for an easier way of handling the instances.
    First of all a path variable gets assigned which includes the absolute path to the .bcs-files. These
    were created by B-SIM and include informations about the nodes, element thickness and temperature
    of the pre
    """

    def __init__(self):
        self.get_latest_deformed = []
        self.get_latest_undeformed = []
        self.ele_undeformed = []
        self.ele_deformed = []
        self.mesh_undeformed = []
        self.mesh_deformed = []
        self.__check_out_path = []

    @property
    def filepath(self):
        return self.__filepath

    @filepath.setter
    def filepath(self, path_str):
        """
        Check if the path Variable is a string. If not print out an error message. Otherwise assign the path
        string to the classes instance bcs_file_name"

        :param path:
        :return:
        """
        if isinstance(path_str, str):
            self.__filepath = path_str
        else:
            raise TypeError("Path not defined as a string")



    def get_path_undeformed(self):
        """"
        Here all undeformed .bcs files will be append to the List "get_undeformed_files". The Glob module finds all
        pathnames to files that are inside a given absolute path to a folder. Afterwards it's necessary to determine
        all undeformed .bcs-files from the folder with an if-statement. Only the latest folders entry will be picked
        because of th os.path.getctime function. Therefore the user is able to reduce possible errors caused by selecting
        the wrong input data for the adaptive mesh refinement. If the .bcs-files list is empty an error message appears.
        """
        get_undeformed_files = glob.glob(self.filepath)
        list_bcs_undeformed = []
        for index in get_undeformed_files:
            if ("deformed") not in index and index.endswith(".bcs"):
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
        :return:
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
        :return:
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

            lines_to_keep = np.arange(start+1, end)

            ele_data_container[i] = pd.read_csv(self.path_lib[i],
                                   delim_whitespace=True,
                                   engine="python",
                                   skiprows=lambda x: x not in lines_to_keep,
                                   names=["Node_1","Node_20","Node_3","Thickness", "Temperature"],
                                   usecols=[1,2,3,4,5]
                                   ).to_numpy()

        self.ele_deformed, self.ele_undeformed= ele_data_container[0], ele_data_container[1]


    def read_mesh(self):
        """
        This function returns the same dataframe as the function before but the deformed .bcs-file is used.
        :return:
        """
        mesh_data_container = [np.array([]), np.array([])]

        for i in range(len(mesh_data_container)):


            start = self.line.index("200.0  Char. dist\n")
            end = self.line.index("-111 1 1 1 1 1 1 END OF COORS\n")

            lines_to_keep = np.arange(start + 1, end)

            mesh_data_container[i] = pd.read_csv(self.path_lib[i],
                                   delim_whitespace=True,
                                   engine="python",
                                   skiprows=lambda x: x not in lines_to_keep,
                                   names = [ "Mesh_x", "Mesh_y", "Mesh_z","None_1","None_2"],
                                   usecols=[1, 2, 3, 4, 5]
                                   ).to_numpy()

            self.mesh_deformed, self.mesh_undeformed = mesh_data_container[0], mesh_data_container[1]


    def read_bc(self):

        self.bc = np.array([])
        bc_index = self.line.index("-111 1 1 1 1 1 1 END OF BCs\n")
        bc_begin_index = self.line.index("-111 1 1 1 1 1 1 END OF COORS\n")

        self.bc = pd.read_csv(self.path_lib[0],
                                      sep="\s+",
                                      engine="python",
                                      skiprows=bc_begin_index+1,
                                      nrows=bc_index - bc_begin_index-2
                                      ).to_numpy()




    def run_reading(self):
        self.get_path_undeformed()
        self.get_path_deformed()
        self.read_ele()
        self.read_mesh()
        self.read_bc()



