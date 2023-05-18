import os

class BSimAmr:
    def __init__(self, path, out_path, thickness):

        self.filepath = path
        self.out_path = out_path
        self.thickness = thickness

    @property
    def filepath(self):
        """
        Filepath setter
        """
        return self.__filepath

    @filepath.setter
    def filepath(self, path):
        """
        Check if the path Variable is a string. If not print out an error message. Otherwise assign the path
        string to the classes instance bcs_file_name"

        @param path_str:
        @return:
        """
        if not isinstance(path, str):
            raise TypeError("Path not defined as a string")
        else:
            self.__filepath = path

    @property
    def out_path(self):
        return self.__out_path

    @out_path.setter
    def out_path(self, out_path):
        if not isinstance(out_path, str):
            raise TypeError("Path not defined as a string")
        elif not out_path.endswith("/out"):
            raise FileNotFoundError("Path not defined as a string")
        else:
            self.__out_path = out_path


    @property
    def thickness(self):
        return self.__thickness

    @thickness.setter
    def thickness(self, thickness):
        if not isinstance(thickness, int):
            raise ValueError("Wall thickness difference must be an integer")
        elif thickness <= 0:
            raise ValueError("Wall thickness difference is negativ")
        elif thickness >= 100:
            raise ValueError("Wall thickness difference is too big")
        else:
            self.__thickness = thickness

