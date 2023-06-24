import os


class BSimAmr:
    def __init__(self, path, out_path, thickness_lower_threshold, thickness_upper_threshold,
                 angular_deviation_threshold, filename_out
                 ):
        self.filepath = path
        self.out_path = out_path
        self.thickness_lower_threshold = thickness_lower_threshold
        self.thickness_upper_threshold = thickness_upper_threshold
        self.angular_deviation_threshold = angular_deviation_threshold
        self.filename_out = filename_out

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

        @param path:
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
            try:
                if not os.path.exists(out_path + "/out"):
                    os.mkdir(out_path + "/" + "out")
                self.__out_path = out_path + "/" + "out"
            except PermissionError:
                print("Could not create an out folder")
                raise
        else:
            self.__out_path = out_path

    @property
    def thickness_lower_threshold(self):
        return self.__thickness_lower_threshold

    @thickness_lower_threshold.setter
    def thickness_lower_threshold(self, thickness_lower_threshold):
        if not isinstance(thickness_lower_threshold, int):
            raise TypeError("Wall thickness difference must be an integer")
        elif thickness_lower_threshold <= 0:
            raise ValueError("Wall thickness difference is negativ")
        elif thickness_lower_threshold >= 100:
            raise ValueError("Wall thickness difference is too big")
        else:
            self.__thickness_lower_threshold = thickness_lower_threshold

    @property
    def thickness_upper_threshold(self):
        return self.__thickness_upper_threshold

    @thickness_upper_threshold.setter
    def thickness_upper_threshold(self, thickness_upper_threshold):
        if not isinstance(thickness_upper_threshold, int):
            raise TypeError("Wall thickness difference must be an integer")
        elif thickness_upper_threshold == self.__thickness_lower_threshold:
            raise ValueError("Wall thickness difference is zero")
        elif thickness_upper_threshold < self.__thickness_lower_threshold:
            raise ValueError("Upper wall thickness threshold is lower than the lower wall thickness threshold")
        else:
            self.__thickness_upper_threshold = thickness_upper_threshold

    @property
    def angular_deviation_threshold(self):
        return self.__angular_deviation_threshold

    @angular_deviation_threshold.setter
    def angular_deviation_threshold(self, angular_deviation_threshold):
        if not isinstance(angular_deviation_threshold, int):
            raise TypeError("Angular deviation threshold must be an integer")
        elif angular_deviation_threshold <= 0:
            raise ValueError("Angular deviation threshold is negativ")
        else:
            self.__angular_deviation_threshold = angular_deviation_threshold

    @property
    def filename_out(self):
        return self.__filename_out

    @filename_out.setter
    def filename_out(self, filename_out):
        if not filename_out.endswith(".bcs"):
            raise TypeError("The output file name doesn't end with .bcs")
        if not isinstance(filename_out, str):
            raise ValueError("Filename is not a string")
        else:
            self.__filename_out = filename_out
