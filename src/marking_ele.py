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

    def __init__(self, path, out_path, thickness):
        super().__init__(path, out_path, thickness)

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

        diff_calc = np.asarray(
            np.abs(
                (y - x) / y
            ) * 100
        )

        return diff_calc

    def thickness_diff(self, thickness_diff):
        """
        Marks all elements whose thickness difference is in a sepcific range.
        """
        arg_list = np.where((thickness_diff > self.thickness) & (thickness_diff < 60))

        ele_list = [arg_list[0].tolist()]
        for sublist in ele_list:
            for val in sublist:
                self.ele_list.append(val)
                self.marked_ele.append(val)

    def run_marking(self):
        """
        Run the main function for the marking strategy.

        """
        self.run_reading()
        thickness_diff = self.thickness_diff_calc()
        self.thickness_diff(thickness_diff)
