import os
from bcs_read import bcs_data
from AMR import AMR
from recursive_AMR import recursive_AMR
from bcs_write import write_file
import cProfile
import io
import pstats

path = r"C:/Users/Fabik/OneDrive - Hochschule Bonn-Rhein-Sieg/Master/MP1/Project/examples/*"
out_path = r"C:/Users/Fabik/OneDrive - Hochschule Bonn-Rhein-Sieg/Master/MP1/Project/out"
thickness = 50


class BSimAmr:
    def __init__(self, path, out_path, thickness):
        self.Bcs = bcs_data()
        self.amr = AMR(thickness)
        self.recursive_amr = recursive_AMR(thickness)

        self._bcs_path_undef = None
        self._bcs_path_def = None

        self.ele_data_def = None
        self.ele_data_undef = None
        self.mesh_data_def = None
        self.mesh_data_undef = None

        self.filepath_in = path
        self.fileout_path = out_path
        self.thickness = thickness
        self.recursive_thickness = thickness

    @property
    def bcs_path_undef(self):
        return self.__bcs_path_undef

    @bcs_path_undef.setter
    def bcs_path_undef(self, path):
        self.Bcs.filepath = self.filepath_in
        self._bcs_path_undef = os.path.abspath(path)
        self.Bcs.path_und = os.path.abspath(path)

    @property
    def check_out_path(self):
        return self.__check_out_path

    @check_out_path.setter
    def check_out_path(self):
        self.Bcs.check_out_path = self.fileout_path

    @property
    def set_thickness(self):
        return self.__thickness

    @set_thickness.setter
    def set_thickness(self):
        self.amr.thickness_diff =thickness

    @property
    def set_thickness_recursive(self):
        return self.__recursive_thickness
    @set_thickness_recursive.setter
    def set_recursive_thickness(self):
        self.recursive_amr.thickness_diff = thickness

    @property
    def bcs_path_def(self):
        return self._bcs_path_def

    @bcs_path_def.setter
    def bcs_path_def(self, path):
        self._bcs_path_def = os.path.abspath(path)
        self.Bcs.path_def = os.path.abspath(path)

    def read_bcs(self):
        self.Bcs.run_reading()
        self.ele_data_def = self.Bcs.ele_undeformed
        self.ele_data_undef = self.Bcs.ele_deformed
        self.mesh_data_def = self.Bcs.mesh_undeformed
        self.mesh_data_undef = self.Bcs.mesh_deformed
        self.bc = self.Bcs.bc
        self.Bcs.filepath


    def run_amr(self):
        self.amr.ele_undeformed = self.ele_data_undef
        self.amr.ele_deformed = self.ele_data_def
        self.amr.mesh_undeformed = self.mesh_data_undef
        self.amr.mesh_deformed = self.mesh_data_def
        self.amr.bc = self.bc
        self.amr.set_thickness_diff
        self.amr.main_amr()
        self.write = write_file(self.amr, out_path)
        self.write.check_out_path
        self.write.check_path()
        self.write.manipulate_ele()
        self.write.append_mesh()
        self.write.write_file()
        self.write.check_success()

    def run_recursive_amr(self):
        self.recursive_amr.ele_undeformed = self.ele_data_undef
        self.recursive_amr.ele_deformed = self.ele_data_def
        self.recursive_amr.mesh_undeformed = self.mesh_data_undef
        self.recursive_amr.mesh_deformed = self.mesh_data_def
        self.recursive_amr.bc = self.bc
        self.recursive_amr.set_thickness_diff
        self.recursive_amr.main_recursive_amr()
        self.write = write_file(self.amr, out_path)
        self.write.check_out_path
        self.write.check_path()
        self.write.manipulate_ele()
        self.write.append_mesh()
        self.write.write_file()
        self.write.check_success()

    def main(self):
        self.bcs_path_undef = path
        self.bcs_path_def = path
        self.read_bcs()
        #self.run_amr()
        self.run_recursive_amr()


if __name__ == '__main__':
    pr = cProfile.Profile()
    pr.enable()
    bSimAmr = BSimAmr(path, out_path, thickness)
    bSimAmr.main()
    pr.disable()
    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats('cumtime')
    ps.print_stats()

    with open('C:/Users/Fabik/OneDrive - Hochschule Bonn-Rhein-Sieg/Master/bsim_amr/cProfile/perf_output.txt', 'w+') as f:
        f.write(s.getvalue())


