from bcs_write import write_file

import cProfile
import io
import pstats
import subprocess

path = r"C:/Users/Fabik/OneDrive - Hochschule Bonn-Rhein-Sieg/Master/MP1/Project/examples/Vierkant/*"
out_path = (
    r"D:\OneDrive - Hochschule Bonn-Rhein-Sieg\Master\MP1\Project\out\Vierkant"
)
thickness = 50

if __name__ == "__main__":
    write_bcs = write_file(path, out_path, thickness)
    write_bcs.run_main()

