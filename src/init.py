from bsim_amr import BSimAmr
from bcs_read import bcs_read
from refine_ele import AMR

import cProfile
import io
import pstats


path = r"C:/Users/Fabik/OneDrive - Hochschule Bonn-Rhein-Sieg/Master/MP1/Project/examples/Vierkant/*"
out_path = r"C:/Users/Fabik/OneDrive - Hochschule Bonn-Rhein-Sieg/Master/MP1/Project/out/Vierkant/out"
thickness = 50

super_obj = BSimAmr(path, out_path, thickness)
bcs_read = bcs_read(path, out_path, thickness)
#refine_ele = AMR()

def get_pstats():
    """
    Write cProfile file and pstats for performance analysis.

    @return:
    """
    pr = cProfile.Profile()
    pr.enable()
    pr.disable()
    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats('cumtime')
    ps.print_stats()

    with open('C:/Users/Fabik/OneDrive - Hochschule Bonn-Rhein-Sieg/Master/bsim_amr/cProfile/perf_output.txt', 'w+'
              ) as f:
        f.write(s.getvalue())





