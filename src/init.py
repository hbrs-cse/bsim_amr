from bcs_write import write_file

import cProfile
import io
import pstats

path = r"C:/Users/Fabik/OneDrive - Hochschule Bonn-Rhein-Sieg/Master/MP1/Project/examples/Vierkant/*"
out_path = r"C:/Users/Fabik/OneDrive - Hochschule Bonn-Rhein-Sieg/Master/MP1/Project/out/Vierkant"
thickness = 50


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


if __name__ == '__main__':
    write_bcs = write_file(path, out_path, thickness)
    write_bcs.run_main()
    #get_pstats()
