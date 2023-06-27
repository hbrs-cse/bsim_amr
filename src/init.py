from bcs_write import write_file

#Please define a path to a folder where a preform .bcs-file and a blown up .bcs-file are both located.
#Make sure that there are only two files in the folder.
#it is also important that the blown up models file name endswith "deformed.bcs"!

path = r"D:\OneDrive - Hochschule Bonn-Rhein-Sieg\Master\MP1\Project\examples\Vierkant\test_files\40k\*"
out_path = (
    r"D:\OneDrive - Hochschule Bonn-Rhein-Sieg\Master\MP1\Project\examples\Vierkant\test_files\40k"
)
thickness_lower_threshold = 50
thickness_upper_threshold = 80
angular_deviation_threshold = 20
filename_out = "Test_mesh_40k.bcs"

if __name__ == "__main__":
    write_bcs = write_file(path, out_path, thickness_lower_threshold, thickness_upper_threshold,
                           angular_deviation_threshold, filename_out
                           )
    write_bcs.run_main()
