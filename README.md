# bsim_amr
Adaptive mesh refinement algorithm based on the RGB-refinement technique.

This algorithm provides a refinement strategy for BSIM blow molding simulation. The refinement is based on a refinement strategy proposed in 
-W. Dörfler, “A Convergent Adaptive Algorithm for Poisson’s Equation”, SIAM
Journal on Numerical Analysis, Jg. 33, Nr. 3, S. 1106–1124, 1996.

To use the mesh refinement, you have to define your properties in the init.py file, which ist located
in the src-folder. It's mandatory to define the following variables:

- **path**: Defines the location of the .bcs-files for the preform and the blown up model. Make sure that both files are in the same folder
and no other .bcs-files are located there. Makre sure the blown up models file name ends with "deformed.bcs".
- **out_path**: Defines the location of the output path for the .bcs-file that includes the refined mesh. If the defined path
does not have a folder called "out" it get's created automatically.
- **thickness_lower_threshold**: Defines the lower threshold for the marking strtagey. Every element get's marked whose
thickness difference between preform and blown up state exceeds the threshold.
- **thickness_upper_threshold**: Defines the upper threshold for the marking strategy and limits the number of marked
elements. Sometimes elements in the clamping area do have high distortions but these elements are not relevant for the 
adaptive mesh refinement.
- **angular_deviation_threshold**: Defines the threshold for the angular deviation between neighboring elements. If
the value exceeds the threshold, both elements are marked.

-**file_name**: The filename of the output file. Make sure it's a string and it ends with the file extension ".bcs".

---


The following refinements for shell elements are used:

![Alt text](img/ref_strategy.jpg?raw=true "Refinement strategy")

**Current status**

The refinement works with an acceptable speed. The blow molding simlulation works with the refined mesh.

![Alt text](img/refined_bottle_cap.jpg?raw=true)



![Alt text](img/refined_bottle_body.jpg?raw=true)

**To do's**

Find an appropiate way to handle the boundary elements in the clamping area. Sometimes hanging nodes appear
at the boundary. There are two way's: 

- Avoid these are by defining an "offset" where elements should'nt be marked
- Include those new hanging edges in 