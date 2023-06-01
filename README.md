# bsim_amr
Adaptive mesh refinement algorithm based on the RGB-refinement technique

This algorithm provides a refinement strategy for BSIM blow molding simulation. The refinement is based on a refinement strategy proposed in 
-W. Dörfler, “A Convergent Adaptive Algorithm for Poisson’s Equation”, SIAM
Journal on Numerical Analysis, Jg. 33, Nr. 3, S. 1106–1124, 1996.


The following refinement strategies are used:

![Alt text](img/ref_strategy.jpg?raw=true "Refinement strategy")

**Current status**

The refinement works with an acceptable speed. The blow molding simlulation works with the refined mesh.

![Alt text](img/refined_bottle_cap.jpg?raw=true =250x250 "Refined bottle head")



![Alt text](img/refined_bottle_body.jpg?raw=true =250x250 "Refined bottle head")

**To do's**

Find an appropiate way to handle the boundary elements in the clamping area. Sometimes hanging nodes appear
at the boundary. There are two way's: 

- Avoid these are by defining an "offset" where elements should'nt be marked
- Include those new hanging edges in 