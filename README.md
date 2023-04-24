# bsim_amr
Adaptive mesh refinement algorithm based on the RGB-refinement technique

This algorithm provides a refinement strategy for BSIM blow molding simulation. The refinement is based on a refinement strategy proposed in 
-W. Dörfler, “A Convergent Adaptive Algorithm for Poisson’s Equation”, SIAM
Journal on Numerical Analysis, Jg. 33, Nr. 3, S. 1106–1124, 1996.


The following refinement strategies are used:

![Alt text](img/ref_strategy.jpg?raw=true "Refinement strategy")

Note that the marking strategy itself is not based on the algorithm that is proposed by Willy Dörfler. At the moment, the marking of an element
is based on the thickness difference between undeformed and deformed shell element. Additional marking strategies should be used to get a better result.


**Current status**
At the moment, the elements seem to be marked correctly, and the refinement takes place. But there are
problems with the node assignment and rotation. 

Some elements work fine with the current state of the algorithm:
![Alt text](img/correct_marked.jpg?raw=true width="300" height="300" "Correct refined")
And some elements are broken because the wrong nodes are assigned:
![Alt text] (img/wrong_marked.jpg?raw=true width="300" height="300" "Wrong refined")