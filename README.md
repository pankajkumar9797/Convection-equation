# Energy-equation

The energy equation can be derived from the concept that the rate of change of energy
of a system is equal to sum of any energy entering or leaving the system and the energy
produced inside the system by the energy source. 

Code is based on deal.II, an open source C++ software library supporting finite element 
code and it is interfaced with OpenBlas, P4est and Trilinos. Adaptive mesh refinement and is used 
and it is based on error estimators which enables refinement and coarsening of cells whenever 
there is a large change in the solutions gradient. This makes the code very useful as it 
doesn not need to run wherever the solution is constant, increases the mesh resolution where 
it is necessary and thereby saving computational time. visualization of solution is done 
with the help of VisIt. VisIt is an open source software developed by Lawrence Livermore 
National Laboratory and it is used for visualization, animation and as a analysis tool.

Problem is a square cavity in which a heat energy from a source is advected along the direction of velocity.

## 1. Temperature plot at a time = 0.84 seconds
![alt tag](https://rawgit.com/pankajkumar9797/Convection-equation/master/doc/pics_and_videos/At_T_0_84.png)

In the code the global matrices and right hand sides are assembled by using a stable scheme, 
thereafter system is solved by using either Generalized minimal residual method( in
short GMRES) solver or CG solver depending upon whether matrix is symmetric positive
definite or not.


