# Note:
* The code requires CUPY to run. 
* It forces the vorticity equation with constant power input specified from the desired kolmogorov length scale.
* Viscosity is applied implicitly while RK4 is applied on the non-linear terms and the forcing.
* The viscosity can be changed to hyperviscosity using the lp ( which is the power of the laplacian).
* Other methods of forcing can also be implemented.

Create a new anaconda environment using the .yml file. See : https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html
