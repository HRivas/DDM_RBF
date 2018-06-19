# Domain Decomposition Method using Shur Complement and Radial Basis Function

This program solves the Laplace equation in a C-shape domain. The substructuring method of Shur Complement and Radial Basis Function to solve the PDE. The domain was divided in two subdomains.

The main program is rbfddm.py, it is implemented in python 3. One can change the number of nodes by modifying the L and H variables, L shuld be a multiple of two and H a mulitple of three.

A parallelized version is found in rbfddm_parallel.py. To execute this code you should have installed MPI and run in a terminal:
```
mpiexec -np 3 python3 rbfddm_parallel.py
```
