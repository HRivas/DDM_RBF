# Domain Decomposition Method using Shur Complement and Radial Basis Function

This program solves the Laplace equation in a C-shape. The substructuring method of Shur Complement is used and Radial Basis Function to solve the PDE.

It is implemented in python 3. The main program is rbfddm.py

A parallelized version is found in rbfddm_parallel.py. To execute this code you should have installed MPI and run in a terminal:
```
mpiexec -np 3 python3 rbfddm_parallel
```
