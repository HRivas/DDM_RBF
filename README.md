# Domain Decomposition Method using Shur Complement and Radial Basis Function

This program solves the Laplace's equation in a C-shape domain. The substructuring method of Shur Complement and Radial Basis Function to solve the PDE. The domain was divided in two subdomains.

The main program is rbfddm.py, it is implemented in python 3. One can change the number of nodes by modifying the `L` and `H` variables, `L` should be a multiple of two and `H` a mulitple of three.

A parallelized version is found in rbfddm_parallel.py. To execute this code you should have installed MPI and run in a terminal:
```
mpiexec -np 3 python3 rbfddm_parallel.py
```

Results.py file shows a graph of the average (10 executions for each number of nodes) elapsed time taken to solve the associated linear system by:
- Complete Matrix
- Shur Complement (Block-Gauss Elimination Algorithm)
- Parallelized Shur Complement (Block-Gauss Elimination Algorithm)
