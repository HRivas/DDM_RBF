#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 25 16:34:52 2018

@author: Ricardo Rivas

This program show the solution of the Laplace equation on a C-Shape domain
using Domain Decomposition with RBF.

"""
import rbf
import matplotlib.pyplot as plt
import numpy as np
import time
import GraphsRBF as grbf

# ------------------------------------------------------------------------------ #
#        Generate three subdomains 
# ------------------------------------------------------------------------------ #
L = 30                                  # Number of nodes in horizontal dimension
H = 30                                  # Number of nodes in vertical dimension - L*H/6
d1, d2, d3 = rbf.genDomain(L, H)
Ntotal = len(d1)+len(d2)+len(d3)        # Total number of nodes

c = 1/np.sqrt(Ntotal)                   # Shape parameter

# ------------------------------------------------------------------------------ #
#                Separate Domain
# ------------------------------------------------------------------------------ #

# Separate domain points in interior, boundary and interface nodes
l = L
h = round(H/3)
inodes1, bnodes1, internodesS = rbf.pointsSeparate(l, h, d1, 'sup')

l = round(L/2)
h = round(H/3)
inodes2, bnodes2, internodes = rbf.pointsSeparate(l, h, d2, 'mid')

l = L
h = round(H/3)
inodes3, bnodes3, internodesI = rbf.pointsSeparate(l, h, d3, 'inf')

# ---------------------------------------------------------------------------- #
#        Compute distance matrices for each subdomain
# ---------------------------------------------------------------------------- #
x1 = np.concatenate((inodes1, bnodes1, internodesS))
r1 = rbf.di.distanceMatrix(x1)
x2 = np.concatenate((inodes2, bnodes2, internodesS, internodesI))
r2 = rbf.di.distanceMatrix(x2)
x3 = np.concatenate((inodes3, bnodes3, internodesI))
r3 = rbf.di.distanceMatrix(x3)
x4 = np.concatenate((inodes1, bnodes1, inodes2, bnodes2, internodesS, internodesI))
r4 = rbf.di.distanceMatrix(x4)
x5 = np.concatenate((inodes2, bnodes2, inodes3, bnodes3, internodesS, internodesI))
r5 = rbf.di.distanceMatrix(x5)

# ------------------------------------------------------------------------------ #
#        Build the DDM Matrix
# ------------------------------------------------------------------------------ #
n1 = len(internodesS)
n2 = len(internodesI)
len_in1 = len(inodes1)
len_in2 = len(inodes2)
len_in3 = len(inodes3)
len_bn1 = len(bnodes1)
len_bn2 = len(bnodes2)
len_bn3 = len(bnodes3)
ninter = n1 + n2
dataD = (n1, n2, len_in1, len_in2, len_in3, len_bn1, len_bn2, len_bn3, r1, r2, r3, Ntotal)
G = rbf.ddmMat(dataD, c)
# ------------------------------------------------------------------------------ #
#                Define Dirichlet Boundary Conditions
# ------------------------------------------------------------------------------ #

Bc = 100
f = np.zeros(Ntotal)
f[len(inodes1)] = Bc
for i in range(0,round(H/3)-2):
    f[len_in1+L+2*i] = Bc
    
for i in range(0,round(H/3)):
    f[len_in1+len_bn1+len_in2+2*i] = Bc
    
for i in range(0,round(H/3)-2):
    f[len_in1+len_bn1+len_in2+len_bn2+len_in3+n2+2*i] = Bc
    
f[-ninter] = Bc
f[-ninter+n1] = Bc

f[len_in1] = 0
# ------------------------------------------------------------------------------ #
#   Solve system
# ------------------------------------------------------------------------------ #

# Solve entire matrix
t1 = time.clock()
lam1 = np.linalg.solve(G,f)
t2 = time.clock()
te = t2 - t1
print(" Elapsed time to solve Ax = b complete system: %g" % te)

# Solve using Schur Complement Algorithm
t1 = time.clock()
sol = rbf.ShurComplement(G, f)
lam = sol.solveSystem(dataD[0:8])
t2 = time.clock()
te = t2 - t1
print(" Elapsed time to solve Ax = b with Shur Complement algorithm: %g" % te)
# ------------------------------------------------------------------------------ #
#   Interpolation
# ------------------------------------------------------------------------------ #
# Apply interpolation function
do1, do2, do3 = rbf.genDomain(L, H, True)
inodes1, bnodes1, internodesS = rbf.pointsSeparate(L,round(H/3),do1,'sup')
inodes2, bnodes2, internodes = rbf.pointsSeparate(round(L/2),round(H/3),do2,'mid')
inodes3, bnodes3, internodesI = rbf.pointsSeparate(L,round(H/3),do3,'inf')

U = np.zeros((H,L))

# Interpolate superior subdomain
u1 = rbf.rbk.evaluate(np.concatenate((lam[0:len_in1+len_bn1], lam[-ninter:-ninter+n1])), r1, c)
x1 = np.concatenate((inodes1, bnodes1))
for i in range(0,len(x1)):
    U[x1[i][0],x1[i][1]] = u1[i]

# Interpolate middle subdomain
u2 = rbf.rbk.evaluate(np.concatenate((lam[len_in1+len_bn1:len_in1+len_bn1+len_in2+len_bn2], lam[-ninter:])), r2, c)
x2 = np.concatenate((inodes2, bnodes2))
for i in range(0,len(x2)):
    U[x2[i][0],x2[i][1]] = u2[i]

# Interpolate inferior subdomain
u3 = rbf.rbk.evaluate(np.concatenate((lam[len_in1+len_bn1+len_in2+len_bn2:len_in1+len_bn1+len_in2+len_bn2+len_in3+len_bn3], lam[-ninter+n1:])), r3, c)
x3 = np.concatenate((inodes3, bnodes3))
for i in range(0,len(x3)):
    U[x3[i][0],x3[i][1]] = u3[i]
    
# Interpolate superior interface
u4 = rbf.rbk.evaluate(np.concatenate((lam[0:len_in1+len_bn1+len_in2+len_bn2], lam[-ninter:])), r4, c)
u4 = u4[-ninter:-ninter+n1]
x4 = internodesS
for i in range(0,len(x4)):
    U[x4[i][0],x4[i][1]] = u4[i]

# Interpolate inferior interface
u5 = rbf.rbk.evaluate(np.concatenate((lam[len_in1+len_bn1:len_in1+len_bn1+len_in2+len_bn2+len_in3+len_bn3], lam[-ninter:])), r5, c)
u5 = u5[-ninter+n1:]
x5 = internodesI
for i in range(0,len(x5)):
    U[x5[i][0],x5[i][1]] = u5[i]

u = np.concatenate((u1[:-ninter+n1], u2[:-ninter], u3[:-ninter+n1], u4, u5))
# ------------------------------------------------------------------------------ #
#   Show Graph Results
# ------------------------------------------------------------------------------ #
plt.close('all')
u_sol = (G, U, u, L, H)
data = (inodes1, bnodes1, inodes2, bnodes2, inodes3, bnodes3, internodesS, internodesI)
grbf.graphsRBF(data, u_sol)