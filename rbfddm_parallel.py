#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 25 16:34:52 2018

@author: Ricardo Rivas

This program show the solution of the Laplace equation on a C-Shape domain
using Domain Decomposition with RBF and parallelized algorithm with MPI.

"""
import rbf
import matplotlib.pyplot as plt
import numpy as np
import time
import GraphsRBF as grbf
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

if rank == 0:
    
    # ------------------------------------------------------------------------------ #
    #        Generate three subdomains 
    # ------------------------------------------------------------------------------ #
    L = 30                                  # Number of nodes in horizontal dimension
    H = 30                                  # Number of nodes in vertical dimension, total (L*H - L*H/6)
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
    # ---------------------------------------------------------------------- #
    #       Solve system - Parallelized Block-Gauss Elimination Algorithm
    # ---------------------------------------------------------------------- #
    t1 = time.clock()
    
    B = G[0:len_in1+len_bn1+len_in2+len_bn2+len_in3+len_bn3,
          0:len_in1+len_bn1+len_in2+len_bn2+len_in3+len_bn3]
    E = G[0:len_in1+len_bn1+len_in2+len_bn2+len_in3+len_bn3,
          -ninter:]
    F = G[-ninter:, 0:len_in1+len_bn1+len_in2+len_bn2+len_in3+len_bn3]
    C = G[-ninter:,-ninter:]
    fi = f[0:len_in1+len_bn1+len_in2+len_bn2+len_in3+len_bn3]
    gi = f[-ninter:]

    B1 = B[0:len_in1+len_bn1,0:len_in1+len_bn1]
    E1 = E[0:len_in1+len_bn1,-ninter:-ninter+n1]
    f1 = fi[0:len_in1+len_bn1]
    
    B2 = B[len_in1+len_bn1:len_in1+len_bn1+len_in2+len_bn2,
           len_in1+len_bn1:len_in1+len_bn1+len_in2+len_bn2]
    E2 = E[len_in1+len_bn1:len_in1+len_bn1+len_in2+len_bn2,
           -ninter:]
    f2 = fi[len_in1+len_bn1:len_in1+len_bn1+len_in2+len_bn2]

    B3 = B[len_in1+len_bn1+len_in2+len_bn2:,
           len_in1+len_bn1+len_in2+len_bn2:]
    E3 = E[len_in1+len_bn1+len_in2+len_bn2:,
           -ninter+n1:]
    f3 = fi[len_in1+len_bn1+len_in2+len_bn2:]
    
    # --------------------------------------------- #
    #  Arrange and send data to different threads
    # --------------------------------------------- #
    
    sizeM2 = np.array([B2.shape[0], E2.shape[1]])
    
    B2 = np.reshape(B2, B2.shape[0]*B2.shape[1])
    E2 = np.reshape(E2, E2.shape[0]*E2.shape[1])
    
    data2 = np.concatenate((B2, E2, f2))
    comm.Send(sizeM2, dest=1, tag=10)
    comm.Send(data2, dest=1, tag=12)

    sizeM3 = np.array([B3.shape[0], E3.shape[1]])
    B3 = np.reshape(B3, B3.shape[0]*B3.shape[1])
    E3 = np.reshape(E3, E3.shape[0]*E3.shape[1])
    
    data3 = np.concatenate((B3, E3, f3))
    comm.Send(sizeM3, dest=2, tag=11)
    comm.Send(data3, dest=2, tag=13)    
    
    # --------------------------------------------- #
    # Solve Partial System in thread 0
    # --------------------------------------------- #
    Ep1 = np.linalg.solve(B1,E1)
    fp1 = np.linalg.solve(B1,f1)
    
if rank == 1:
    # --------------------------------------------- #
    #  Receive Data from thread 0
    # --------------------------------------------- #
    sizeM2 = np.empty(2, dtype=int)
    comm.Recv(sizeM2, source=0, tag=10)
    
    data2 = np.empty(sizeM2[0]*sizeM2[0]+sizeM2[0]*sizeM2[1]+sizeM2[0])
    comm.Recv(data2, source=0, tag=12)
    B2 = data2[0:sizeM2[0]*sizeM2[0]]
    E2 = data2[sizeM2[0]*sizeM2[0]:sizeM2[0]*sizeM2[0]+sizeM2[0]*sizeM2[1]]
    f2 = data2[sizeM2[0]*sizeM2[0]+sizeM2[0]*sizeM2[1]:]
    
    # --------------------------------------------- #
    #     Solve Partial System in thread 1
    # --------------------------------------------- #
    Ep2 = np.linalg.solve(np.reshape(B2,(sizeM2[0],sizeM2[0])), np.reshape(E2,(sizeM2[0],sizeM2[1])))
    fp2 = np.linalg.solve(np.reshape(B2,(sizeM2[0],sizeM2[0])), f2)
    
    # --------------------------------------------- #
    #     Send partial solution from thread 1 to 0
    # --------------------------------------------- #
    Ep2 = np.reshape(Ep2,(sizeM2[0]*sizeM2[1]))
    
    data2 = np.concatenate((Ep2, fp2))
    comm.Send(data2, dest=0, tag=19)
        
if rank == 2:
    # --------------------------------------------- #
    #      Receive Data from thread 0
    # --------------------------------------------- #
    sizeM3 = np.empty(2, dtype=int)
    comm.Recv(sizeM3, source=0, tag=11)
    
    data3 = np.empty(sizeM3[0]*sizeM3[0]+sizeM3[0]*sizeM3[1]+sizeM3[0])
    comm.Recv(data3, source=0, tag=13)
    B3 = data3[0:sizeM3[0]*sizeM3[0]]
    E3 = data3[sizeM3[0]*sizeM3[0]:sizeM3[0]*sizeM3[0]+sizeM3[0]*sizeM3[1]]
    f3 = data3[sizeM3[0]*sizeM3[0]+sizeM3[0]*sizeM3[1]:]
    
    # --------------------------------------------- #
    #      Solve Partial System in thread 2
    # --------------------------------------------- #
    Ep3 = np.linalg.solve(np.reshape(B3,(sizeM3[0],sizeM3[0])), np.reshape(E3,(sizeM3[0],sizeM3[1])))
    fp3 = np.linalg.solve(np.reshape(B3,(sizeM3[0],sizeM3[0])), f3)
    
    # --------------------------------------------- #
    #     Send partial solution from thread 2 to 0
    # --------------------------------------------- #
    Ep3 = np.reshape(Ep3,(sizeM3[0]*sizeM3[1]))
    
    data3 = np.concatenate((Ep3, fp3))
    comm.Send(data3, dest=0, tag=21)

if rank == 0:
    # --------------------------------------------- #
    #  Receive patial solution from thread 1 and 2
    # --------------------------------------------- #
    data2 = np.empty(sizeM2[0]*sizeM2[1]+sizeM2[0])
    comm.Recv(data2, source=1, tag=19)
    
    data3 = np.empty(sizeM3[0]*sizeM3[1]+sizeM3[0])
    comm.Recv(data3, source=2, tag=21)
    
    Ep2 = data2[0:sizeM2[0]*sizeM2[1]]
    fp2 = data2[sizeM2[0]*sizeM2[1]:]
    Ep2.shape = (sizeM2[0], sizeM2[1])
    
    Ep3 = data3[0:sizeM3[0]*sizeM3[1]]
    fp3 = data3[sizeM3[0]*sizeM3[1]:]
    Ep3.shape = (sizeM3[0], sizeM3[1])
    
    # --------------------------------------------- #
    #       Join Partial solution
    # --------------------------------------------- #
    Ep = np.zeros((B.shape[0],E.shape[1]))
    fp = np.zeros((B.shape[0],len(fi)))

    Ep[0:len_in1+len_bn1, :-n1] = Ep1
    Ep[len_in1+len_bn1:len_in1+len_bn1+len_in2+len_bn2, :] = Ep2
    Ep[len_in1+len_bn1+len_in2+len_bn2:,-n1:] = Ep3
    
    fp = np.concatenate((fp1, fp2, fp3))
    
    gp = gi-F.dot(fp)
    S = C-F.dot(Ep)
    ys = np.linalg.solve(S,gp)
    xs = fp-Ep@ys
    
    lam = np.concatenate((xs,ys))
    
    t2 = time.clock()
    te = t2 - t1
    print(" Elapsed time to solve Ax = b with parallelization of Shur Complement algorithm: %g" % te)
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