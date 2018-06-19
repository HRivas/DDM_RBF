#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 13 00:31:36 2018

@author: Ricardo Rivas
"""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def graphsRBF(data, u_sol):
    """
    Show different graphs of the domain and solution
    
    Parameters
    ----------
    data : tuple
        a tuple with the data nodes of each subdomain (inodes1, bnodes1, inodes2, bnodes2, inodes3, bnodes3, internodesS, internodesI)
    
    u_sol : tuple
        a tuple with (G matrix, 2D-narray solution, 1D-narray solution, number of nodes in x-direction, number of nodes in y-direction)
    """
    inodes1, bnodes1, inodes2, bnodes2, inodes3, bnodes3, internodesS, internodesI = data
    G, U, u, L, H = u_sol
    
    # ------------------------------------------------------------------------
    # Rotate nodes to show C-Shapa domain
    # ------------------------------------------------------------------------
    theta = np.radians(-90)
    c, s = np.cos(theta), np.sin(theta)
    R = np.array(((c, -s), (s, c)))
    inodes1 = (R@inodes1.T).T
    bnodes1 = (R@bnodes1.T).T
    internodesS = (R@internodesS.T).T
    inodes2 = (R@inodes2.T).T
    bnodes2 = (R@bnodes2.T).T
    inodes3 = (R@inodes3.T).T
    bnodes3 = (R@bnodes3.T).T
    internodesI = (R@internodesI.T).T
    
    fig = plt.figure()
    siz = 8
    
    # First subdomain
    plt.scatter(inodes1[:,0]/L,inodes1[:,1]*3/H+3, s=siz)
    plt.scatter(bnodes1[:,0]/L,bnodes1[:,1]*3/H+3, s=siz)
    plt.scatter(internodesS[:,0]/L,internodesS[:,1]*3/H+3, s=siz)
    
    # Second subdomain
    plt.scatter(inodes2[:,0]/L,inodes2[:,1]*3/H+3, s=siz)
    plt.scatter(bnodes2[:,0]/L,bnodes2[:,1]*3/H+3, s=siz)
    
    # Third subdomain
    plt.scatter(inodes3[:,0]/L,inodes3[:,1]*3/H+3, s=siz)
    plt.scatter(bnodes3[:,0]/L,bnodes3[:,1]*3/H+3, s=siz)
    plt.scatter(internodesI[:,0]/L,internodesI[:,1]*3/H+3, s=siz)
    plt.xlabel('x [m]')
    plt.ylabel('y [m]')
    plt.title('Cloud Points Domain')
    plt.show(block=False)
    
    # ------------------------------------------------------------------------
    #  Show DDM matrix
    # ------------------------------------------------------------------------
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.spy(G)
    plt.xlabel('Node')
    plt.ylabel('Node')
    plt.title('Matrix associated with RBF') 
    plt.show(block=False)
    
    # ------------------------------------------------------------------------
    #  Show 2D solution
    # ------------------------------------------------------------------------
    plt.figure()
    plt.imshow(U, extent = [0,1,0,3])
    plt.title('Stady-State Heat Equation  ' r'$ \nabla^2 T = 0 $')
    plt.xlabel('x [m]')
    plt.ylabel('y [m]')
    plt.show(block=False)
    
    # ------------------------------------------------------------------------
    #  Show 3D represenattion
    # ------------------------------------------------------------------------
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    x = np.linspace(0,1, L)
    y = np.linspace(0,3, H)
    X,Y = np.meshgrid(x,y)
    ax.plot_surface(X, Y, U, rstride=1, cstride=1, cmap='viridis', edgecolor='none')
    plt.xlabel('x [m]')
    plt.ylabel('y [m]')
    ax.set_zlabel('T [°C]')
    ax.set_title('Stady-State Heat Equation  ' r'$ \nabla^2 T = 0 $')
    plt.show(block=False)
    
    # ------------------------------------------------------------------------
    #  Show 3D represenattion
    # ------------------------------------------------------------------------    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    x1 = np.concatenate((inodes1,bnodes1,inodes2,bnodes2,inodes3,bnodes3,internodesS,internodesI))
    x = x1[:,0]/(H-1)
    y = x1[:,1]*3/(L-1)+3
    ax.scatter(x, y, u, c=u, cmap='viridis', linewidth=0.5)
    plt.xlabel('x [m]')
    plt.ylabel('y [m]')
    ax.set_zlabel('T [°C]')
    ax.set_title('Stady-State Heat Equation  ' r'$ \nabla^2 T = 0 $')
    plt.show()