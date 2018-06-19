#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 23 21:10:48 2018

@author: Ricardo Rivas
"""
import numpy as np
import matplotlib.pyplot as plt
from Distance import ComputeDistance

class KernelRBF():
    """
    Contains the interpolation functions for RBF method
    """
    
    @staticmethod
    def __mq(r, c):
        """
        Multiquadric Interpolation Function
        
        Parameters
        -----------
        
        r : 2D ndarray
            Distance Matrix
            
        c : float
            Shape Parameter
        """
        return np.sqrt(r ** 2 + c ** 2)
    
    @staticmethod
    def __d1x(r, c):
        """
        First Derivative of Multiquadric Function
        
        Parameters
        -----------
        
        r : 2D ndarray
            Distance Matrix
            
        c : float
            Shape Parameter
        """
        return r / np.sqrt( r ** 2 + c ** 2 )
    
    @staticmethod
    def __d2x(r, c):
        """
        Second Derivative of Multiquadric Function
        
        Parameters
        -----------
        
        r : 2D ndarray
            Distance Matrix
            
        c : float
            Shape Parameter
        """
        return c ** 2 / ( np.sqrt(r ** 2 + c ** 2) * (r ** 2 + c ** 2) )
    
    @staticmethod
    def kernel(r, c):
        """
        Apply Kernel
        
        Parameters
        -----------
        
        r : 2D ndarray
            Distance Matrix
            
        c : float
            Shape Parameter
        """
        return KernelRBF.__mq(r,c)
    
    @staticmethod
    def kernel2(r, c):
        """
        Apply Second derivative of a given Kernel
        
        Parameters
        -----------
        
        r : 2D ndarray
            Distance Matrix
            
        c : float
            Shape Parameter
        """
        return KernelRBF.__d2x(r,c)
    
    @staticmethod
    def evaluate(lam, r, c):
        """
        Evaluate the interpolation functions with the lambda coefficients
        
        Parameters
        -----------
        
        lam : ndarray
            Lambda coefficients
        
        r : 2D ndarray
            Distance Matrix
            
        c : float
            Shape Parameter
        """
        W = KernelRBF.kernel(r, c)
        u_new = W.dot(lam)
        return u_new

if __name__ == '__main__':
    """
     -------------------------------------------------------
     Example of soluton of the Laplace Equation using RBF
     -------------------------------------------------------
    """
    # -------------------------------------------------------
    # Domain generation (square domain)
    # -------------------------------------------------------
    L = 30
    H = L
    x = []
    for i in range(0,H):
        for j in range(0,L):
            x.append([i,j])

    inod = []
    bnod = []
    k = 0
    for i in range(0,H):
        for j in range(0,L):
            if i == 0 or i == (H-1):
                bnod.append(x[k])
            elif j == 0 or j == (L-1):
                bnod.append(x[k])
            else:
                inod.append(x[k])
            k += 1
            
    xN = np.array(inod+bnod)
    x = xN/(L-1)
    plt.scatter(x[:,0],x[:,1], s=1)
    
    # --------------------------------
    #   Compute Distance Matrix
    # --------------------------------
    di = ComputeDistance()
    r = di.distanceMatrix(x)
    
    # --------------------------------
    #   Compute Gramm Matrix
    # --------------------------------
    N = len(x)              # Number of total nodes
    NI = len(inod)          # Number of interior nodes
    
    c = 1/np.sqrt(N)
    rbf = KernelRBF()
    
    # Compute  wL matrix
    rL = r[0:NI][:]
    wL = rbf.kernel2(rL, c)
    
    # Compute wB matrix
    rB = r[NI:][:]
    wB = rbf.kernel(rB, c)
    
    # Fill Gramm matrix
    G = np.zeros((N,N))
    G[0:NI][:] = wL
    G[NI:N][:] = wB
    
    # --------------------------------
    #   Boundary Conditions
    # --------------------------------
    B = 100                         # Dirichlet condition
    f = np.zeros(len(x))
    f[NI+1:NI+L-1] = B
    
    # --------------------------------
    #   Solve system
    # --------------------------------    
    lam = np.linalg.solve(G,f)
    
    u = rbf.evaluate(lam, r, c)         # Evaluate interpolation function
    
    # --------------------------------
    #   Show graph
    # --------------------------------
    U = np.zeros((L,H))
    for i in range(0,N):
        U[xN[i][0], xN[i][1]] = u[i]
    
    plt.close('all')
    plt.imshow(U)
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    X1 = np.arange(0, L)/(L-1)
    Y1 = np.arange(0, H)/(H-1)
    X, Y = np.meshgrid(X1, Y1)
    ax.plot_surface(X, Y, U, cmap='viridis', antialiased=False)
    
    # --------------------------------------------------------
    # Analytical solution
    # --------------------------------------------------------
    axd = 0
    bxd = 1
    ayd = 0
    byd = 1
    boundA = 100
    boundB = 0
    boundC = 0
    boundD = 0
    def solucionAnalitica(x,y,N):
        T=np.zeros((len(y),len(x)))
        L=bxd-axd
        H=byd-ayd
        pi=np.pi
        for i in range(0,len(x)):
            for j in range(0,len(y)):
                suma=0
                for n in range(1,N):
                    suma += (1-(-1)**n)*np.sinh((n*pi*(H-y[j]))/L)*np.sin(n*pi*x[i]/L)/(n*pi*np.sinh(n*pi*H/L))
                T[j,i]=boundA*2*suma
        return T
    
    x = X1
    y = Y1
    T = solucionAnalitica(x, y, 100)
    
    fig = plt.figure()
    plt.imshow(T-U)         # Show absolute error
    
    E=np.sqrt(np.sum(np.square(T[1:-1,1:-1]-U[1:-1,1:-1])))
    print('Error:', E)
