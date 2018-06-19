#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 12 23:37:18 2018

@author: Ricardo Rivas
"""
import numpy as np

class ShurComplement():
    """
    This class implements the Schur Complement Algorithm for solving 
    the matrix associated to the domain decomposition method
    """
    
    def __init__(self, G, f):
        """
        Constructor
        -----------
        
        G : 2D ndarray
            Matrix associated to the domain decomposition method
        f : ndarray
            RHS narray
        """
        self.__G = G
        self.__f = f
        
    def __del__(self):
        del(self.__G)
        
    def solveSystem(self,  data):
        """
        Block-Gauss Elimination Algorithm
        
        data : tuple 
            A tuple containing the data length for each subdomain (n1, n2, len_in1, len_in2, len_in3, len_bn1, len_bn2, len_bn3)
        """
        (n1, n2, len_in1, len_in2, len_in3, len_bn1, len_bn2, len_bn3) = data
        ninter = n1 + n2

        # ---------------------------------------------------------------------- #
        #             Block-Gauss Elimination Algorithm
        # ---------------------------------------------------------------------- #
        
        B = self.__G[0:len_in1+len_bn1+len_in2+len_bn2+len_in3+len_bn3,
              0:len_in1+len_bn1+len_in2+len_bn2+len_in3+len_bn3]
        E = self.__G[0:len_in1+len_bn1+len_in2+len_bn2+len_in3+len_bn3,
              -ninter:]
        F = self.__G[-ninter:, 0:len_in1+len_bn1+len_in2+len_bn2+len_in3+len_bn3]
        C = self.__G[-ninter:,-ninter:]
        fi = self.__f[0:len_in1+len_bn1+len_in2+len_bn2+len_in3+len_bn3]
        gi = self.__f[-ninter:]
    
        B1 = B[0:len_in1+len_bn1,0:len_in1+len_bn1]
        E1 = E[0:len_in1+len_bn1,-ninter:-ninter+n1]
        f1 = fi[0:len_in1+len_bn1]
        Ep1 = np.linalg.solve(B1,E1)
        fp1 = np.linalg.solve(B1,f1)
        
        B2 = B[len_in1+len_bn1:len_in1+len_bn1+len_in2+len_bn2,
               len_in1+len_bn1:len_in1+len_bn1+len_in2+len_bn2]
        E2 = E[len_in1+len_bn1:len_in1+len_bn1+len_in2+len_bn2,
               -ninter:]
        f2 = fi[len_in1+len_bn1:len_in1+len_bn1+len_in2+len_bn2]
        Ep2 = np.linalg.solve(B2,E2)
        fp2 = np.linalg.solve(B2,f2)
        
        B3 = B[len_in1+len_bn1+len_in2+len_bn2:,
               len_in1+len_bn1+len_in2+len_bn2:]
        E3 = E[len_in1+len_bn1+len_in2+len_bn2:,
               -ninter+n1:]
        f3 = fi[len_in1+len_bn1+len_in2+len_bn2:]
        Ep3 = np.linalg.solve(B3,E3)
        fp3 = np.linalg.solve(B3,f3)
        
        Ep = np.zeros((B.shape[0],E.shape[1]))
        fp = np.zeros((B.shape[0],len(fi)))
        
        Ep[0:len_in1+len_bn1, :-n1] = Ep1
        Ep[len_in1+len_bn1:len_in1+len_bn1+len_in2+len_bn2, :] = Ep2
        Ep[len_in1+len_bn1+len_in2+len_bn2:,-n1:] = Ep3
        
        fp = np.concatenate((fp1, fp2, fp3))
        
        #Ep = np.linalg.solve(B,E)          # Direct Shur Solution
        #fp = np.linalg.solve(B,fi)
        gp = gi-F.dot(fp)
        S = C-F.dot(Ep)
        ys = np.linalg.solve(S,gp)
        xs = fp-Ep@ys
        
        lam = np.concatenate((xs,ys))
        
        return lam
