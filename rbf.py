#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 11 21:59:13 2018

@author: Ricardo Rivas

Generate the array of nodes for the three subdomains
"""
import numpy as np
from Domain import DomainC
from Distance import ComputeDistance as di
from Rbfn import KernelRBF as rbk
from ShurComplement import ShurComplement

def genDomain(L, H, inte = False):
    """
    Generate the array of nodes for the three subdomains
    
    Parameters
    ----------
    
    L: int
        number of nodes of the horizontal longer side
    
    H : int
        number of nodes of the vertical side
        
    inte : Boolean (True), optional
        Return the value of integer nodes
        
    Returns
    -------
    res : ndarray, ndarray, ndarray
        Three numpy arrays for each subdomain.
    """
    dom = DomainC(L, H)
    do1, do2, do3 = dom.generateDomain()
    d1 = do1.astype(int)
    d2 = do2.astype(int)
    d3 = do3.astype(int)
    
    if inte == False:
        d1 = do1.astype(float)
        d2 = do2.astype(float)
        d3 = do3.astype(float)
        
        d1[:,1] = d1[:,1]/(L-1)
        d2[:,1] = d2[:,1]/(L-1)
        d3[:,1] = d3[:,1]/(L-1)
    
        d1[:,0] = d1[:,0]*3/(H-1)
        d2[:,0] = d2[:,0]*3/(H-1)
        d3[:,0] = d3[:,0]*3/(H-1)
    
    return d1, d2, d3

def pointsSeparate(l, h, d, dom):
    """
    Separate interior, boundary and interface nodes for each subdomain
    
    Parameters
    ----------
    
    l : int
        Length of the given subdomain (in number of nodes)
        
    h : int
        Height of the given subdomain (in number of nodes)
        
    d : int
        Array of nodes for the given subdomain
        
    dom : string
        'sup'|'inf'|'mid' depending on the subdomain
    
    Returns    
    --------
    
    res : ndarray, ndarray, ndarray
    """
    inodes = []
    bnodes = []
    internodes = []
    k = 0
    for i in range(0,h):
        for j in range(0,l):
            if i == 0 or i == (h-1):
                if i == (h-1) and j < round(l/2) and dom == 'sup':
                    internodes.append(d[k])
                elif i == 0 and j < round(l/2) and dom == 'inf':
                    internodes.append(d[k])
                elif dom == 'mid' and (j > 0 and j < l-1):
                    inodes.append(d[k])
                else:
                    bnodes.append(d[k])
            elif j == 0 or j == (l-1):
                bnodes.append(d[k])
            else:
                inodes.append(d[k])
            k += 1
    return np.array(inodes), np.array(bnodes), np.array(internodes)

def ddmMat(data, c):
    """
    Build the matrix associated to the domain decomposition method
    
    Parameters
    -----------
    
    data : 
        A tuple containing (inodes1, bnodes1, inodes2, bnodes2, inodes3, bnodes3, internodesS, internodesI, r1, r2, r3, Ntotal)
    c : 
        Shape Parameter
    """
    
    (n1, n2, len_in1, len_in2, len_in3, len_bn1, len_bn2, len_bn3, r1, r2, r3, Ntotal) = data
    ninter = n1 + n2
    
    G = np.zeros((Ntotal, Ntotal))
    
    G[0:len_in1, 0:len_in1] = rbk.kernel2(r1[0:len_in1, 0:len_in1],c)
    G[0:len_in1, len_in1:len_in1+len_bn1] = rbk.kernel2(r1[0:len_in1, len_in1:len_in1+len_bn1],c)
    G[len_in1:len_in1+len_bn1, 0:len_in1+len_bn1] = rbk.kernel(r1[len_in1:len_in1+len_bn1, 0:len_in1+len_bn1],c)
    G[0:len_in1, -ninter:-ninter+n1] = rbk.kernel2(r1[0:len_in1,-n1:],c)
    G[len_in1:len_in1+len_bn1, -ninter:-ninter+n1] = rbk.kernel(r1[len_in1:-n1,-n1:],c)
    G[-ninter:-ninter+n1, 0:len_in1+len_bn1] = rbk.kernel(r1[-n1:,:-n1],c)
    G[-ninter+1:-ninter+n1-1, 0:len_in1+len_bn1] = rbk.kernel2(r1[-n1+1:-1,:-n1],c)
    G[-ninter:-ninter+n1, -ninter:-ninter+n1] = rbk.kernel(r1[-n1:,-n1:],c)
    G[-ninter+1:-ninter+n1-1, -ninter:-ninter+n1] = rbk.kernel2(r1[-n1+1:-1,-n1:],c)
    
    G[len_in1+len_bn1:len_in1+len_bn1+len_in2, len_in1+len_bn1:len_in1+len_bn1+len_in2+len_bn2] = rbk.kernel2(r2[0:len_in2, 0:len_in2+len_bn2],c)
    G[len_in1+len_bn1+len_in2:len_in1+len_bn1+len_in2 + len_bn2, len_in1+len_bn1:len_in1+len_bn1+len_in2 + len_bn2] = rbk.kernel(r2[len_in2:len_in2 + len_bn2, 0:len_in2 + len_bn2],c)
    G[len_in1+len_bn1:len_in1+len_bn1+len_in2, -ninter:] = rbk.kernel2(r2[0:len_in2,-ninter:],c)
    G[len_in1+len_bn1+len_in2:len_in1+len_bn1+len_in2 + len_bn2, -ninter:] = rbk.kernel(r2[len_in2:len_in2 + len_bn2,-ninter:],c)
    G[-ninter:, len_in1+len_bn1:len_in1+len_bn1+len_in2 + len_bn2] = rbk.kernel(r2[-ninter:, 0:len_in2 + len_bn2],c)
    G[-ninter+1:-ninter+n1-1, len_in1+len_bn1:len_in1+len_bn1+len_in2 + len_bn2] = rbk.kernel2(r2[-ninter+1:-ninter+n1-1, 0:len_in2 + len_bn2],c)
    G[-ninter+n1+1:-ninter+n1+n2-1, len_in1+len_bn1:len_in1+len_bn1+len_in2 + len_bn2] = rbk.kernel2(r2[-ninter+n1+1:-ninter+n1+n2-1, 0:len_in2 + len_bn2],c)
    G[-ninter+n1:, -ninter:] = rbk.kernel(r2[-ninter+n1:,-ninter:],c)
    G[-ninter+n1+1:-1, -ninter:] = rbk.kernel2(r2[-ninter+n1+1:-1,-ninter:],c)
    G[-ninter:-ninter+n1,-ninter+n1:] = rbk.kernel(r2[-ninter:-ninter+n1,-ninter+n1:],c)
    G[-ninter+1:-ninter+n1-1,-ninter+n1:] = rbk.kernel2(r2[-ninter+1:-ninter+n1-1,-ninter+n1:],c)
    
    rs0 = len_in1+len_bn1+len_in2 + len_bn2
    rs1 = len_in1+len_bn1+len_in2 + len_bn2
    G[len_in1+len_bn1+len_in2 + len_bn2:len_in1+len_bn1+len_in2 + len_bn2+len_in3, len_in1+len_bn1+len_in2 + len_bn2:len_in1+len_bn1+len_in2 + len_bn2+len_in3 + len_bn3] = rbk.kernel2(r3[0:len_in3 ,0:len_in3 + len_bn3],c)
    G[len_in1+len_bn1+len_in2 + len_bn2+len_in3:rs0+len_in3 + len_bn3, rs1:rs1+len_in3 + len_bn3] = rbk.kernel(r3[len_in3:len_in3 + len_bn3,0:len_in3 + len_bn3],c)
    G[len_in1+len_bn1+len_in2 + len_bn2:len_in1+len_bn1+len_in2 + len_bn2+len_in3,-n2:] = rbk.kernel2(r3[0:len_in3,-n2:],c)
    G[len_in1+len_bn1+len_in2 + len_bn2+len_in3:len_in1+len_bn1+len_in2 + len_bn2+len_in3 + len_bn3,-n2:] = rbk.kernel(r3[len_in3:len_in3 + len_bn3,-n2:],c)
    G[-n2:,len_in1+len_bn1+len_in2 + len_bn2:len_in1+len_bn1+len_in2 + len_bn2+len_in3 + len_bn3] = rbk.kernel(r3[-n2:,:len_in3 + len_bn3],c)
    G[-n2+1:-1,len_in1+len_bn1+len_in2 + len_bn2:len_in1+len_bn1+len_in2 + len_bn2+len_in3 + len_bn3] = rbk.kernel2(r3[-n2+1:-1,:len_in3 + len_bn3],c)
    
    return G
    
if __name__ == '__main__':
    x = np.concatenate(genDomain(30,30))
    print(di.distanceMatrix(x))