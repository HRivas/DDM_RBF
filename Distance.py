#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 21 14:42:28 2018

@author: Ricardo Rivas
"""
import numpy as np

class ComputeDistance():
    """
    This class computes the distance between points of a given domain
    """
    @staticmethod
    def __distEuclidian(p1, p2):
        """
        Compute euclidian distance between two points p1, p2
        
        Parameters
        -----------
        
        p1 : Tuple
            Point 1 of a tuple of size 2 (x, y)
            
        p2 : Tuple
            Point 2 of a Tuple of size 2 (x, y)
        """
        d = np.sqrt((p1[0]-p2[0])**2+(p1[1]-p2[1])**2)
        return d
    
    @staticmethod
    def distanceMatrix(x):
        """
        Compute r distance matrix
        
        Parameters
        -----------
        
        x : 
            a list of tuples with position (x,y)
        """
        N = len(x)
        r = np.zeros((N,N))
        for i in range(0,N):
            for j in range(0,N):
                r[i][j] = ComputeDistance.__distEuclidian(x[i], x[j])
        return r
    

if __name__ == '__main__':
    from Domain import DomainC
    import matplotlib.pyplot as plt
    
    L = 12
    H = 9
    ob = DomainC(L, H)
    x = np.concatenate(ob.generateDomain())
    
    r = ComputeDistance.distanceMatrix(x)
    plt.imshow(r)