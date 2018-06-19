#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 21 12:29:28 2018

@author: Ricardo Rivas
"""
import numpy as np
import matplotlib.pyplot as plt

class DomainC():
    """
    This class generates a 2D C-shape domain
    
          ---|-------- length -----|
           . * * * * * * * * * * * *
           . * * * * * * * * * * * *
           . * * * * * * * * * * * *
           . * * * * * *
    height . * * * * * *
           . * * * * * *
           . * * * * * * * * * * * *
           . * * * * * * * * * * * *
          ---* * * * * * * * * * * *
    
    """
    
    def __init__(self, length = 3, height = 3):
        """
        Constructor 
        
        Parameters
        -----------
        
        length : int
            Number of nodes of the horizontal longer side
            
        height : int
            Number of nodes of the vertical side
        """
        self.__length = length
        self.__heigth = height

    def __del__(self):
        """
        Destructor
        """
        del(self.__length)
        del(self.__heigth)
        
    def generateDomain(self):
        """
        Generate three subdomains of the C-Shape domain and
        return an arrays with the list of points for each subdomain
        """
        L = self.__length
        H = self.__heigth
        x1 = []
        for i in range(0, round(H/3)):
            for j in range(0,L):
                x1.append([i,j])
        
        x2 = []
        for i in range(int(H/3), round(2*H/3)):
            for j in range(0,round(L/2)):
                x2.append([i,j])
            
        x3 = []
        for i in range(round(2*H/3), H):
            for j in range(0,L):
                x3.append([i,j])
        return np.array(x1), np.array(x2), np.array(x3)
                
if __name__ == "__main__":    
    L = 30
    H = 60
    ob = DomainC(L, H)
    dom1, dom2, dom3 = ob.generateDomain()
    
    theta = np.radians(-90)
    c, s = np.cos(theta), np.sin(theta)
    R = np.array(((c, -s), (s, c)))
    
    x = np.concatenate((dom1, dom2, dom3), axis=0)
    x = (R@x.T).T
    plt.scatter(x[:,0], x[:,1]+H)