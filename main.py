# -*- coding: utf-8 -*-
"""
pop-classify main function

Authors:
    J. Andrew Casey-Clyde (@jacaseyclyde)
    Alex Colebaugh
    Kevin Prasad
"""

import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt


def SVMclassifier(data,classes):
    (m, n) = data.shape  # dimensionality and number of points, respectively
    
    # computationally cheaper way to compute Gram matrix
    gram = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1):
            gram[i, j] = np.dot(data[:, i], data[:, j])
            gram[j, i] = gram[i, j]
            
    #print(gram)

if __name__ == "__main__":
    # Import the data in 2 seperate stmts b/c genfromtxt doesnt like multityping
    u, g, r, i, z = np.genfromtxt('data.csv', delimiter=',', skip_header=2,
                                  usecols=(0,1,2,3,4)).T
    subClass = np.genfromtxt('data.csv', delimiter=',',skip_header=2,usecols=5,
                             dtype=str)
    
    colordata = np.array([u-g, g-r, r-i, i-z])
    
    # convert string class labels to numeric class labels (for use w/ scatter)
    # for now this is just what's in the sample data. this could be automated
    # but I want to keep some control over the colors themselves
    cdict = {'A0':0, 'F2':1, 'F5':2, 'F9':3, 'G0':4, 'G2':5, 'K1':6, 'K3':7,
             'K5':8, 'K7':9, 'M0':10, 'M1':11, 'M2':12, 'M2V':13, 'M3':14,
             'M6':15, 'M8':16, 'T2':17, 'Carbon_lines':18, 'WD':19}
    
    numClass = []
    for c in subClass:
        numClass.append(cdict[c])
        
    numClass = np.array(numClass)
    
    # plot the classes/colors
    plt.scatter(colordata[0], colordata[1], c=numClass, cmap=plt.cm.jet)
    plt.show()
    
    SVMclassifier(colordata,subClass)
