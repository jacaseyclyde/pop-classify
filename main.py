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
    cdict = {'A0':'white', 'F2':'lightyellow', 'F5':'lightyellow', 'F9':'lightyellow', 'G0':'yellow',
             'G2':'yellow', 'K1':'orange', 'K3':'orange', 'K5':'orange', 'K7':'orange',
             'M0':'red', 'M1':'red', 'M2':'red', 'M2V':'red', 'M3':'red',
             'M6':'red', 'M8':'red', 'T2':'brown', 'Carbon_lines':'blue',
             'WD':'purple'}
    
    numClass = []
    for c in subClass:
        numClass.append(cdict[c])
        
    numClass = np.array(numClass)
    
    # plot the classes/colors
    fig1, ax1 = plt.subplots(3, 3, sharex=True, sharey=True)
    fig1.set_size_inches(24,24)
    
    ax1[0, 0].scatter(colordata[0], colordata[1], c=numClass, cmap=cdict, s=50)
    ax1[0, 0].set_ylabel('$g-r$')
    ax1[0, 0].set_xticklabels([])
    ax1[0, 0].set_yticklabels([])
    
    ax1[0, 1].axis('off')
    ax1[0, 2].axis('off')
    
    ax1[1, 0].scatter(colordata[0], colordata[2], c=numClass, s=50)
    ax1[1, 0].set_ylabel('$r-i$')
    
    ax1[1, 1].scatter(colordata[1], colordata[2], c=numClass, s=50)
    
    ax1[1, 2].axis('off')
    
    ax1[2, 0].scatter(colordata[0], colordata[3], c=numClass, s=50)
    ax1[2, 0].set_xlabel('$u-g$')
    ax1[2, 0].set_ylabel('$i-z$')
    
    ax1[2, 1].scatter(colordata[1], colordata[3], c=numClass, s=50)
    ax1[2, 1].set_xlabel('$g-r$')
    
    ax1[2, 2].scatter(colordata[0], colordata[3], c=numClass, s=50)
    ax1[2, 2].set_xlabel('$r-i$')
    
    fig1.subplots_adjust(hspace=0, wspace=0)
    fig1.show()
    fig1.savefig('./out/color_corner.pdf')
    fig1.savefig('./out/color_corner.png')
    
    SVMclassifier(colordata,subClass)
