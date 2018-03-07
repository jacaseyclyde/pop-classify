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
    
    colordata = np.array([u-g, g-r, r-i, i-z, u-r, u-i, u-z, g-i, g-z, r-z])
    
    # convert string class labels to color labels (for use w/ scatter)
    # for now this is just what's in the sample data. this could be automated
    # but I want to keep some control over the colors themselves.
    
    classes = np.unique(subClass)
    cdict = {}
    
    cdict = {'O':'blue','B':'lightskyblue','A':'white','F':'lightyellow',
             'G':'yellow','K':'orange','M':'red','T':'brown','L':'saddlebrown',
             'C':'black','W':'purple'}
    
    colClass = []
    for c in subClass:
        colClass.append(cdict[c[0]])
        
    colClass = np.array(colClass)
    
    # plot the classes/colors
    fig1, ax1 = plt.subplots(9, 9, sharex=True, sharey=True)
    fig1.set_size_inches(24,24)
    
    ax1[0, 0].scatter(colordata[0], colordata[1], c=colClass, s=50)
    ax1[0, 0].set_ylabel('$g-r$')
    ax1[0, 0].set_xticklabels([])
    ax1[0, 0].set_yticklabels([])
    
    ax1[0, 1].axis('off')
    ax1[0, 2].axis('off')
    ax1[0, 3].axis('off')
    ax1[0, 4].axis('off')
    ax1[0, 5].axis('off')
    ax1[0, 6].axis('off')
    ax1[0, 7].axis('off')
    ax1[0, 8].axis('off')
    
    ax1[1, 0].scatter(colordata[0], colordata[2], c=colClass, s=50)
    ax1[1, 0].set_ylabel('$r-i$')
    
    ax1[1, 1].scatter(colordata[1], colordata[2], c=colClass, s=50)
    
    ax1[1, 2].axis('off')
    ax1[1, 3].axis('off')
    ax1[1, 4].axis('off')
    ax1[1, 5].axis('off')
    ax1[1, 6].axis('off')
    ax1[1, 7].axis('off')
    ax1[1, 8].axis('off')
    
    ax1[2, 0].scatter(colordata[0], colordata[3], c=colClass, s=50)
    ax1[2, 0].set_ylabel('$i-z$')
    
    ax1[2, 1].scatter(colordata[1], colordata[3], c=colClass, s=50)
    
    ax1[2, 2].scatter(colordata[2], colordata[3], c=colClass, s=50)
    
    ax1[2, 3].axis('off')
    ax1[2, 4].axis('off')
    ax1[2, 5].axis('off')
    ax1[2, 6].axis('off')
    ax1[2, 7].axis('off')
    ax1[2, 8].axis('off')
    
    ax1[3, 0].scatter(colordata[0], colordata[4], c=colClass, s=50)
    ax1[3, 0].set_ylabel('$u-r$')
    
    ax1[3, 1].scatter(colordata[1], colordata[4], c=colClass, s=50)
    
    ax1[3, 2].scatter(colordata[2], colordata[4], c=colClass, s=50)
    
    ax1[3, 3].scatter(colordata[3], colordata[4], c=colClass, s=50)
    
    ax1[3, 4].axis('off')
    ax1[3, 5].axis('off')
    ax1[3, 6].axis('off')
    ax1[3, 7].axis('off')
    ax1[3, 8].axis('off')
    
    ax1[4, 0].scatter(colordata[0], colordata[5], c=colClass, s=50)
    ax1[4, 0].set_ylabel('$u-i$')
    
    ax1[4, 1].scatter(colordata[1], colordata[5], c=colClass, s=50)
    
    ax1[4, 2].scatter(colordata[2], colordata[5], c=colClass, s=50)
    
    ax1[4, 3].scatter(colordata[3], colordata[5], c=colClass, s=50)
    
    ax1[4, 4].scatter(colordata[4], colordata[5], c=colClass, s=50)
    
    ax1[4, 5].axis('off')
    ax1[4, 6].axis('off')
    ax1[4, 7].axis('off')
    ax1[4, 8].axis('off')
    
    ax1[5, 0].scatter(colordata[0], colordata[6], c=colClass, s=50)
    ax1[5, 0].set_ylabel('$u-z$')
    
    ax1[5, 1].scatter(colordata[1], colordata[6], c=colClass, s=50)
    
    ax1[5, 2].scatter(colordata[2], colordata[6], c=colClass, s=50)
    
    ax1[5, 3].scatter(colordata[3], colordata[6], c=colClass, s=50)
    
    ax1[5, 4].scatter(colordata[4], colordata[6], c=colClass, s=50)
    
    ax1[5, 5].scatter(colordata[5], colordata[6], c=colClass, s=50)
    
    ax1[5, 6].axis('off')
    ax1[5, 7].axis('off')
    ax1[5, 8].axis('off')
    
    ax1[6, 0].scatter(colordata[0], colordata[7], c=colClass, s=50)
    ax1[6, 0].set_ylabel('$g-i$')
    
    ax1[6, 1].scatter(colordata[1], colordata[7], c=colClass, s=50)
    
    ax1[6, 2].scatter(colordata[2], colordata[7], c=colClass, s=50)
    
    ax1[6, 3].scatter(colordata[3], colordata[7], c=colClass, s=50)
    
    ax1[6, 4].scatter(colordata[4], colordata[7], c=colClass, s=50)
    
    ax1[6, 5].scatter(colordata[5], colordata[7], c=colClass, s=50)
    
    ax1[6, 6].scatter(colordata[6], colordata[7], c=colClass, s=50)
    
    ax1[6, 7].axis('off')
    ax1[6, 8].axis('off')
    
    ax1[7, 0].scatter(colordata[0], colordata[8], c=colClass, s=50)
    ax1[7, 0].set_ylabel('$g-z$')
    
    ax1[7, 1].scatter(colordata[1], colordata[8], c=colClass, s=50)
    
    ax1[7, 2].scatter(colordata[2], colordata[8], c=colClass, s=50)
    
    ax1[7, 3].scatter(colordata[3], colordata[8], c=colClass, s=50)
    
    ax1[7, 4].scatter(colordata[4], colordata[8], c=colClass, s=50)
    
    ax1[7, 5].scatter(colordata[5], colordata[8], c=colClass, s=50)
    
    ax1[7, 6].scatter(colordata[6], colordata[8], c=colClass, s=50)
    
    ax1[7, 7].scatter(colordata[7], colordata[8], c=colClass, s=50)
    
    ax1[7, 8].axis('off')
    
    ax1[8, 0].scatter(colordata[0], colordata[9], c=colClass, s=50)
    ax1[8, 0].set_xlabel('$u-g$')
    ax1[8, 0].set_ylabel('$r-z$')
    
    ax1[8, 1].scatter(colordata[1], colordata[9], c=colClass, s=50)
    ax1[8, 1].set_xlabel('$g-r$')
    
    ax1[8, 2].scatter(colordata[2], colordata[9], c=colClass, s=50)
    ax1[8, 2].set_xlabel('$r-i$')
    
    ax1[8, 3].scatter(colordata[3], colordata[9], c=colClass, s=50)
    ax1[8, 3].set_xlabel('$i-z$')
    
    ax1[8, 4].scatter(colordata[4], colordata[9], c=colClass, s=50)
    ax1[8, 4].set_xlabel('$u-r$')
    
    ax1[8, 5].scatter(colordata[5], colordata[9], c=colClass, s=50)
    ax1[8, 5].set_xlabel('$u-i$')
    
    ax1[8, 6].scatter(colordata[6], colordata[9], c=colClass, s=50)
    ax1[8, 6].set_xlabel('$u-z$')
    
    ax1[8, 7].scatter(colordata[7], colordata[9], c=colClass, s=50)
    ax1[8, 7].set_xlabel('$g-i$')
    
    ax1[8, 8].scatter(colordata[8], colordata[9], c=colClass, s=50)
    ax1[8, 8].set_xlabel('$g-z$')
    
    
    fig1.subplots_adjust(hspace=0, wspace=0)
    fig1.show()
    fig1.savefig('./out/color_corner.pdf')
    fig1.savefig('./out/color_corner.png')
    
    SVMclassifier(colordata,subClass)
