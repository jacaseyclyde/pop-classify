# -*- coding: utf-8 -*-
"""
pop-classify main function

Authors:
    J. Andrew Casey-Clyde (@jacaseyclyde)
    Alex Colebaugh
    Kevin Prasad
"""

import numpy as np

import matplotlib.pyplot as plt

def CornerPlot(data,labels):
    # convert string class labels to color labels (for use w/ scatter)
    # for now this is just what's in the sample data. this could be automated
    # but I want to keep some control over the colors themselves.
        
    cdict = {'O':'blue','B':'lightskyblue','A':'white','F':'lightyellow',
             'G':'yellow','K':'orange','M':'red','T':'brown','L':'saddlebrown',
             'C':'black','W':'purple'}
    
    colClass = []
    for c in subClass:
        colClass.append(cdict[c[0]])
        
    colClass = np.array(colClass)
    
    # plot the classes/colors
    nAx = len(data)
    
    fig1, ax1 = plt.subplots(nAx - 1, nAx - 1, sharex=True, sharey=True)
    fig1.set_size_inches(12,12)
    
    ax1[0,0].set_xticklabels([])
    ax1[0,0].set_yticklabels([])
    
    for i in range(nAx - 1):
        for j in range(nAx - 1):
            if j > i:
                ax1[i, j].axis('off')
                
            else:
                ax1[i, j].scatter(colordata[j], colordata[nAx - 1 - i], c=colClass, s=50)
                
            if j == 0:
                ax1[i, j].set_ylabel(labels[nAx - 1 - i])
            
            if i == nAx - 2:
                ax1[i, j].set_xlabel(labels[j])
    
    
    fig1.subplots_adjust(hspace=0, wspace=0)
    fig1.show()
    fig1.savefig('./out/color_corner.pdf')
    fig1.savefig('./out/color_corner.png')


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
    
    # TODO: Optimize for Carbon star classes/white dwarfs/brown dwarfs
    stellar_class = []
    for c in subClass:
        stellar_class.append(c[0])
   
    axLabels = ['$u-g$', '$g-r$', '$r-i$', '$i-z$', '$u-r$', '$u-i$', '$u-z$',
                '$g-i$', '$g-z$', '$r-z$']
    
    #CornerPlot(colordata,axLabels)
    
    #SVMclassifier(colordata,subClass)
