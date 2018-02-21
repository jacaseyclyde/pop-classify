# -*- coding: utf-8 -*-
"""
pop-classify main function

Authors:
    J. Andrew Casey-Clyde (@jacaseyclyde)
    Alex Colebaugh
    Kevin Prasad
"""

import numpy as np


def SVMclassifier(data):
    (m, n) = data.shape  # dimensionality and number of points, respectively
    # computationally cheaper way to compute Gram matrix
    gram = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1):
            gram[i, j] = np.dot(data[:, i], data[:, j])
            gram[j, i] = gram[i, j]
            
    print(gram)
    print(np.dot(data.T, data))


if __name__ == "__main__":
    ra, dec, u, g, r, i, z, redshift = np.genfromtxt('data.csv', delimiter=',',
                                                     skip_header=2).T
    data = np.array([u-g, g-r, r-i])
    SVMclassifier(np.array([[1,2,3,4,5],[6,7,8,9,10],[11,12,13,14,15]]))
