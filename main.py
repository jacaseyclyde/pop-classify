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



if __name__ == "__main__":
    # import the data from csv. we can either keep doing a file like this
    # or do the data import in python. current data is just anything
    # useful, we should feel free to change the data we pull based on what
    # we actually use
    [ra, dec, u, g, r, i, z, redshift] = np.genfromtxt('data.csv', delimiter=',',
                                                     skip_header=2)
    
    data = np.array([u-g,g-r,r-i])
    print(data.shape)
