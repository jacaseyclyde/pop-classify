# -*- coding: utf-8 -*-
"""
pop-classify main function

Authors:
    J. Andrew Casey-Clyde (@jacaseyclyde)
    Alex Colebaugh
    Kevin Prasad
"""

import numpy as np


if __name__ == "__main__":
    # import the data from csv. we can either keep doing a file like this
    # or do the data import in python. current data is just anything
    # useful, we should feel free to change the data we pull based on what
    # we actually use
    ra, dec, u, g, r, i, z, redshift = np.genfromtxt('data.csv', delimiter=',',
                                                     skip_header=2).T
    # the rest of this space can be used for actually running everything. do
    # other development either in a different file or above the main function
