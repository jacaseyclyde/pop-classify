"""
gmm implementation
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture as GMM
from sklearn.model_selection import train_test_split

cdict = {'O':'blue','B':'lightskyblue','A':'white','F':'lightyellow',
             'G':'yellow','K':'orange','M':'red','T':'brown','L':'saddlebrown',
             'C':'black','W':'purple'}

# create data
u, g, r, i, z = np.genfromtxt('data.csv', delimiter=',', skip_header=2,
                                  usecols=(0,1,2,3,4)).T
subclass = np.genfromtxt('data.csv', delimiter=',',skip_header=2,usecols=5,
                             dtype=str)
colordata = np.array([u-g, g-r, r-i, i-z]).T

# deletes outliers
i_extr = np.where(np.logical_or.reduce(np.abs(colordata) > 100, axis=1))
colordata = np.delete(colordata, i_extr, axis=0)
subclass = np.delete(subclass, i_extr, axis=0)

# create subclass array
stellar_class = []
for c in subclass:
    stellar_class.append(c[0])
stellar_class = np.array(stellar_class)

# split training / test data
clr_train, clr_test, cls_train, cls_test = train_test_split(colordata, stellar_class,
                                                    test_size=.5, random_state=0)
"""
# basic setup for GMM, plotting u-g vs g-r to visualize
gmm = GMM(n_components=4).fit(colordata)
labels = gmm.predict(colordata)
plt.figure(figsize=(10,10))
plt.scatter(colordata[:,0],colordata[:,1],c=labels)
"""
# calculate AIC and BIC for each covariance type and n_component to optimize fit
n_total = 50
aic = np.zeros((4,n_total))
bic = np.copy(aic)
cv_types = ['spherical','tied','diag','full']
for cov in range(4):
    for n in range(1,n_total+1):
        gmm = GMM(n_components=n,covariance_type=cv_types[cov],random_state=0)
        gmm.fit(colordata)
        aic[cov,n-1] = gmm.aic(colordata)
        bic[cov,n-1] = gmm.bic(colordata)

# plot to evaluate
fig, axes = plt.subplots(nrows=2, ncols=2)
fig.set_size_inches(12,12)
k = 0
for a in range(2):
    for b in range(2):
        axes[a,b].set_title(cv_types[k])
        axes[a,b].plot(range(1,n_total+1),aic[k,:],label='AIC')
        axes[a,b].plot(range(1,n_total+1),bic[k,:],label='BIC')
        if a == 1:
            axes[a,b].set_xlabel('n_components')
        if b == 0:
            axes[a,b].set_ylabel('information criterion')
        axes[a,b].legend()
        k += 1
print "Optimal covariance type = 'full'"
print "Minimum n-component value = {}".format(np.argmin(bic[3,:]))