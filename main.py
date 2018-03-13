# -*- coding: utf-8 -*-
"""
pop-classify main function

Authors:
    J. Andrew Casey-Clyde (@jacaseyclyde)
    Alex Colebaugh
    Kevin Prasad
"""
import time
import os
import warnings

import numpy as np
from scipy import interp

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

from astroML.classification import GMMBayes

from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.mixture import GaussianMixture
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import roc_curve, auc


warnings.filterwarnings('ignore', category=DeprecationWarning)

# globals
ckeys = ['O', 'B', 'A', 'F', 'G', 'K', 'M', 'T', 'L', 'C', 'W']
cols = ['#006D82', '#82139F', '#005AC7', '#009FF9', '#F978F9', '#13D2DC',
        '#AA093B', '#F97850', '#09B45A', '#EFEF31', '#9FF982', '#F9E6BD']
cdict = dict(zip(ckeys, sns.color_palette(cols, len(ckeys))))

# init data save locations
if not os.path.exists('./out'):
    os.makedirs('./out')

if not os.path.exists('./out/pdf'):
    os.makedirs('./out/pdf')

if not os.path.exists('./out/png'):
    os.makedirs('./out/png')


def CornerPlot(data, cat, labels, dataAmt, filename):
    # convert string class labels to color labels (for use w/ scatter)
    print("Creating corner plot of {0} data...".format(dataAmt))

    colClass = []
    for c in cat:
        colClass.append(cdict[c])

    colClass = np.array(colClass)

    # plot the classes/colors
    nAx = len(data)

    fig1, ax1 = plt.subplots(nAx - 1, nAx - 1, sharex=True, sharey=True)
    fig1.suptitle('color-color Corner Plot: {0} Data'.format(dataAmt),
                  fontsize=16)
    fig1.set_size_inches(4 * (nAx - 1), 4 * (nAx - 1))

    ax1[0, 0].set_xticklabels([])
    ax1[0, 0].set_yticklabels([])

    for i in range(nAx - 1):
        for j in range(nAx - 1):
            if j > i:
                ax1[i, j].axis('off')

            else:
                ax1[i, j].scatter(data[j], data[i + 1], c=colClass, s=50)

            if j == 0:
                ax1[i, j].set_ylabel(labels[i + 1])

            if i == nAx - 2:
                ax1[i, j].set_xlabel(labels[j])

    fig1.subplots_adjust(hspace=0, wspace=0)

    recs = []
    for i in range(0, len(ckeys)):
        recs.append(mpatches.Circle((0, 0), radius=50, fc=cdict[ckeys[i]]))

    ax1[0, nAx - 2].legend(recs, ckeys, loc="upper right", ncol=2)

    plt.show()

    fig1.savefig('./out/pdf/{0}.pdf'.format(filename))
    fig1.savefig('./out/png/{0}.png'.format(filename))

    print("Corner plots complete!")


def ROC(clfFn, X_train, X_test, y_train, y_test, clfType, shortType):
    print("Generating ROC Curves...")

    y_train = label_binarize(y_train, classes=ckeys)
    y_test = label_binarize(y_test, classes=ckeys)
    n_classes = len(ckeys)

    clf = OneVsRestClassifier(clfFn)
    y_score = clf.fit(X_train, y_train).predict_proba(X_test)

    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    # aggregate fpr
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

    # interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])

    # average and compute AUC
    mean_tpr /= n_classes

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    fig1 = plt.figure(figsize=(12, 12))
    plt.plot(fpr['micro'], tpr['micro'],
             label='micro-average ROC curve (area = {0:0.3f})'
             .format(roc_auc["micro"]), color='deeppink', linestyle=':',
             linewidth=4)

    plt.plot(fpr['macro'], tpr['macro'],
             label='macro-average ROC curve (area = {0:0.3f})'
             .format(roc_auc["macro"]), color='navy', linestyle=':',
             linewidth=4)

    for i in range(n_classes):
        label = ''
        if ckeys[i] == 'W':
            label = 'White Dwarf  (area = {0:0.3f})'.format(roc_auc[i])
        elif ckeys[i] == 'C':
            label = 'Carbon Star  (area = {0:0.3f})'.format(roc_auc[i])
        else:
            label = 'Class {0} Stars (area = {1:0.3f})'.format(ckeys[i],
                                                               roc_auc[i])

        plt.plot(fpr[i], tpr[i], color=cdict[ckeys[i]], lw=2,
                 label=label)

    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Stellar Type Classification ROC curves: {0}'.format(clfType),
              fontsize=16)
    plt.legend(loc="lower right")

    plt.show()

    fig1.savefig('./out/pdf/{0}_roc.pdf'.format(shortType))
    fig1.savefig('./out/png/{0}_roc.png'.format(shortType))


def GramMatrix(data):
    (m, n) = data.shape  # dimensionality and number of points, respectively

    # computationally cheaper way to compute Gram matrix
    gram = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1):
            gram[i, j] = np.dot(data[:, i], data[:, j])
            gram[j, i] = gram[i, j]

    return gram


def SVMAnalysis(X_train, X_test, y_train, y_test):
    print("Starting Support Vector Machine analysis")
    print("Initializing...")
    t0 = time.time()

    clf = svm.SVC(kernel='precomputed', probability=True)

    # Compute gram matrices for both sets
    print("Computing training Gram matrix...")
    t1 = time.time()
    gram_train = GramMatrix(X_train.T)
    t2 = time.time()

    t_gram_train = t2 - t1
    print("Training Gram matrix complete. Time to compute for {0} pts: \
          {1:0.3f} s"
          .format(len(X_train), t_gram_train))

    print("Computing test Gram matrix...")
    t1 = time.time()
    gram_test = GramMatrix(X_test.T)
    t2 = time.time()

    t_gram_test = t2 - t1
    print("Test Gram matrix complete. Time to compute for {0} points: \
          2{1:0.3f} s"
          .format(len(X_test), t_gram_test))

    # Compute basic statistics for SVM
    print("Training SVM...")
    t1 = time.time()
    clf.fit(gram_train, y_train)
    t2 = time.time()

    t_train = t2 - t1
    print("SVM training complete. Training time for {0} points: {1:0.3f} s"
          .format(len(X_train), t_train))

    print("Scoring...")
    t1 = time.time()
    score = clf.score(gram_test, y_test)
    t2 = time.time()

    t_test = t2 - t1
    print("Scoring complete. Classification time for {0} points: {1:0.3f} s"
          .format(len(X_train), t_test))
    print("Classifier score: {0:0.3f}".format(score))

    # Generate graphs/data for analysis
    ROC(svm.SVC(kernel='precomputed', probability=True), gram_train, gram_test,
        y_train, y_test, "Support Vector Machine", 'svm')

    t2 = time.time()
    print("SVM analysis complete. Total runtime: {0:0.3f} s".format(t2 - t0))

    # return clf, fpr, tpr, roc_auc # can add this back in for debug/dev


def RandForestAnalysis(X_train, X_test, y_train, y_test):
    print("Starting Random Forest analysis")
    print("Initializing...")
    t0 = time.time()

    clf = RandomForestClassifier(n_estimators=1000)

    # Compute basic statistics for SVM
    print("Training Random Forest...")
    t1 = time.time()
    clf.fit(X_train, y_train)
    t2 = time.time()

    t_train = t2 - t1
    print("Random Forest training complete. Training time for {0} pts: \
          {1:0.3f} s"
          .format(len(X_train), t_train))

    print("Scoring...")
    t1 = time.time()
    score = clf.score(X_test, y_test)
    t2 = time.time()

    t_test = t2 - t1
    print("Scoring complete. Classification time for {0} points: {1:0.3f} s"
          .format(len(X_train), t_test))
    print("Classifier score: {0:0.3f}".format(score))

    # Generate graphs/data for analysis
    ROC(RandomForestClassifier(n_estimators=1000), X_train, X_test, y_train,
        y_test, "Random Forest", 'rf')

    t2 = time.time()
    print("Random Forest analysis complete. Total runtime: {0:0.3f} s"
          .format(t2 - t0))

    # return clf, fpr, tpr, roc_auc # can add this back in for debug/dev


def GMM32Analysis(X_train, X_test, y_train, y_test):
    print("Starting 32-component Gaussian Mixture analysis")
    print("Initializing...")
    t0 = time.time()

    clf = GaussianMixture(n_components=32, covariance_type='full',
                          random_state=0)

    print("Training Gaussian Mixture Model...")
    t1 = time.time()
    clf.fit(X_train, y_train)
    t2 = time.time()

    t_train = t2 - t1
    print("Gaussian Mixture training complete. Training time for {0} pts: \
          {1:0.3f} s"
          .format(len(X_train), t_train))

    print("Scoring...")
    t1 = time.time()
    score = clf.score(X_test, y_test)
    t2 = time.time()

    t_test = t2 - t1
    print("Scoring complete. Classification time for {0} points: {1:0.3f} s"
          .format(len(X_train), t_test))
    print("Classifier score: {0:0.3f}".format(score))

    # Generate graphs/data for analysis
    ROC(GaussianMixture(n_components=32, covariance_type='full',
                        random_state=0), X_train, X_test, y_train,
        y_test, "32-component Gaussian Mixture Model", 'GMM32')

    t2 = time.time()
    print("GMM32 analysis complete. Total runtime: {0:0.3f} s"
          .format(t2 - t0))


def GMM11Analysis(X_train, X_test, y_train, y_test):
    print("Starting 11-component Gaussian Mixture analysis")
    print("Initializing...")
    t0 = time.time()

    clf = GaussianMixture(n_components=11, covariance_type='full',
                          random_state=0)

    print("Training Gaussian Mixture Model...")
    t1 = time.time()
    clf.fit(X_train, y_train)
    t2 = time.time()

    t_train = t2 - t1
    print("Gaussian Mixture training complete. Training time for {0} pts: \
          {1:0.3f} s"
          .format(len(X_train), t_train))

    print("Scoring...")
    t1 = time.time()
    score = clf.score(X_test, y_test)
    t2 = time.time()

    t_test = t2 - t1
    print("Scoring complete. Classification time for {0} points: {1:0.3f} s"
          .format(len(X_train), t_test))
    print("Classifier score: {0:0.3f}".format(score))

    # Generate graphs/data for analysis
    ROC(GaussianMixture(n_components=11, covariance_type='full',
                        random_state=0), X_train, X_test, y_train, y_test,
        "11-component Gaussian Mixture Model", 'GMM11')

    t2 = time.time()
    print("GMM11 analysis complete. Total runtime: {0:0.3f} s"
          .format(t2 - t0))


def GNBAnalysis(X_train, X_test, y_train, y_test):
    print("Starting Gaussian Naive Bayesian analysis")
    print("Initializing...")
    t0 = time.time()

    clf = GaussianNB()

    print("Training Gaussian Naive Bayes...")
    t1 = time.time()
    clf.fit(X_train, y_train)
    t2 = time.time()

    t_train = t2 - t1
    print("GNB training complete. Training time for {0} pts: {1:0.3f} s"
          .format(len(X_train), t_train))

    print("Scoring...")
    t1 = time.time()
    score = clf.score(X_test, y_test)
    t2 = time.time()

    t_test = t2 - t1
    print("Scoring complete. Classification time for {0} points: {1:0.3f} s"
          .format(len(X_train), t_test))
    print("Classifier score: {0:0.3f}".format(score))

    # Generate graphs/data for analysis
    ROC(GaussianNB(), X_train, X_test, y_train,
        y_test, "Gaussian Naive Bayes", 'GNB')

    t2 = time.time()
    print("GNB analysis complete. Total runtime: {0:0.3f} s"
          .format(t2 - t0))


def GMMBayesAnalysis(X_train, X_test, y_train, y_test):
    print("Starting Gaussian Mixture Model Bayesian analysis")
    print("Initializing...")
    t0 = time.time()

    clf = GMMBayes()

    print("Training GMMBayes...")
    t1 = time.time()
    clf.fit(X_train, y_train)
    t2 = time.time()

    t_train = t2 - t1
    print("GMMBayes training complete. Training time for {0} pts: {1:0.3f} s"
          .format(len(X_train), t_train))

    print("Scoring...")
    t1 = time.time()
    score = clf.score(X_test, y_test)
    t2 = time.time()

    t_test = t2 - t1
    print("Scoring complete. Classification time for {0} points: {1:0.3f} s"
          .format(len(X_train), t_test))
    print("Classifier score: {0:0.3f}".format(score))

    # Generate graphs/data for analysis
    ROC(GMMBayes(), X_train, X_test, y_train,
        y_test, "Gaussian Mixture Model Bayesian", 'GMMB')

    t2 = time.time()
    print("GMMBayes analysis complete. Total runtime: {0:0.3f} s"
          .format(t2 - t0))


def knneighbors(neighbors, wweights, clr_train, clr_test, cls_train, cls_test):
    print('K-Nearest Neighbors Classification')
    t0 = time.time()
    # n_neighbors = number of neighbors by which selection is made
    # weights = 'uniform' where all points are weighed equally or 'distance'
    # where points are weighed as an inverse of distance from test point
    neigh = KNeighborsClassifier(n_neighbors=neighbors, weights=wweights)

    print('Training')
    t1 = time.time()
    neigh.fit(clr_train, cls_train)
    t2 = time.time()

    t_train = t2-t1
    print('Training complete. Time for {0} points was {1:0.3f} s'
          .format(len(clr_train), t_train))

    print('Scoring')
    t1 = time.time()
    score = neigh.score(clr_test, cls_test)
    t2 = time.time()

    t_score = t2-t1
    print('Scoring complete. Time for {0} points was {1:0.3f} s'
          .format(len(clr_test), t_score))
    print("Classifier score: {0:0.3f}".format(score))

    # Analysis Graph
    ROC(KNeighborsClassifier(neighbors, wweights), clr_train, clr_test,
        cls_train, cls_test, "K-Nearest Neighbors-{0} weighting"
        .format(wweights), 'knn')

    tf = time.time()

    print('K-Nearest Neighbors Complete. Runtime {0:0.3f} s.'.format(tf-t0))


if __name__ == "__main__":
    # Import the data in 2 stmts b/c genfromtxt doesnt like multi-typing
    print("Importing data...")
    u, g, r, i, z = np.genfromtxt('data.csv', delimiter=',', skip_header=2,
                                  usecols=(0, 1, 2, 3, 4)).T
    subclass = np.genfromtxt('data.csv', delimiter=',', skip_header=2,
                             usecols=5, dtype=str)
    print("Import complete!")

    print("Preprocessing...")
    colordata = np.array([u-g, g-r, r-i, i-z]).T

    # check for extreme outliers with magnitude differences greater than 100
    i_extr = np.where(np.logical_or.reduce(np.abs(colordata) > 100, axis=1))

    colordata = np.delete(colordata, i_extr, axis=0)
    subclass = np.delete(subclass, i_extr, axis=0)

    print("Complete!")
    print('==================================================================')

    # Plot data along each set of axes
    # TODO: Optimize for Carbon star classes/white dwarfs/brown dwarfs
    stellar_class = []
    for c in subclass:
        stellar_class.append(c[0])
    stellar_class = np.array(stellar_class)

    axLabels = ['$u-g$', '$g-r$', '$r-i$', '$i-z$']

    CornerPlot(colordata.T, stellar_class, axLabels, 'All', 'color_corner')

    # split data into training and test sets
    clr_train, clr_test, cls_train, cls_test = train_test_split(colordata,
                                                                stellar_class,
                                                                test_size=.5,
                                                                random_state=0)

    # Plot the training and test sets - just in case it's a weird split
    CornerPlot(clr_train.T, cls_train, axLabels, 'Training', 'train_corner')
    CornerPlot(clr_test.T, cls_test, axLabels, 'Test', 'test_corner')

    # Analysis
    SVMAnalysis(clr_train, clr_test, cls_train, cls_test)
    print("==================================================================")
    RandForestAnalysis(clr_train, clr_test, cls_train, cls_test)
    print("==================================================================")
    GMM32Analysis(clr_train, clr_test, cls_train, cls_test)
    print("==================================================================")
    GMM11Analysis(clr_train, clr_test, cls_train, cls_test)
    print("==================================================================")
    GNBAnalysis(clr_train, clr_test, cls_train, cls_test)
    print("==================================================================")
    GMMBayesAnalysis(clr_train, clr_test, cls_train, cls_test)
    print("==================================================================")
    knneighbors(1000, 'distance', clr_train, clr_test, cls_train, cls_test)

    print('Analysis Complete!')
