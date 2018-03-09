# -*- coding: utf-8 -*-
"""
pop-classify main function

Authors:
    J. Andrew Casey-Clyde (@jacaseyclyde)
    Alex Colebaugh
    Kevin Prasad
"""
import time

import numpy as np
from scipy import interp

import matplotlib.pyplot as plt

from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import roc_curve, auc

# globals
cdict = {'O':'blue','B':'lightskyblue','A':'white','F':'lightyellow',
             'G':'yellow','K':'orange','M':'red','T':'brown','L':'saddlebrown',
             'C':'black','W':'purple'}

def CornerPlot(data,cat,labels):
    # convert string class labels to color labels (for use w/ scatter)
    
    colClass = []
    for c in cat:
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
                ax1[i, j].scatter(colordata[j], colordata[i + 1], c=colClass, s=50)
                
            if j == 0:
                ax1[i, j].set_ylabel(labels[i + 1])
            
            if i == nAx - 2:
                ax1[i, j].set_xlabel(labels[j])
    
    
    fig1.subplots_adjust(hspace=0, wspace=0)
    fig1.show()
    fig1.savefig('./out/color_corner.pdf')
    fig1.savefig('./out/color_corner.png')

def GramMatrix(data):
    (m, n) = data.shape  # dimensionality and number of points, respectively
    
    # computationally cheaper way to compute Gram matrix
    gram = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1):
            gram[i, j] = np.dot(data[:, i], data[:, j])
            gram[j, i] = gram[i, j]
    
    return gram

def SVMAnalysis(X_train,X_test,y_train,y_test):
    print("Starting SVM analysis")
    print("Initializing...")
    t0 = time.time()
    
    clf = svm.SVC(kernel='precomputed')
    
    # Compute gram matrices for both sets
    print("Computing training Gram matrix...")
    t1 = time.time()
    gram_train = GramMatrix(X_train.T)
    t2 = time.time()
    
    t_gram_train = t2 - t1
    print("Training Gram matrix complete. Time to compute for {0} points: {1} s"
          .format(len(X_train),t_gram_train))
    
    print("Computing test Gram matrix...")
    t1 = time.time()
    gram_test = GramMatrix(X_test.T)
    t2 = time.time()
    
    t_gram_test = t2 - t1
    print("Test Gram matrix complete. Time to compute for {0} points: {1} s"
          .format(len(X_test),t_gram_test))
    
    # Compute basic statistics for SVM
    print("Training SVM...")
    t1 = time.time()
    clf.fit(gram_train, y_train)
    t2 = time.time()
    
    t_train = t2 - t1
    print("SVM training complete. Training time for {0} points: {1} s"
          .format(len(X_train), t_train))
    
    print("Scoring...")
    t1 = time.time()
    score = clf.score(gram_test, y_test)
    t2 = time.time()
    
    t_test = t2 - t1
    print("Scoring complete. Classification time for {0} points: {1} s"
          .format(len(X_train), t_test))
    print("Classifier score: {0}".format(score))
    
    # Generate graphs/data for analysis
    print("Generating ROC Curves...")
    y_unique = np.unique(np.concatenate((y_train,y_test)))
    
    y_train = label_binarize(y_train, classes=y_unique)
    y_test = label_binarize(y_test, classes=y_unique)
    n_classes = len(y_unique)
    
    classifier = OneVsRestClassifier(svm.SVC(kernel='precomputed'))
    y_score = classifier.fit(gram_train, y_train).decision_function(gram_test)
    
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
    
    plt.figure(figsize=(12,12))
    plt.plot(fpr['micro'], tpr['micro'], label='micro-average ROC curve (area = {0:0.2f})'
             .format(roc_auc["micro"]), color='deeppink', linestyle=':',
             linewidth=4)
    
    plt.plot(fpr['macro'], tpr['macro'], label='macro-average ROC curve (area = {0:0.2f})'
             .format(roc_auc["macro"]), color='navy', linestyle=':', linewidth=4)
    
    #colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
    for i in range(n_classes):
        plt.plot(fpr[i], tpr[i], color=cdict[y_unique[i]], lw=2,
                 label='Class {0} Stars (area = {1:0.2f})'
                 ''.format(y_unique[i], roc_auc[i]))
    
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Stellar Class Receiver Operating Characteristics: Support Vector Machine')
    plt.legend(loc="lower right")
    plt.show()
    
    print("SVM analysis complete. Total runtime: {0} s".format(t2 - t0))
    
    return clf, fpr, tpr, roc_auc
    

if __name__ == "__main__":
    # Import the data in 2 seperate stmts b/c genfromtxt doesnt like multityping
    u, g, r, i, z = np.genfromtxt('data.csv', delimiter=',', skip_header=2,
                                  usecols=(0,1,2,3,4)).T
    subclass = np.genfromtxt('data.csv', delimiter=',',skip_header=2,usecols=5,
                             dtype=str)
    
    colordata = np.array([u-g, g-r, r-i, i-z]).T #, u-r, u-i, u-z, g-i, g-z, r-z])
    
    # TODO: Optimize for Carbon star classes/white dwarfs/brown dwarfs
    stellar_class = []
    for c in subclass:
        stellar_class.append(c[0])
    stellar_class = np.array(stellar_class)
   
    axLabels = ['$u-g$', '$g-r$', '$r-i$', '$i-z$'] #, '$u-r$', '$u-i$', '$u-z$',
                #'$g-i$', '$g-z$', '$r-z$']
    
    #CornerPlot(colordata,stellar_class,axLabels)

    # split data into training and test sets
    clr_train, clr_test, cls_train, cls_test = train_test_split(colordata, stellar_class,
                                                                test_size=.5, random_state=0)
    
    clf, fpr, tpr, roc_auc = SVMAnalysis(clr_train, clr_test, cls_train, cls_test)
    #SVMAnalysis(clr_train, clr_test, cls_train, cls_test)
