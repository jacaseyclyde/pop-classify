#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 16 23:26:14 2018

@author: jacaseyclyde
"""
import time

import numpy as np
from scipy import interp

from astroML.classification import GMMBayes

from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.mixture import GaussianMixture
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier

from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import roc_curve, auc

ckeys = ['O', 'B', 'A', 'F', 'G', 'K', 'M', 'T', 'L', 'C', 'W']
n_classes = len(ckeys)


def roc_calc(clfFn, X_train, X_test, y_train, y_test):
    y_train = label_binarize(y_train, classes=ckeys)
    y_test = label_binarize(y_test, classes=ckeys)

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

    return tpr, fpr, roc_auc


def svm_analysis(X_train, X_test, y_train, y_test):
    scale_max = np.max(np.abs(X_train))
    X_train = X_train / scale_max
    X_test = X_test / scale_max

    clf = svm.SVC(kernel='precomputed', probability=True)

    # Compute gram matrices for both sets
    t1 = time.time()
    train = np.dot(X_train, X_train.T)
    t2 = time.time()

    t_gram_train = t2 - t1

    t1 = time.time()
    test = np.dot(X_test, X_train.T)
    t2 = time.time()

    t_gram_test = t2 - t1

    # Compute basic statistics for SVM
    t1 = time.time()
    clf.fit(train, y_train)
    t2 = time.time()

    t_train = t2 - t1

    t1 = time.time()
    score = clf.score(test, y_test)
    t2 = time.time()

    t_test = t2 - t1

    t_train += t_gram_train
    t_test += t_gram_test

    # Generate graphs/data for analysis
    tpr, fpr, roc_auc = roc_calc(svm.SVC(kernel='precomputed',
                                         probability=True), train, test,
                                 y_train, y_test)

    return tpr, fpr, roc_auc, t_train, t_test, score


def svm_rbf_analysis(X_train, X_test, y_train, y_test):
    # preprocessing
    scale_max = np.max(np.abs(X_train))
    X_train = X_train / scale_max
    X_test = X_test / scale_max

    clf = svm.SVC(kernel='rbf', probability=True)
    # Compute basic statistics for SVM
    t1 = time.time()
    clf.fit(X_train, y_train)
    t2 = time.time()

    t_train = t2 - t1

    t1 = time.time()
    score = clf.score(X_test, y_test)
    t2 = time.time()

    t_test = t2 - t1

    # Generate graphs/data for analysis
    tpr, fpr, roc_auc = roc_calc(svm.SVC(kernel='rbf',
                                         probability=True), X_train, X_test,
                                 y_train, y_test)

    return tpr, fpr, roc_auc, t_train, t_test, score


def svm_lin_analysis(X_train, X_test, y_train, y_test):
    # preprocessing
    scale_max = np.max(np.abs(X_train))
    X_train = X_train / scale_max
    X_test = X_test / scale_max

    clf = svm.SVC(kernel='linear', probability=True)
    # Compute basic statistics for SVM
    t1 = time.time()
    clf.fit(X_train, y_train)
    t2 = time.time()

    t_train = t2 - t1

    t1 = time.time()
    score = clf.score(X_test, y_test)
    t2 = time.time()

    t_test = t2 - t1

    # Generate graphs/data for analysis
    tpr, fpr, roc_auc = roc_calc(svm.SVC(kernel='linear',
                                         probability=True), X_train, X_test,
                                 y_train, y_test)

    return tpr, fpr, roc_auc, t_train, t_test, score


def rand_forest_analysis(X_train, X_test, y_train, y_test):
    n_est = 1000

    clf = RandomForestClassifier(n_estimators=n_est)

    # Compute basic statistics for SVM
    t1 = time.time()
    clf.fit(X_train, y_train)
    t2 = time.time()

    t_train = t2 - t1

    t1 = time.time()
    score = clf.score(X_test, y_test)
    t2 = time.time()

    t_test = t2 - t1

    # Generate graphs/data for analysis
    tpr, fpr, roc_auc = roc_calc(RandomForestClassifier(n_estimators=n_est),
                                 X_train, X_test, y_train, y_test)

    return tpr, fpr, roc_auc, t_train, t_test, score


def gmm_32_analysis(X_train, X_test, y_train, y_test):
    clf = GaussianMixture(n_components=32, covariance_type='full',
                          random_state=0)

    t1 = time.time()
    clf.fit(X_train, y_train)
    t2 = time.time()

    t_train = t2 - t1

    t1 = time.time()
    score = clf.score(X_test, y_test)
    t2 = time.time()

    t_test = t2 - t1

    # Generate graphs/data for analysis
    tpr, fpr, roc_auc = roc_calc(GaussianMixture(n_components=32,
                                                 covariance_type='full',
                                                 random_state=0), X_train,
                                 X_test, y_train, y_test)

    return tpr, fpr, roc_auc, t_train, t_test, score


def gmm_11_analysis(X_train, X_test, y_train, y_test):
    clf = GaussianMixture(n_components=11, covariance_type='full',
                          random_state=0)

    t1 = time.time()
    clf.fit(X_train, y_train)
    t2 = time.time()

    t_train = t2 - t1

    t1 = time.time()
    score = clf.score(X_test, y_test)
    t2 = time.time()

    t_test = t2 - t1

    # Generate graphs/data for analysis
    tpr, fpr, roc_auc = roc_calc(GaussianMixture(n_components=11,
                                                 covariance_type='full',
                                                 random_state=0),
                                 X_train, X_test, y_train, y_test)

    return tpr, fpr, roc_auc, t_train, t_test, score


def gnb_analysis(X_train, X_test, y_train, y_test):
    clf = GaussianNB()

    t1 = time.time()
    clf.fit(X_train, y_train)
    t2 = time.time()

    t_train = t2 - t1

    t1 = time.time()
    score = clf.score(X_test, y_test)
    t2 = time.time()

    t_test = t2 - t1

    # Generate graphs/data for analysis
    tpr, fpr, roc_auc = roc_calc(GaussianNB(), X_train, X_test,
                                 y_train, y_test)

    return tpr, fpr, roc_auc, t_train, t_test, score


def gmm_bayes_analysis(X_train, X_test, y_train, y_test):
    clf = GMMBayes()

    t1 = time.time()
    clf.fit(X_train, y_train)
    t2 = time.time()

    t_train = t2 - t1

    t1 = time.time()
    score = clf.score(X_test, y_test)
    t2 = time.time()

    t_test = t2 - t1

    # Generate graphs/data for analysis
    tpr, fpr, roc_auc = roc_calc(GMMBayes(), X_train, X_test, y_train, y_test)

    return tpr, fpr, roc_auc, t_train, t_test, score


def knneighbors(clr_train, clr_test, cls_train, cls_test):
    nn = 1000
    # n_neighbors = number of neighbors by which selection is made
    # weights = 'uniform' where all points are weighed equally or 'distance'
    # where points are weighed as an inverse of distance from test point
    neigh = KNeighborsClassifier(n_neighbors=nn, weights='distance')

    t1 = time.time()
    neigh.fit(clr_train, cls_train)
    t2 = time.time()

    t_train = t2-t1

    t1 = time.time()
    score = neigh.score(clr_test, cls_test)
    t2 = time.time()

    t_test = t2-t1

    # Analysis Graph
    tpr, fpr, roc_auc = roc_calc(KNeighborsClassifier(nn, 'distance'),
                                 clr_train, clr_test, cls_train, cls_test)

    return tpr, fpr, roc_auc, t_train, t_test, score
