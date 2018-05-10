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


# =============================================================================
# Plotting
# =============================================================================

def corner_plot(data, cat, labels, data_amt, filename):
    # convert string class labels to color labels (for use w/ scatter)
    colClass = []
    for c in cat:
        colClass.append(cdict[c])

    colClass = np.array(colClass)

    # plot the classes/colors
    nAx = len(data)

    fig1, ax1 = plt.subplots(nAx - 1, nAx - 1, sharex=True, sharey=True)
    fig1.suptitle('color-color Corner Plot: {0} Data'.format(data_amt),
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

    fig1.savefig('./out/{0}.pdf'.format(filename))


def roc_plot(tpr, fpr, roc_auc, clfType, shortType):
    print('Plotting ROC curve...')
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

    fig1.savefig('./out/{0}_roc.pdf'.format(shortType))

    print('Plot complete!')


def k_fold_analysis(func, X_train, y_train, name, s_name):
    cv = StratifiedKFold(n_splits=5, random_state=0)

    micro_tprs = []
    macro_tprs = []

    micro_aucs = []
    macro_aucs = []

    t_trains = []
    t_tests = []
    scores = []

    mean_fpr = np.linspace(0, 1, 1000)

    i = 0
    for train, test in cv.split(X_train, y_train):
        print("Fold {0}...".format(i))
        tpr, fpr, roc_auc, t_train, t_test, score = func(X_train[train],
                                                         X_train[test],
                                                         y_train[train],
                                                         y_train[test])

        micro_tprs.append(interp(mean_fpr, fpr['micro'], tpr['micro']))
        macro_tprs.append(interp(mean_fpr, fpr['macro'], tpr['macro']))

        micro_aucs.append(roc_auc['micro'])
        macro_aucs.append(roc_auc['macro'])

        t_trains.append(t_train)
        t_tests.append(t_test)
        scores.append(score)

        # Plot micro stats
        plt.figure(0, figsize=(12, 12))
        plt.plot(fpr['micro'], tpr['micro'], lw=1, alpha=0.3,
                 label='ROC fold {0} (AUC = {1:.3f})'
                 .format(i, roc_auc['micro']))

        # Plot macro stats
        plt.figure(1, figsize=(12, 12))
        plt.plot(fpr['macro'], tpr['macro'], lw=1, alpha=0.3,
                 label='ROC fold {0} (AUC = {1:.3f})'
                 .format(i, roc_auc['macro']))

        i += 1

    # Micro
    plt.figure(0)
    plt.plot([0, 1], [0, 1], linestyle='--', lw=2,
             color='k', alpha=.8)

    mean_tpr = np.mean(micro_tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(micro_aucs)
    plt.plot(mean_fpr, mean_tpr, color='b',
             label=r'Mean ROC (AUC = {0:.3f} $\pm$ {1:.3f})'
             .format(mean_auc, std_auc), lw=2, alpha=.8)

    std_tpr = np.std(micro_tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                     label=r'$\pm$ 1 std. dev.')

    plt.xlim([0., 1.0])
    plt.ylim([0., 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Micro-averaged ROC curve: {0}'.format(name))
    plt.legend(loc="lower right")

    plt.savefig('./out/{0}_micro_roc.pdf'.format(s_name))

    # Macro
    plt.figure(1)
    plt.plot([0, 1], [0, 1], linestyle='--', lw=2,
             color='k', alpha=.8)

    mean_tpr = np.mean(macro_tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(macro_aucs)
    plt.plot(mean_fpr, mean_tpr, color='b',
             label=r'Mean ROC (AUC = {0:.3f} $\pm$ {1:.3f})'
             .format(mean_auc, std_auc), lw=2, alpha=.8)

    std_tpr = np.std(macro_tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                     label=r'$\pm$ 1 std. dev.')

    plt.xlim([0., 1.0])
    plt.ylim([0., 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Macro-averaged ROC curve: {0}'.format(name))
    plt.legend(loc="lower right")

    plt.savefig('./out/{0}_macro_roc.pdf'.format(s_name))

    plt.show()

    return t_trains, t_tests, scores

