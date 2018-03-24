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

from sklearn import tree

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import roc_curve, auc


warnings.filterwarnings('ignore', category=DeprecationWarning)

# =============================================================================
# =============================================================================
# # Globals/Init
# =============================================================================
# =============================================================================
ckeys = ['O', 'B', 'A', 'F', 'G', 'K', 'M', 'T', 'L', 'C', 'W']
cols = ['#006D82', '#82139F', '#005AC7', '#009FF9', '#F978F9', '#13D2DC',
        '#AA093B', '#F97850', '#09B45A', '#EFEF31', '#9FF982', '#F9E6BD']
cdict = dict(zip(ckeys, sns.color_palette(cols, len(ckeys))))

n_classes = len(ckeys)

# init data save locations
if not os.path.exists('./out'):
    os.makedirs('./out')

if not os.path.exists('../doc/img'):
    os.makedirs('../doc/img')

if not os.path.exists('./out/pdf'):
    os.makedirs('./out/pdf')

if not os.path.exists('./out/png'):
    os.makedirs('./out/png')

# =============================================================================
# =============================================================================
# # Function Definitions
# =============================================================================
# =============================================================================

# =============================================================================
# Analysis Functions
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
                  fontsize=20)
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
                ax1[i, j].set_ylabel(labels[i + 1], fontsize=20)

            if i == nAx - 2:
                ax1[i, j].set_xlabel(labels[j], fontsize=20)

    fig1.subplots_adjust(hspace=0, wspace=0)

    recs = []
    for i in range(0, len(ckeys)):
        recs.append(mpatches.Circle((0, 0), radius=50, fc=cdict[ckeys[i]]))

    labels = []
    for c in range(ckeys):
        label = ''
        if c == 'W':
            label.append('White Dwarf')
        elif c == 'C':
            label.append('Carbon Star')
        else:
            label.append('Class {0} Stars'.format(c))

    ax1[0, nAx - 2].legend(recs, ckeys, loc="upper right", ncol=2, fontsize=20)

#    plt.show()

    fig1.savefig('../doc/img/{0}.png'.format(filename))


def decision_plot(X, y):

    col_class = []
    for c in y:
        col_class.append(cdict[c])

    col_class = np.array(col_class)

    fig1 = plt.figure(figsize=(12, 12))

    plt.xlabel('$g-r$', fontsize=40)
    plt.ylabel('$r-i$', fontsize=40)

    # Plot the training points
    plt.scatter(X[:, 1], X[:, 2], c=col_class, edgecolor='black', s=50)
    plt.plot([np.min(X[:, 1]), np.max(X[:, 1])], [.62, .62], color='k',
             linestyle='-', linewidth=2)

    plt.title("Single Decision Boundary", fontsize=40)
    plt.axis("tight")
    fig1.savefig('../doc/img/rf_boundary.png')
    plt.show()


def decision_tree_plot(X, y, labels):
    clf = tree.DecisionTreeClassifier(random_state=0).fit(X, y)
    tree.export_graphviz(clf, out_file='./out/tree.dot', max_depth=1,
                         feature_names=labels, class_names=ckeys,)


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
    plt.xlabel('False Positive Rate', fontsize=40)
    plt.ylabel('True Positive Rate', fontsize=40)
    plt.title('Stellar Type Classification ROC curves: {0}'.format(clfType),
              fontsize=40)
    plt.legend(loc="lower right", fontsize=40)

#    plt.show()

    fig1.savefig('./out/pdf/{0}_roc.pdf'.format(shortType))
    fig1.savefig('./out/png/{0}_roc.png'.format(shortType))
    fig1.savefig('../pres/img/{0}_roc.png'.format(shortType))

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
    plt.xlabel('False Positive Rate', fontsize=40)
    plt.ylabel('True Positive Rate', fontsize=40)
    plt.title('Micro-averaged ROC curve: {0}'.format(name), fontsize=40)
    plt.legend(loc="lower right", fontsize=40)

    plt.savefig('./out/pdf/{0}_micro_roc.pdf'.format(s_name))
    plt.savefig('./out/png/{0}_micro_roc.png'.format(s_name))
    plt.savefig('../pres/img/{0}_micro_roc.png'.format(s_name))

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
    plt.xlabel('False Positive Rate', fontsize=40)
    plt.ylabel('True Positive Rate', fontsize=40)
    plt.title('Macro-averaged ROC curve: {0}'.format(name), fontsize=40)
    plt.legend(loc="lower right", fontsize=40)

    plt.savefig('./out/pdf/{0}_macro_roc.pdf'.format(s_name))
    plt.savefig('./out/png/{0}_macro_roc.png'.format(s_name))
    plt.savefig('../pres/img/{0}_macro_roc.png'.format(s_name))

#    plt.show()

    return t_trains, t_tests, scores

# =============================================================================
# Classifiers
# =============================================================================


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
    n_est = 10000

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
# =============================================================================
# # Main Program
# =============================================================================
# =============================================================================


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

    # TODO: Optimize for Carbon star classes/white dwarfs/brown dwarfs
    stellar_class = []
    for c in subclass:
        stellar_class.append(c[0])
    stellar_class = np.array(stellar_class)

    # split data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(colordata,
                                                        stellar_class,
                                                        test_size=.2,
                                                        random_state=0,
                                                        stratify=stellar_class)

    print("Complete!")
    print('==================================================================')
    
    decision_plot(colordata, stellar_class)

#    # Plot all datasets
#    print("Plotting data...")
#
#
    ax_labels = ['u-g', 'g-r', 'r-i', 'i-z']
#    corner_plot(colordata.T, stellar_class, ax_labels, 'All', 'color_corner')
#    corner_plot(X_train.T, y_train, ax_labels, 'Training', 'train_corner')
#    corner_plot(X_test.T, y_test, ax_labels, 'Test', 'test_corner')
#
#    print("Complete!")
#    print('==================================================================')

    #decision_tree_plot(X_train, y_train, ax_labels)

    # Do cross validation on training data for better statistics
#    funcs = [svm_analysis, svm_rbf_analysis, svm_lin_analysis]#, rand_forest_analysis, knneighbors,
#             #gnb_analysis]
#    names = ['Support Vector Machine', 'SVM RBF', 'SVM Linear']#, 'Random Forest',
#             #'K-Nearest Neighbors-distance weighting', 'Gaussian Naive Bayes']
#    s_names = ['svm', 'svm_rbf', 'svm_lin']#, 'rf', 'knn', 'gnb']
#    for func, name, s_name in zip(funcs, names, s_names):
#        print('Starting {0} k-fold analysis'.format(name))
#        t_trains, t_tests, scores = k_fold_analysis(func, X_train, y_train,
#                                                    name, s_name)
#        tpr, fpr, roc_auc, t_train, t_test, score = func(X_train, X_test,
#                                                         y_train, y_test)
#
#        print('t_train = {0:.3f} +/- {1:.3f}, t_test = {2:.3f} +/- {3:.3f}, \
#              score = {4:.3f} +/- {5:.3f}'.format(np.mean(t_trains),
#                                                  np.std(t_trains),
#                                                  np.mean(t_tests),
#                                                  np.std(t_tests),
#                                                  np.mean(scores),
#                                                  np.std(scores)))
#
#        roc_plot(tpr, fpr, roc_auc, name, s_name)
#        print('t_train = {0:.3f}, t_test = {1:.3f}, score = {2:.3f}'
#              .format(t_train, t_test, score))
#        print("==============================================================")

    print('Analysis Complete!')
