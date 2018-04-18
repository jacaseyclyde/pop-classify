# -*- coding: utf-8 -*-
"""
pop-classify main function

Authors:
    J. Andrew Casey-Clyde (@jacaseyclyde)
    Alex Colebaugh
    Kevin Prasad
"""
from __future__ import absolute_import, division, print_function

from clustering import svm_analysis, svm_rbf_analysis, svm_lin_analysis

import os
import warnings

import numpy as np
from scipy import interp

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

import tensorflow as tf

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import auc


warnings.filterwarnings('ignore', category=DeprecationWarning)

# =============================================================================
# =============================================================================
# # Globals/Init
# =============================================================================
# =============================================================================
ckeys = ['O', 'B', 'A', 'F', 'G', 'K', 'M',
         'T Tauri', 'L', 'Carbon', 'White Dwarf']

class_dict = {'O': 0, 'B': 1, 'A': 2, 'F': 3, 'G': 4, 'K': 5, 'M': 6, 'L': 7,
              'T': 8, 'Carbon': 9, 'Carbon White Dwarf': 10,
              'Carbon Lines': 11, 'White Dwarf': 12, 'Magnetic White Dwarf':13,
              'Cataclysmic Variable': 14}

cols = ['#006D82', '#82139F', '#005AC7', '#009FF9', '#F978F9', '#13D2DC',
        '#AA093B', '#F97850', '#09B45A', '#EFEF31', '#9FF982', '#F9E6BD']
cdict = dict(zip(ckeys, sns.color_palette(cols, len(ckeys))))

n_classes = len(ckeys)

# init data save locations
if not os.path.exists('./out'):
    os.makedirs('./out')


# =============================================================================
# =============================================================================
# # Function Definitions
# =============================================================================
# =============================================================================

# =============================================================================
# Data Handling
# =============================================================================

def import_data():
    print("Importing data...")
    u, g, r, i, z = np.genfromtxt('./data/data.csv', delimiter=',',
                                  skip_header=2, usecols=(0, 1, 2, 3, 4)).T
    subclass = np.genfromtxt('./data/data.csv', delimiter=',', skip_header=2,
                             usecols=5, dtype=str)
    print("Import complete!")

    print("Preprocessing...")
    features = np.array([u-g, g-r, r-i, i-z]).T

    # check for extreme outliers with magnitude differences greater than 100
    i_extr = np.where(np.logical_or.reduce(np.abs(features) > 100, axis=1))

    features = np.delete(features, i_extr, axis=0)
    subclass = np.delete(subclass, i_extr, axis=0)

    # TODO: Optimize for Carbon star classes/white dwarfs/brown dwarfs
    labels = []
    for c in subclass:
        if c[0] in ['O', 'B', 'A', 'F', 'G', 'K', 'M', 'L', 'T']:
            labels.append(class_dict[c[0]])
        elif c == 'Carbon':
            labels.append(class_dict['Carbon'])
        elif c == 'CarbonWD':
            labels.append(class_dict['Carbon White Dwarf'])
        elif c == 'Carbon_lines':
            labels.append(class_dict['Carbon Lines'])
        elif c == 'WD':
            labels.append(class_dict['White Dwarf'])
        elif c == 'WDmagnetic':
            labels.append(class_dict['Magnetic White Dwarf'])
        elif c == 'CV':
            labels.append(class_dict['Cataclysmic Variable'])

    labels = np.array(labels)

    return features, labels


def input_fn(features, labels, batch_size):
    """An input function for training"""
    # Convert the inputs to a Dataset.
    dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))

    # Shuffle, repeat, and batch the examples.
    return dataset.shuffle(1000).repeat().batch(batch_size)


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


# =============================================================================
# Analysis
# =============================================================================

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


def main():
    # Import the data in 2 stmts b/c genfromtxt doesnt like multi-typing

    # split data into training and test sets
    features, labels = import_data()
    train_x, test_x, train_y, test_y = train_test_split(features,
                                                        labels,
                                                        test_size=.2,
                                                        random_state=0,
                                                        stratify=labels)

    print("Complete!")
    print('==================================================================')

    # Plot all datasets
    print("Plotting data...")

#    axLabels = ['$u-g$', '$g-r$', '$r-i$', '$i-z$']
#    corner_plot(colordata.T, stellar_class, axLabels, 'All', 'color_corner')
#    corner_plot(X_train.T, y_train, axLabels, 'Training', 'train_corner')
#    corner_plot(X_test.T, y_test, axLabels, 'Test', 'test_corner')

    print("Complete!")
    print('==================================================================')

    # Do cross validation on training data for better statistics
    funcs = [svm_analysis, svm_rbf_analysis, svm_lin_analysis]
    names = ['Support Vector Machine', 'SVM RBF', 'SVM Linear']
    s_names = ['svm', 'svm_rbf', 'svm_lin']
    for func, name, s_name in zip(funcs, names, s_names):
        print('Starting {0} k-fold analysis'.format(name))
        t_trains, t_tests, scores = k_fold_analysis(func, train_x, train_y,
                                                    name, s_name)
        tpr, fpr, roc_auc, t_train, t_test, score = func(train_x, test_x,
                                                         train_y, test_y)

        print('t_train = {0:.3f} +/- {1:.3f}, t_test = {2:.3f} +/- {3:.3f}, \
              score = {4:.3f} +/- {5:.3f}'.format(np.mean(t_trains),
                                                  np.std(t_trains),
                                                  np.mean(t_tests),
                                                  np.std(t_tests),
                                                  np.mean(scores),
                                                  np.std(scores)))

        roc_plot(tpr, fpr, roc_auc, name, s_name)
        print('t_train = {0:.3f}, t_test = {1:.3f}, score = {2:.3f}'
              .format(t_train, t_test, score))
        print("==============================================================")

    print('Analysis Complete!')


if __name__ == "__main__":
    # main()
    features, labels = import_data()

    train_x, test_x, train_y, test_y = train_test_split(features,
                                                        labels,
                                                        test_size=.2,
                                                        random_state=0,
                                                        stratify=labels)

    train_x = dict(zip(['u-g', 'g-r', 'r-i', 'i-z'], train_x.T))
    test_x = dict(zip(['u-g', 'g-r', 'r-i', 'i-z'], test_x.T))

    my_feature_columns = []
    for key in train_x.keys():
        my_feature_columns.append(tf.feature_column.numeric_column(key=key))

    classifier = tf.estimator.DNNClassifier(feature_columns=my_feature_columns,
                                            hidden_units=[10, 10],
                                            n_classes=len(np.unique(labels)))

    classifier.train(input_fn=lambda: input_fn(train_x, train_y, 100),
                     steps=1000)

    eval_result = classifier.evaluate(input_fn=lambda: input_fn(test_x, test_y,
                                                                100),
                                      steps=1000)

