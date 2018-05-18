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
import shutil
import warnings

from tqdm import tqdm, trange

import numpy as np

from scipy import interp
from scipy.optimize import curve_fit

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

import tensorflow as tf

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import auc


warnings.filterwarnings('ignore', category=DeprecationWarning)
tf.logging.set_verbosity(tf.logging.ERROR)
tqdm.monitor_interval = 0


# =============================================================================
# =============================================================================
# # Globals/Init
# =============================================================================
# =============================================================================
ckeys = ['O', 'B', 'A', 'F', 'G', 'K', 'M',
         'T Tauri', 'L', 'Carbon', 'White Dwarf']

cols = ['#006D82', '#82139F', '#005AC7', '#009FF9', '#F978F9', '#13D2DC',
        '#AA093B', '#F97850', '#09B45A', '#EFEF31', '#9FF982', '#F9E6BD']
cdict = dict(zip(ckeys, sns.color_palette(cols, len(ckeys))))

n_classes = len(ckeys)

model_dir = './models'

# init data save locations
if not os.path.exists('./out'):
    os.makedirs('./out')

# neural network configs
n_layers = 8
n_nodes = 26
m_train = 1.

fig_size=(3, 3)


# =============================================================================
# =============================================================================
# # Function Definitions
# =============================================================================
# =============================================================================

# =============================================================================
# Math
# =============================================================================

def sigmoid(x, x0, k, a):
    y = a / (1 + np.exp(-k*(x-x0)))  # + c
    return y


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
            labels.append(c[0])
        elif c == 'Carbon':
            labels.append('Carbon')
        elif c == 'CarbonWD':
            labels.append('Carbon White Dwarf')
        elif c == 'CalciumWD':
            labels.append('Calcium White Dwarf')
        elif c == 'Carbon_lines':
            labels.append('Carbon Lines')
        elif c == 'WD':
            labels.append('White Dwarf')
        elif c == 'WDcooler':
            labels.append('Cool White Dwarf')
        elif c == 'WDhotter':
            labels.append('Hot White Dwarf')
        elif c == 'WDmagnetic':
            labels.append('Magnetic White Dwarf')
        elif c == 'CV':
            labels.append('Cataclysmic Variable')
        elif ('sd:F0' in c) or ('sdF3' in c):
            labels.append('F')

    labels = np.array(labels)

    return features, labels


def train_fn(features, labels, batch_size):
    """An input function for training"""
    # Convert the inputs to a Dataset.
    dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))

    # Shuffle, repeat, and batch the examples.
    dataset = dataset.shuffle(1000).repeat().batch(batch_size)

    # Return the dataset.
    return dataset


def eval_fn(features, labels, batch_size):
    """An input function for edelete files pythonvaluation and prediction"""
    features = dict(features)
    if labels is None:
        # No labels, use only features.
        inputs = features
    else:
        inputs = (features, labels)

    # Convert the inputs to a Dataset.
    dataset = tf.data.Dataset.from_tensor_slices(inputs)

    # Batch the examples
    assert batch_size is not None, "batch_size must not be None"
    dataset = dataset.batch(batch_size)

    # Return the dataset.
    return dataset


# =============================================================================
# Analysis
# =============================================================================

def acc_analysis(train_x, train_y, test_x, test_y):
    n_sample = 100
    accuracies = np.zeros(n_sample)
    for i in trange(n_sample, desc='Samples'):
        # clean out the model directory
        for filename in os.listdir(model_dir):
            filepath = os.path.join(model_dir, filename)
            try:
                if os.path.isfile(filepath):
                    os.unlink(filepath)
                elif os.path.isdir(filepath):
                    shutil.rmtree(filepath)
            except Exception as e:
                print(e)

        hidden_units = hidden_units = [n_nodes] * n_layers
        classifier = tf.estimator.DNNClassifier(feature_columns=ftr_cols,
                                                hidden_units=hidden_units,
                                                n_classes=n_classes,
                                                label_vocabulary=label_voc,
                                                model_dir=model_dir,
                                                activation_fn=tf.nn.elu)

        classifier.train(steps=np.floor(m_train * len(train_y)),
                         input_fn=lambda: train_fn(train_x, train_y, 100))

        eval_result = classifier.evaluate(input_fn=lambda: eval_fn(test_x,
                                                                   test_y,
                                                                   100))
        accuracies[i] = eval_result['accuracy']

    acc_mean = np.mean(accuracies)
    acc_std = np.std(accuracies)

    return acc_mean, acc_std, accuracies


def train_analysis(srange, train_x, train_y, test_x, test_y):
    n_sample = 5
    n_test = 20
    accuracies = np.zeros((n_test, n_sample))
    x_layers = np.zeros((n_test, n_sample), dtype=int)
    for i, mult in enumerate(tqdm(np.linspace(np.min(srange), np.max(srange),
                                              num=n_test),
                                  desc='Sample Set Size')):
        for j in trange(n_sample, desc='Samples'):
            # clean out the model directory
            for filename in os.listdir(model_dir):
                filepath = os.path.join(model_dir, filename)
                try:
                    if os.path.isfile(filepath):
                        os.unlink(filepath)
                    elif os.path.isdir(filepath):
                        shutil.rmtree(filepath)
                except Exception as e:
                    print(e)

            hidden_units = hidden_units = [n_nodes] * n_layers
            classifier = tf.estimator.DNNClassifier(feature_columns=ftr_cols,
                                                    hidden_units=hidden_units,
                                                    n_classes=n_classes,
                                                    label_vocabulary=label_voc,
                                                    model_dir=model_dir,
                                                    activation_fn=tf.nn.elu)

            classifier.train(steps=np.floor(mult * len(train_y)),
                             input_fn=lambda: train_fn(train_x, train_y, 100))

            eval_result = classifier.evaluate(input_fn=lambda: eval_fn(test_x,
                                                                       test_y,
                                                                       100))
            accuracies[i, j] = eval_result['accuracy']
            x_layers[i, j] = np.floor(mult * len(train_y))

    acc_mean = np.mean(accuracies, axis=1)
    acc_std = np.std(accuracies, axis=1)

    tr_x = x_layers[:, 0]
    x_mult = np.linspace(np.min(srange), np.max(srange), num=n_test)

    popt_tr, pcov_tr = curve_fit(sigmoid, x_mult, acc_mean, maxfev=8000)
    xlin = np.linspace(np.min(srange), np.max(srange), num=1000)
    yfit = sigmoid(xlin, *popt_tr)

    plt.figure(figsize=fig_size)
    plt.scatter(x_layers, accuracies, marker='o', color='k', alpha=0.5,
                label='accuracies', zorder=0)
    plt.plot(tr_x, acc_mean, marker='x', color='r',
             linewidth=1, label='mean accuracy')
    plt.errorbar(tr_x, acc_mean, color='r', yerr=acc_std,
                 linewidth=1, label='$\pm 1 \sigma_{n}$', capsize=2)
    plt.plot(xlin * len(train_y), yfit, 'b--',
             linewidth=1, label='Accuracy Model')

    plt.legend()
    plt.xlabel('Training Set Size')
    plt.ylabel('Accuracy')
    plt.savefig('train_accuracy.pdf')

    sig = sigmoid(x_mult, *popt_tr)

    grad = np.gradient(sig)

    return x_mult[np.argmax(grad < 1e-4)]


def neuron_analysis(n_neurons, train_x, train_y, test_x, test_y):
    n_pts = 5
    accuracies = np.zeros((n_neurons, n_pts))
    x_layers = np.zeros((n_neurons, n_pts), dtype=int)
    for neurons in trange(n_neurons, desc='Neurons'):
        for i in trange(n_pts, desc='Points'):
            # clean out the model directory
            for filename in os.listdir(model_dir):
                filepath = os.path.join(model_dir, filename)
                try:
                    if os.path.isfile(filepath):
                        os.unlink(filepath)
                    elif os.path.isdir(filepath):
                        shutil.rmtree(filepath)
                except Exception as e:
                    print(e)
            hidden_units = [neurons + 1] * n_layers
            classifier = tf.estimator.DNNClassifier(feature_columns=ftr_cols,
                                                    hidden_units=hidden_units,
                                                    n_classes=n_classes,
                                                    label_vocabulary=label_voc,
                                                    model_dir=model_dir,
                                                    activation_fn=tf.nn.elu)

            classifier.train(steps=np.floor(m_train * len(train_y)),
                             input_fn=lambda: train_fn(train_x, train_y, 100))

            eval_result = classifier.evaluate(input_fn=lambda: eval_fn(test_x,
                                                                       test_y,
                                                                       100))
            accuracies[neurons, i] = eval_result['accuracy']
            x_layers[neurons, i] = int(neurons + 1)

    acc_mean = np.mean(accuracies, axis=1)
    acc_std = np.std(accuracies, axis=1)

    nr_x = x_layers[:, 0]

    popt_nr, pcov_nr = curve_fit(sigmoid, nr_x, acc_mean)
    xlin = np.linspace(1, n_neurons, num=1000)
    yfit = sigmoid(xlin, *popt_nr)

    plt.figure(figsize=fig_size)
    plt.scatter(x_layers, accuracies, marker='o', color='k', alpha=0.5,
                label='accuracies', zorder=0)
    plt.scatter(nr_x, acc_mean, marker='x', color='r',
                linewidth=1, label='mean accuracy')
    plt.errorbar(nr_x, acc_mean, color='r', yerr=acc_std,
                 linewidth=1, label='$\pm 1 \sigma_{n}$', capsize=2)
    plt.plot(xlin, yfit, 'b--', linewidth=1, label='Accuracy Model')

    plt.legend()
    plt.xlabel('Nodes')
    plt.ylabel('Accuracy')
    plt.savefig('neuron_accuracy.pdf')

    sig = sigmoid(nr_x, *popt_nr)

    grad = np.gradient(sig)

    return int(nr_x[np.argmax(grad < 1e-4)])


def layer_analysis(n_layers, train_x, train_y, test_x, test_y):
    n_pts = 5
    accuracies = np.zeros((n_layers + 1, n_pts))
    x_layers = np.zeros((n_layers + 1, n_pts), dtype=int)
    for layers in trange(n_layers+1, desc='Layers'):
        for i in trange(n_pts, desc='Points'):
            # clean out the model directory
            for filename in os.listdir(model_dir):
                filepath = os.path.join(model_dir, filename)
                try:
                    if os.path.isfile(filepath):
                        os.unlink(filepath)
                    elif os.path.isdir(filepath):
                        shutil.rmtree(filepath)
                except Exception as e:
                    print(e)
            hidden_units = [n_nodes] * layers
            classifier = tf.estimator.DNNClassifier(feature_columns=ftr_cols,
                                                    hidden_units=hidden_units,
                                                    n_classes=n_classes,
                                                    label_vocabulary=label_voc,
                                                    model_dir=model_dir,
                                                    activation_fn=tf.nn.elu)

            classifier.train(steps=np.floor(m_train * len(train_y)),
                             input_fn=lambda: train_fn(train_x, train_y, 100))

            eval_result = classifier.evaluate(input_fn=lambda: eval_fn(test_x,
                                                                       test_y,
                                                                       100))
            accuracies[layers, i] = eval_result['accuracy']
            x_layers[layers, i] = int(layers)

    acc_mean = np.mean(accuracies, axis=1)
    acc_std = np.std(accuracies, axis=1)

    lyr_x = x_layers[:, 0]

    popt_lyr, pcov_lyr = curve_fit(sigmoid, lyr_x, acc_mean, maxfev=8000,
                                   bounds=([-np.inf, -np.inf, 0.],
                                           [np.inf, np.inf, 1.]))
    xlin = np.linspace(0, n_layers, num=1000)
    yfit = sigmoid(xlin, *popt_lyr)

    plt.figure(figsize=fig_size)
    plt.scatter(x_layers, accuracies, marker='o', color='k', alpha=0.5,
                label='accuracies', zorder=0)
    plt.scatter(lyr_x, acc_mean, marker='x', color='r',
                linewidth=1, label='mean accuracy')
    plt.errorbar(lyr_x, acc_mean, color='r', yerr=acc_std,
                 linewidth=1, label='$\pm 1 \sigma_{n}$', capsize=2)
    plt.plot(xlin, yfit, 'b--', linewidth=1, label='Accuracy Model')

    plt.legend()
    plt.xlabel('Number of Hidden Layers')
    plt.ylabel('Accuracy')
    plt.savefig('layer_accuracy.pdf')

    sig = sigmoid(lyr_x, *popt_lyr)

    grad = np.gradient(sig)

    return int(lyr_x[np.argmax(grad < 1e-4)])


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
    label_voc = np.unique(labels).tolist()

    train_x, test_x, train_y, test_y = train_test_split(features,
                                                        labels,
                                                        test_size=.2,
                                                        random_state=0,
                                                        stratify=labels)

    train_x = dict(zip(['u-g', 'g-r', 'r-i', 'i-z'], train_x.T))
    pred_x = dict(zip(['u-g', 'g-r', 'r-i', 'i-z'], test_x[:5].T))
    test_x = dict(zip(['u-g', 'g-r', 'r-i', 'i-z'], test_x.T))

    ftr_cols = []
    for key in train_x.keys():
        ftr_cols.append(tf.feature_column.numeric_column(key=key))

    n_classes = len(np.unique(labels))

    lyr_opt = layer_analysis(10, train_x, train_y, test_x, test_y)
    n_layers = lyr_opt

    nr_opt = neuron_analysis(70, train_x, train_y, test_x, test_y)
    n_nodes = nr_opt

    tr_opt = train_analysis([0.05, 3.], train_x, train_y, test_x, test_y)
    m_train = tr_opt

    acc_mean, acc_std, accuracies = acc_analysis(train_x, train_y,
                                                 test_x, test_y)

#    pred_result = classifier.predict(
#            input_fn=lambda: eval_fn(pred_x, labels=None, batch_size=100))
#
#    for pred_dict, expec in zip(pred_result, test_y[:5]):
#        template = ('\nPrediction is "{}" ({:.1f}%), expected "{}"')
#
#        class_id = pred_dict['class_ids'][0]
#        probability = pred_dict['probabilities'][class_id]
#
#        print(template.format(label_voculary[class_id],
#                              100 * probability, expec))
