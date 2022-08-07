import csv
import os
import numpy as np
import pandas as pd
import torch

from sklearn.linear_model import RidgeClassifier
from sklearn.feature_selection import RFE
from scipy.spatial import distance
from scipy.special import softmax
from sklearn.metrics import roc_auc_score
from sklearn import metrics
from sklearn.metrics import confusion_matrix


def calculate_metric(gt, pred):
    confusion = confusion_matrix(gt, pred)
    TP = confusion[1, 1]
    TN = confusion[0, 0]
    FP = confusion[0, 1]
    FN = confusion[1, 0]
    Sen = TP / float(TP+FN)
    Specificity = TN / float(TN+FP)
    return Sen, Specificity

def get_subject_score(score, phenotype_path):
    scores_dict = {}
    with open(phenotype_path) as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
                scores_dict[row['subjectIdentifier']] = row[score]

    return scores_dict


def feature_selection(matrix, labels, train_ind, fnum):
    """
        matrix       : feature matrix (num_subjects x num_features)
        labels       : ground truth labels (num_subjects x 1)
        fnum         : size of the feature vector after feature selection

    return:
        x_data      : feature matrix of lower dimension (num_subjects x fnum)
    """

    estimator = RidgeClassifier()
    selector = RFE(estimator=estimator, n_features_to_select=fnum, step=100, verbose=0)

    featureX = matrix[train_ind, :]
    featureY = labels[train_ind]
    selector = selector.fit(featureX, featureY.ravel())
    x_data = selector.transform(matrix)
    # x_data = matrix

    # print("Number of labeled samples %d" % len(train_ind))
    print("Number of features selected %d" % x_data.shape[1])

    return x_data


def create_affinity_graph_from_scores(scores, pd_dict):
    num_nodes = len(pd_dict[scores[0]])
    graph = np.zeros((num_nodes, num_nodes))

    for l in scores:
        label_dict = pd_dict[l]
        if l == 'AGE':
            for k in range(num_nodes):
                for j in range(k + 1, num_nodes):
                    try:
                        val = abs(float(label_dict[k]) - float(label_dict[j]))
                        if val < 5:
                            graph[k, j] += 1
                            graph[j, k] += 1
                    except ValueError:  # missing label
                        pass
        if l == 'MMSE':
            for k in range(num_nodes):
                for j in range(k + 1, num_nodes):
                    try:
                        val = abs(float(label_dict[k]) - float(label_dict[j]))
                        if val < 2:
                            graph[k, j] += 1
                            graph[j, k] += 1
                    except ValueError:  # missing label
                        pass
        if l == 'PTGENDER':
            for k in range(num_nodes):
                for j in range(k + 1, num_nodes):
                    if label_dict[k] == label_dict[j]:
                        graph[k, j] += 1
                        graph[j, k] += 1

    return graph


def get_static_affinity_adj(features, pd_dict):
    pd_affinity = create_affinity_graph_from_scores(['PTGENDER', 'AGE', 'MMSE'], pd_dict)
    distv = distance.pdist(features, metric='correlation')
    dist = distance.squareform(distv)
    sigma = np.mean(dist)
    feature_sim = np.exp(- dist ** 2 / (2 * sigma ** 2))
    adj = pd_affinity * feature_sim
    return adj


def write_raw_score(f, preds, labels):
    for index, pred in enumerate(preds):
        label = str(labels[index])
        pred = "__".join(map(str, list(pred)))
        f.write(pred + '__' + label + '\n')


def get_confusion_matrix(preds, labels):
    matrix = [[0, 0], [0, 0]]
    for index, pred in enumerate(preds):
        if np.amax(pred) == pred[0]:
            if labels[index] == 0:
                matrix[0][0] += 1
            if labels[index] == 1:
                matrix[0][1] += 1
        elif np.amax(pred) == pred[1]:
            if labels[index] == 0:
                matrix[1][0] += 1
            if labels[index] == 1:
                matrix[1][1] += 1
    return matrix


def matrix_sum(A, B):
    return [[A[0][0] + B[0][0], A[0][1] + B[0][1]],
            [A[1][0] + B[1][0], A[1][1] + B[1][1]]]


def get_acc(matrix):
    return float(matrix[0][0] + matrix[1][1]) / float(sum(matrix[0]) + sum(matrix[1]))


def get_sen(matrix):
    return float(matrix[1][1]) / float(matrix[0][1] + matrix[1][1])


def get_spe(matrix):
    return float(matrix[0][0]) / float(matrix[0][0] + matrix[1][0])


def accuracy(preds, labels):
    """
    Accuracy, auc with masking. Acc of the masked samples
    """
    correct_prediction = np.equal(np.argmax(preds, 1), labels).astype(np.float32)
    return np.sum(correct_prediction), np.mean(correct_prediction)


def auc(preds, labels, is_logit=True):
    '''
    input: logits, labels
    '''
    if is_logit:
        pos_probs = softmax(preds, axis=1)[:, 1]
    else:
        pos_probs = preds[:,1]
    try:
        auc_out = roc_auc_score(labels, pos_probs)
    except:
        auc_out = 0
    return auc_out
