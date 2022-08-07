import numpy as np
import torch
from sklearn.metrics import roc_curve, auc, roc_auc_score
from utils import *
import matplotlib.pyplot as plt
from metrics import roc_auc_score
from scipy.special import softmax
from sklearn import metrics as me


def get_confusion_matrix(preds, labels):
    matrix = [[0, 0], [0, 0]]
    for index, pred in enumerate(preds):
        if pred == 0:
            if labels[index] == 0:
                matrix[0][0] += 1
            if labels[index] == 1:
                matrix[0][1] += 1
        elif pred == 1:
            if labels[index] == 0:
                matrix[1][0] += 1
            if labels[index] == 1:
                matrix[1][1] += 1
    return matrix


def matrix_sum(A, B):
    return [[A[0][0]+B[0][0], A[0][1]+B[0][1]],
            [A[1][0]+B[1][0], A[1][1]+B[1][1]]]


def first_stage(fea_name_list, n_sample_list, task, modal, threshold):

    for i in range(10):
        new_preds = [0] * n_sample_list[i]
        labels = []
        prods = np.zeros((n_sample_list[i], 2))
        for k, fea_name in enumerate(fea_name_list):
            pred_labels = []
            tmp_all = []

            with open('save_models/{}/{}/fold{}_score_test.txt'.format(task, fea_name, i), 'r') as f:
                file = f.readlines()
                for j in range(len(file)):
                    data = file[j].split("__")
                    tmp = [data[0], data[1]]
                    tmp = np.array(tmp).astype(np.float32)

                    if tmp[0] > tmp[1]:
                        pred_label = 0
                    else:
                        pred_label = 1

                    pred_labels.append(pred_label)

                    prods[j, 0] += float(data[0])
                    prods[j, 1] += float(data[1])

                    if k == 0:
                        label = data[2].split()[0]
                        labels.append(round(float(label)))
            f.close()
            for index, pred in enumerate(pred_labels):
                new_preds[index] += pred

        f1 = open('save_models/{}/2StageVoting/first_stage/{}_fold{}_score_test.txt'.format(task, modal, i), 'w')
        for index, new_pred in enumerate(new_preds):
            label = str(labels[index])
            if new_pred >= threshold:
                pred_label = 1
                weight = str(new_pred)
            else:
                pred_label = 0
                weight = str(len(fea_name_list) - new_pred)
            pred_label = str(pred_label)
            f1.write(pred_label + '__' + label + '__' + weight + '\n')


def sec_stage(path, task):

    n_fold = 10
    weights_all = []
    labels_all = []
    predMRI = []
    predSNP = []
    for i in range(n_fold):
        labels = []
        preds_MRI = []
        preds_SNP = []
        weights_MRI = []
        weights_SNP = []

        with open(path + 'MRI_fold{}_score_test.txt'.format(i), 'r') as f:
            file = f.readlines()
            for j in range(len(file)):
                data = file[j].split("__")
                preds_MRI.append(int(data[0]))
                label = data[1]
                labels.append(round(float(label)))
                labels_all.append(round(float(label)))
                weights_MRI.append(data[2].split()[0])
        f.close()

        with open(path + 'SNP_fold{}_score_test.txt'.format(i), 'r') as f:
            file = f.readlines()
            for j in range(len(file)):
                data = file[j].split("__")
                preds_SNP.append(int(data[0]))
                weights_SNP.append(data[2].split()[0])
        f.close()

        f = open('save_models/{}/2StageVoting/sec_stage/fold{}_score_test.txt'.format(task, i), 'w')
        for index, pred_MRI in enumerate(preds_MRI):
            pred_SNP = preds_SNP[index]
            predMRI.append(pred_MRI)
            predSNP.append(pred_SNP)
            weight_MRI = float(weights_MRI[index])
            weight_SNP = float(weights_SNP[index])

            if pred_MRI == pred_SNP:
                pred_label = pred_MRI
                weights_all.append(weight_SNP)
            else:
                if weight_SNP > weight_MRI:
                    pred_label = pred_SNP
                    weights_all.append(weight_SNP)
                elif weight_MRI > weight_SNP:
                    pred_label = pred_MRI
                    weights_all.append(weight_MRI)
                else:
                    # MCI vs. HC
                    weights_all.append(weight_SNP)
                    pred_label = pred_SNP
                    # MCI vs. AD
                    # pred_label = pred_MRI

            label = str(labels[index])
            pred_label = str(pred_label)
            f.write(pred_label + '__' + label + '\n')
        f.close()
    print(predMRI)
    print(predSNP)
    return weights_all, labels_all


def metrics(path, task):
    n_fold = 10
    matrix = [[0, 0], [0, 0]]
    preds = []
    labels = []
    for i in range(n_fold):

        with open(path + 'fold{}_score_test.txt'.format(i), 'r') as f:
            file = f.readlines()
            for j in range(len(file)):
                data = file[j].split("__")
                tmp = int(data[0])
                preds.append(tmp)
                label = data[1].split()[0]
                labels.append(round(float(label)))
        f.close()
    matrix = matrix_sum(matrix, get_confusion_matrix(preds, labels))
    acc = (matrix[0][0] + matrix[1][1]) / (matrix[0][0] + matrix[0][1] + matrix[1][0] + matrix[1][1])
    sen = matrix[1][1] / (matrix[0][1] + matrix[1][1])  
    spe = matrix[0][0] / (matrix[0][0] + matrix[1][0])
    PPV = (matrix[0][0]) / (matrix[0][0] + matrix[0][1])
    f1 = 2 / (1 / spe + 1 / PPV)
    auc1 = roc_auc_score(preds, labels)
    print(matrix)
    print("acc: {:.4}%".format(acc*100))
    print("sen: {:.4}%".format(sen*100))
    print("spe: {:.4}%".format(spe*100))
    print("F1: {:.4}%".format(f1*100))
    print("auc: {:.4}".format(auc1))


if __name__ == '__main__':
    task = "MCI_CN"
    fea_name_list1 = ["MRI_CorticalThickness", "MRI_SurfaceArea", "MRI_Volume"]
    fea_name_list2 = ["SNP_APOE", "SNP_BIN1", "SNP_CLU"]
    n_sample_list = [19, 19, 19, 19, 19, 19, 19, 19, 18, 18]
    

    first_stage(fea_name_list1, n_sample_list, task, modal="MRI", threshold=2)
    first_stage(fea_name_list2, n_sample_list, task, modal="SNP", threshold=2)
    fold_path = 'save_models/{}/2StageVoting/'.format(task)
    path = fold_path + "first_stage/"
    metrics(path+"MRI_", task)
    metrics(path + "SNP_", task)

    weights, labels = sec_stage(path, task)
    path = fold_path + "sec_stage/"
    metrics(path, task)
