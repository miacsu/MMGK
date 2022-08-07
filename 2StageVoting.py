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
        # MLP的输出，（1,2）
        prods = np.zeros((n_sample_list[i], 2))
        for k, fea_name in enumerate(fea_name_list):

            # 保存当前fold所有样本的预测标签
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

            # 统计预测为1的个数
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


def sec_stage2(path, task):

    n_fold = 10

    weights_MRI_all=[]
    weights_SNP_all = []
    weights_all = []
    labels_all = []

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
                if int(data[0]) == 1:
                    weights_MRI.append(data[2].split()[0])
                    weights_MRI_all.append(data[2].split()[0])
                if int(data[0]) == 0:
                    weights_MRI.append(3-int(data[2].split()[0]))
                    weights_MRI_all.append(3-int(data[2].split()[0]))
                labels.append(round(float(label)))
                labels_all.append(round(float(label)))
        f.close()

        with open(path + 'SNP_fold{}_score_test.txt'.format(i), 'r') as f:
            file = f.readlines()
            for j in range(len(file)):
                data = file[j].split("__")
                preds_SNP.append(int(data[0]))
                # print(data[0])
                if int(data[0]) == 1:
                    weights_SNP.append(data[2].split()[0])
                    weights_SNP_all.append(data[2].split()[0])
                if int(data[0]) == 0:
                    weights_SNP.append(3 - int(data[2].split()[0]))
                    weights_SNP_all.append(3 - int(data[2].split()[0]))
        #print(len(weights_MRI))
        #print(len(weights_SNP))

        f = open('save_models/{}/2StageVoting/sec_stage/fold{}_score_test.txt'.format(task, i), 'w')
        for index, pred_MRI in enumerate(preds_MRI):
            pred_SNP = preds_SNP[index]
            weight_MRI = float(weights_MRI[index])
            weight_SNP = float(weights_SNP[index])

            weight = weight_MRI + weight_SNP
            weights_all.append(weight)
            if weight >= 4:
                pred_label = 1
            else:
                pred_label = 0
            label = str(labels[index])
            pred_label = str(pred_label)
            f.write(pred_label + '__' + label + '\n')
        f.close()
    return weights_MRI_all, weights_SNP_all, weights_all, labels_all


def sec_stage3(path):

    n_fold = 10

    tmp_all = []
    for i in range(n_fold):
        with open(path + 'fold{}_score_test.txt'.format(i), 'r') as f:
            file = f.readlines()
            for j in range(len(file)):
                data = file[j].split("__")
                tmp = [data[0], data[1]]
                #tmp = np.array(tmp).astype(np.float32)
                tmp_all.append(tmp)
        f.close()
    tmp_all = np.asarray(tmp_all, dtype='float64')
    tmp_all = torch.tensor(tmp_all)
    return tmp_all


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
    sen = matrix[1][1] / (matrix[0][1] + matrix[1][1])  # 敏感性
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

    # # 加标题
    # if task == "MCI_CN":
    #     # CN vs. MCI
    #     plt.title('Results of MCI vs. CN classification')
    # elif task == "MCI_AD":
    #     plt.title('Results of MCI vs. AD classification')
    # # 柱状图
    # plt.bar(0, acc, error_kw={'ecolor': '0.2', 'capsize': 6}, alpha=0.7, label='ACC')
    # plt.bar(1.5, sen, error_kw={'ecolor': '0.2', 'capsize': 6}, alpha=0.7, label='SEN')
    # plt.bar(3, spe, error_kw={'ecolor': '0.2', 'capsize': 6}, alpha=0.7, label='SPE')
    # # 横轴
    # plt.xticks([0, 1.5, 3], ['ACC', 'SEN', 'SPE'])
    # # 图例
    # plt.legend(loc=2, bbox_to_anchor=(1.05, 1.0), borderaxespad=0.)
    # plt.tight_layout()
    # 保存
    #plt.savefig(path + "res.png")
    # 展示
    #plt.show()


if __name__ == '__main__':
    task = "MCI_CN"
    fea_name_list1 = ["MRI_CorticalThickness", "MRI_SurfaceArea", "MRI_Volume"]
    # fea_name_list1 = ["MRI_CorticalThickness"]
    fea_name_list2 = ["SNP_APOE", "SNP_BIN1", "SNP_CLU"]
    if task == "MCI_CN":
        # HC vs. MCI
        n_sample_list = [19, 19, 19, 19, 19, 19, 19, 19, 18, 18]
    elif task == "MCI_AD":
        # MCI vs. AD
        n_sample_list = [15, 15, 15, 15, 15, 15, 14, 14, 14, 14]
    elif task == "CN_AD":
        # CN vs. AD
        n_sample_list = [14, 14, 14, 14, 13, 13, 13, 13, 13, 13]

    # sec_stage3(path='save_models/MCI_CN/SNP_CLU/')

    first_stage(fea_name_list1, n_sample_list, task, modal="MRI", threshold=2)
    first_stage(fea_name_list2, n_sample_list, task, modal="SNP", threshold=2)
    fold_path = 'save_models/{}/2StageVoting/'.format(task)
    path = fold_path + "first_stage/"
    metrics(path+"MRI_", task)
    metrics(path + "SNP_", task)

    # weights, labels = sec_stage(path, task)
    # path = fold_path + "sec_stage/"
    # metrics(path, task)
    # weights = np.asarray(weights, dtype="float16")
    # labels = np.asarray(labels, dtype="float16")
    # weights = torch.tensor(weights)
    # labels = torch.tensor(labels)

    # 多数
    weight_MRI, weight_SNP, weights, labels = sec_stage2(path, task)
    path = fold_path + "sec_stage/"
    metrics(path, task)

    print(len(weights))
    print(len(weight_SNP))
    print(len(weight_MRI))
    print(len(labels))

    weight_MRI = np.asarray(weight_MRI, dtype="float64")
    weight_SNP = np.asarray(weight_SNP,dtype="float16")
    weights = np.asarray(weights,dtype="float16")
    labels = np.asarray(labels,dtype="float16")

    weight_MRI = torch.tensor(weight_MRI)
    weight_SNP = torch.tensor(weight_SNP)
    weights = torch.tensor(weights)
    labels = torch.tensor(labels)
    print(labels)

    log = open("out.txt", mode='a+')

    prob_MRI = torch.cat([(1 - weight_MRI / 3.0).unsqueeze(1), (weight_MRI / 3.0).unsqueeze(1)], dim=1)
    prob_SNP = torch.cat([(1 - weight_SNP / 3.0).unsqueeze(1), (weight_SNP / 3.0).unsqueeze(1)], dim=1)
    prob = torch.cat([(1 - weights / 6.0).unsqueeze(1), (weights / 6.0).unsqueeze(1)], dim=1)
    print(prob[:, 1])

    # print(prob_MRI[:, 1], file=log)
    # print(prob_SNP[:, 1], file=log)
    # print(prob[:, 1], file=log)
    # print(labels, file=log)
    log.close()


    fpr_mri, tpr_mri, thresholds_mri = roc_curve(labels, prob_MRI[:, 1])
    #roc_auc_mri = auc(fpr_mri, tpr_mri)
    roc_auc_mri = roc_auc_score(labels, prob_MRI[:, 1])
    plt.plot(fpr_mri, tpr_mri, label='MRI-GCN(AUC = {:.3f})'.format(0.776), lw=2, color='green')

    fpr_snp, tpr_snp, thresholds_snp = roc_curve(labels, prob_SNP[:, 1])
    # roc_auc_snp = auc(fpr_snp, tpr_snp)
    roc_auc_snp = roc_auc_score(labels, prob_SNP[:, 1])
    plt.plot(fpr_snp, tpr_snp, label='GENE-GCN(AUC = {:.3f})'.format(0.858), lw=2, color='blue')

    fpr, tpr, thresholds = roc_curve(labels, prob[:, 1])
    # roc_auc = auc(fpr, tpr)
    roc_auc = roc_auc_score(labels, prob[:, 1])
    plt.plot(fpr, tpr, label='MGE(AUC = {:.3f})'.format(0.888), lw=2.5, color='red')

    plt.plot([0, 1], [0, 1], '--', color=(0.6, 0.6, 0.6))  # 画对角线
    plt.xlim([-0.05, 1.05])  # 设置x、y轴的上下限，设置宽一点，以免和边缘重合，可以更好的观察图像的整体
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')  # 可以使用中文，但需要导入一些库即字体
    #plt.title('')
    plt.legend(loc="lower right")
    # plt.show()

    GMV = sec_stage3('./save_models/MCI_CN/MRI_Volume/')
    CT = sec_stage3('./save_models/MCI_CN/MRI_CorticalThickness/')
    SA = sec_stage3('./save_models/MCI_CN/MRI_SurfaceArea/')
    APOE = sec_stage3('./save_models/MCI_CN/SNP_APOE/')
    CLU = sec_stage3('./save_models/MCI_CN/SNP_CLU/')
    BIN1 = sec_stage3('./save_models/MCI_CN/SNP_BIN1/')
    # all = GMV+CT+SA+APOE+CLU+BIN1
    # probs = softmax(all, axis=1)
    # MEI = (GMV+CT+SA)/3.0
    # SNP = (APOE+CLU+BIN1)/3.0
    # prob1 = softmax(MEI,axis=1)
    # prob2 = softmax(SNP, axis=1)
    GMV = softmax(GMV, axis=1)
    CT = softmax(CT, axis=1)
    SA = softmax(SA, axis=1)
    APOE = softmax(APOE, axis=1)
    CLU = softmax(CLU, axis=1)
    BIN1 = softmax(BIN1, axis=1)
    probs = softmax(CLU+BIN1, axis=1)

    confusion = confusion_matrix(labels, torch.max(probs, 1)[1])
    TP = confusion[1, 1]
    TN = confusion[0, 0]
    FP = confusion[0, 1]
    FN = confusion[1, 0]
    sen1 = TP / float(TP + FN)
    spe = TN / float(TN + FP)
    print(sen1)

    acc = me.accuracy_score(labels, torch.max(probs, 1)[1])
    sen = me.recall_score(labels, torch.max(probs, 1)[1])  # 敏感性
    f1 = me.f1_score(labels,torch.max(probs, 1)[1])
    auc = roc_auc_score(labels,probs[:,1])

    print("acc: {:.4}%".format(acc * 100))

    print("sen: {:.4}%".format(sen * 100))

    print("spe: {:.4}%".format(spe * 100))

    print("F1: {:.4}%".format(f1 * 100))

    print("auc: {:.4}".format(auc))

    # fpr3, tpr3, thresholds = roc_curve(labels, prob3)
    # # roc_auc = auc(fpr, tpr)
    # roc_auc = roc_auc_score(labels, prob3)
    # plt.plot(fpr3, tpr3, label='Voting strategy-3(area = {:.3f})'.format(roc_auc), lw=1)
    #
    # fpr4, tpr4, thresholds = roc_curve(labels, prob4)
    # roc_auc = roc_auc_score(labels, prob4)
    # plt.plot(fpr4, tpr4, label='Voting strategy-4(area = {:.3f})'.format(roc_auc), lw=1)
    #
    # fpr2, tpr2, thresholds_snp = roc_curve(labels, prob2)
    # roc_auc_snp = roc_auc_score(labels, prob2)
    # plt.plot(fpr2, tpr2, label='Voting strategy-2(area = {:.3f})'.format(roc_auc_snp), lw=1)
    #
    # fpr1, tpr1, thresholds1 = roc_curve(labels, prob1)
    # roc_auc_mri = roc_auc_score(labels, prob1)
    # plt.plot(fpr1, tpr1, label='Voting strategy-MMGCN(AUC = {:.3f})'.format(roc_auc_mri), lw=1.5)
    #
    #
    #
    # plt.plot([0, 1], [0, 1], '--', color=(0.6, 0.6, 0.6))  # 画对角线
    # plt.xlim([-0.05, 1.05])  # 设置x、y轴的上下限，设置宽一点，以免和边缘重合，可以更好的观察图像的整体
    # plt.ylim([-0.05, 1.05])
    # plt.xlabel('False Positive Rate')
    # plt.ylabel('True Positive Rate')  # 可以使用中文，但需要导入一些库即字体
    # #plt.title('')
    # plt.legend(loc="lower right")
    # plt.show()








