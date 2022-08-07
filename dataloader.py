import os
from utils import *
import numpy as np
import pandas as pd
from gcn_utils import preprocess_features
from sklearn.model_selection import StratifiedKFold

num_classes = 2

def load_data(modal_name, data_folder, phenotype_path):
    '''
    load multimodal data from ADNI
    return: imaging features (raw), labels, non-image data
    '''
    features = np.loadtxt(os.path.join(data_folder, modal_name + ".csv"), delimiter=',')

    ID_reader = pd.read_csv(phenotype_path)
    subject_IDs = ID_reader["subjectIdentifier"]
    labels = get_subject_score(score='DX_Group', phenotype_path=phenotype_path)
    num_nodes = len(subject_IDs)

    ages = get_subject_score(score='AGE', phenotype_path=phenotype_path)
    genders = get_subject_score(score='PTGENDER', phenotype_path=phenotype_path)
    mmses = get_subject_score(score='MMSE', phenotype_path=phenotype_path)

    y_onehot = np.zeros([num_nodes, num_classes])
    y = np.zeros([num_nodes])
    age = np.zeros([num_nodes], dtype=np.float32)
    gender = np.zeros([num_nodes], dtype=np.int)
    mmse = np.zeros([num_nodes], dtype=np.int)

    for i in range(num_nodes):
        y_onehot[i, int(labels[subject_IDs[i]])] = 1
        y[i] = int(labels[subject_IDs[i]])
        age[i] = float(ages[subject_IDs[i]])
        gender[i] = genders[subject_IDs[i]]
        mmse[i] = mmses[subject_IDs[i]]
        # pteducat[i] = pteducats[subject_IDs[i]]

    y = y

    pd_dict = {}

    phonetic_data = np.zeros([num_nodes, 3], dtype=np.float32)
    phonetic_data[:, 0] = gender
    phonetic_data[:, 1] = age
    phonetic_data[:, 2] = mmse

    pd_dict['PTGENDER'] = np.copy(phonetic_data[:, 0])
    pd_dict['AGE'] = np.copy(phonetic_data[:, 1])
    pd_dict['MMSE'] = np.copy(phonetic_data[:, 2])

    return features, y, phonetic_data, pd_dict


def data_split(features, y, n_folds):
    # split data by k-fold CV
    skf = StratifiedKFold(n_splits=n_folds)
    cv_splits = list(skf.split(features, y))
    return cv_splits


def get_node_features(features, y, train_ind, node_ftr_dim=None, feature_select=False):
    '''
    preprocess node features for GCN
    '''
    if feature_select:
        node_ftr = feature_selection(features, y, train_ind, node_ftr_dim)
        node_ftr = preprocess_features(node_ftr)
    else:
        node_ftr = preprocess_features(features)
    return node_ftr


def get_aff_adj(features, pd_dict):
    n = features.shape[0]
    num_edge = n*(1+n)//2 - n
    edge_index = np.zeros([2, num_edge], dtype=np.int64)
    adj = np.zeros([num_edge, 1], dtype=np.float32)

    aff_adj = get_static_affinity_adj(features, pd_dict)

    flatten_ind = 0
    for i in range(n):
        for j in range(i+1, n):
            edge_index[:, flatten_ind] = [i, j]
            adj[flatten_ind] = aff_adj[i][j]
            flatten_ind += 1

    return edge_index, adj


def get_data(features, subject_IDs, ages, genders, labels):
    hc_nodes = len(np.where(labels == 0))
    mci_nodes = len(np.where(labels == 1))

    features_hc = np.zeros([hc_nodes], dtype=np.float32)
    y_hc = np.zeros([hc_nodes])
    age_hc = np.zeros([hc_nodes], dtype=np.float32)
    gender_hc = np.zeros([hc_nodes], dtype=np.int)
    features_mci = np.zeros([mci_nodes], dtype=np.float32)
    y_mci = np.zeros([mci_nodes])
    age_mci = np.zeros([mci_nodes], dtype=np.float32)
    gender_mci = np.zeros([mci_nodes], dtype=np.int)

    num_nodes = len(subject_IDs)
    j = 0
    k = 0
    for i in range(num_nodes):
        if labels[subject_IDs[i]] == 0:
            features_hc[j] = features[i]
            y_hc[j] = int(labels[subject_IDs[i]])
            age_hc[j] = float(ages[subject_IDs[i]])
            gender_hc[j] = genders[subject_IDs[i]]
            j = j+1
        if labels[subject_IDs[i]] == 0:
            features_mci[k] = features[i]
            y_mci[k] = int(labels[subject_IDs[i]])
            age_mci[k] = float(ages[subject_IDs[i]])
            gender_mci[k] = genders[subject_IDs[i]]
            k = k+1

    i = 0
    while(i<num_nodes):
        features[i:i+5] = features_mci[j:j+5]



