# MMGK
Code for "Multi-modality Multi-view Graph Representations and Knowledge Embedding for Mild Cognitive Impairmen"

## Description
![image](https://github.com/miacsu/MMGK/blob/main/images/framework.png)
Fig. 1. Framework of our proposed method

As is shown in Fig.1. , we propose a multi-modality multi-view graph representations and knowledge embedding (MMGK) framework to predict MCI. Firstly, to obtain the rich information in multi-modality data, we extract multi-view feature representations from magnetic resonance imaging (MRI) and genetic data. Afterwards, considering the correlations between subjects, all subjects are constructed into a graph based on the different single-view feature representations, respectively. Moreover, since knowledge is not considered in the previous graph, demographic data is added through knowledge embedding to enrich the correlation information. Then, to perform convolutional operations on graph, we utilize graph convolutional network (GCN) for feature extraction. Finally, to obtain ensemble information from multi-modality data, a two-step ensemble learning is proposed to perform MCI diagnosis.

## Requirements
python==3.6 

torch-geometric==2.0.3

torchvision==0.11.1 

Dependency packages can be installed using following command:

pip install -r requirements.txt

## Dateset

We use the data from the public Alzheimer's Disease Neuroimaging Initiative (ADNI) database with MRI and genetic data (http://adni.loni.usc.edu/).

### Data preprocess
MRI: FreeSurfer with "recon-all" command( https://surfer.nmr.mgh.harvard.edu/fswiki/recon-all).

Genetic data: SNP data based on Illumina Omni 2.5m (WGS Platform) by genotyping of genetic data is utilized from http://adni.loni.usc.edu/data-samples/data-types/.

## Run the code

### 1.Train
python train_eval_PGCN.py --train=1

### 2.Test
python train_eval_PGCN.py --train=0

### 3.Get the MLP result
python metrics.py

### 4.Two_stage_Ensemble_Learning
python 2StageVoting.py
