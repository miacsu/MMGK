# MMGK
Code for "Multi-modality Multi-view Graph Representations and Knowledge Embedding for Mild Cognitive Impairmen"

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
