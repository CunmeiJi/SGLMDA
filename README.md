# SGLMDA

## Introduction

In this repo, we implement the paper "SGLMDA: A subgraph learning-based method for miRNA-disease association prediction". SGLMDA samples $K$-hop subgraphs from the global heterogeneous miRNA-disease graph, extracts subgraph representation based on Graph Neural Networks (GNNs). Extensive experiments conducted on benchmark datasets demonstrate that SGLMDA can effectively and robustly predict potential miRNA-disease associations. 

## Requirements

- PyTorch >=1.6 
- Deep Graph Libray (DGL) >=0.7.2
- CUDA >=10.1
- GPU (default).

## Data

- miRBase, see http://www.mirbase.org/
- Mesh, see https://www.ncbi.nlm.nih.gov/mesh
- HMDD, see https://www.cuilab.cn/hmdd

We have included preprocessed data in the "data" folder, comprising two sub-folders: HMDD2 and HMDD3.2. Each folder contains information on disease names, miRNA names, miRNA sequencing, and miRNA-disease associations. To calculate similarities of miRNA/disease, you can perform the calculations independently using the provided datasets.

## Usage

1. Download the code and data.

2. For case study,  such as  colon neoplasms, you can execute the python notebook at src/case_study.ipynb

3. For 5-fold cross-validation (5cv), follow these steps 

   1.  Divide the data into five parts, referring to :https://git.l3s.uni-hannover.de/dong/simplifying_mirna_disease.

   2. Run the evaluation test

      ```
      python eval_irgat.py
      ```
