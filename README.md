# DPSBA
The source code for "Stealthy Yet Effective: Distribution-Preserving Backdoor Attacks on Graph Classification" at NeurIPS 2025.

Part of code is referenced from paper **[Graph Backdoor](https://arxiv.org/abs/2006.11890)**.


## Datasets
You can download the datasets from **[TUDataset](https://chrsmrrs.github.io/datasets/docs/home/)**

## Requirements
- Python >= 3.9
- dgl = 2.2.1
- numpy =  1.26.4
- torch = 2.1.2+cu121
- torch-geometric = 2.6.1
- tqdm = 4.64.1
## Run
```
cd main
python attackk22.py --use_org_node_attr --save_clean_model --save_bkd_model
```
## Cite
Please cite our paper if it is helpful in your own work.
