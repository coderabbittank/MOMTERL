# MOMTERL
 
---

## Environment Setup

To set up the environment, you need to install the following dependencies:

```bash
python
rdkit
scipy
torch
torch-geometric
torch-sparse
tqdm
networkx
```

## dataset
All the necessary data files can be downloaded from the following links.

For the chemistry dataset, download from [chem data](https://snap.stanford.edu/gnn-pretrain/data/chem_dataset.zip) (2.5GB), unzip it, and put it under dataset/.

## train
You can pretrain the MOMTERL model by executing the following steps:
```bash
# Create a directory to save the pretrained model
mkdir saved_model

# Run the pretraining script
python pretrain.py
```

## eval
In the ```bash finetune/standard ``` directory,excute:
```bash
python finetune.py
```


