# RXNGraphormer

[![Python 3.8](https://img.shields.io/badge/python-3.8-blue)](https://www.python.org/) [![PyTorch 1.12.1+cu113](https://img.shields.io/badge/PyTorch-1.12.1%2Bcu113-red)](https://pytorch.org/) [![PyG 2.3.1](https://img.shields.io/badge/torch__geometric-2.3.1-green)](https://pytorch-geometric.readthedocs.io/) [![RDKit](https://img.shields.io/badge/Chemoinformatics-RDKit-blueviolet)](https://www.rdkit.org/) [![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

Official implementation of *"A unified pre-trained deep learning framework for cross-task reaction performance prediction and synthesis planning"* (under review)

## 📌 Key Features

- **Unified Architecture**: Integrates graph neural networks with Transformer to capture both intra and inter-molecular information.
- **Pre-trained Model**: Chemical-aware pre-training on 13M reaction dataset
- **Cross-task Reaction Prediction**: Simultaneous handling of:
  - Yield, regioselectivity and enantioselectivity estimation (regression)
  - Forward- and retro-synthesis planning (sequence generation)
- **Performance**: RXNGraphormer achieves state-of-the-art performance on 21 of 23 metrics across eight benchmark datasets for reactivity/selectivity prediction and forward-/retro-synthesis planning.

## 🚀 Quick Start

### Installation

```bash
conda create -n rxngraphormer python=3.8
conda activate rxngraphormer
pip install -r requirements.txt
pip install localmapper==0.1.4             # to generate delta-mol graph
pip install .
```

### Download model weights and dataset

All model weights and preprocessed datasets are available via our [figshare repository](https://figshare.com/s/decc64a868ab64a93099). Please put these files in `model_path` and `dataset` folders, like the following structure:

```
RXNGraphormer
|── rxngraphormer
├── model_path
│   ├── buchwald_harwig
|   ├── C_H_func
|   ├── pretrained_classification_model
|   ├── suzuki_miyaura
│   ├── thiol_addition
|   ├── USPTO_50k
|   ├── USPTO_480k
|   └── ...
|── dataset
│   ├── 50k_with_rxn_type
│   ├── OOS
│   ├── pretrain
│   ├── USPTO_50k
│   ├── USPTO_480k
│   ├── USPTO_full
│   └── ...
|── ...
```

### Basic Usage

**Reaction performance prediction**

```python3
from rxngraphormer.eval import reaction_prediction
# reactivity prediction
bh_model_path = "./model_path/buchwald_harwig/seed0"
rxn_smiles_lst = ["CC(C)c1cc(C(C)C)c(-c2ccccc2P(C2CCCCC2)C2CCCCC2)c(C(C)C)c1.CN(C)C(=NC(C)(C)C)N(C)C.Cc1ccc(N)cc1.FC(F)(F)c1ccc(I)cc1.c1ccc(-c2ccon2)cc1>>Cc1ccc(Nc2ccc(C(F)(F)F)cc2)cc1",
                  "CC(C)c1cc(C(C)C)c(-c2ccccc2P(C2CCCCC2)C2CCCCC2)c(C(C)C)c1.CCOC(=O)c1cc(C)on1.CN(C)C(=NC(C)(C)C)N(C)C.Cc1ccc(N)cc1.FC(F)(F)c1ccc(Cl)cc1>>Cc1ccc(Nc2ccc(C(F)(F)F)cc2)cc1"]
bh_react_preds = reaction_prediction(bh_model_path,rxn_smiles_lst,task_type="reactivity")

# selectivity prediction
thiol_add_model_path = "./model_path/thiol_addition/seed0"
rxn_smiles_lst = ["COCc1cccc(-c2cc3c(c4c2OP(=O)(O)Oc2c(-c5cccc(COC)c5)cc5c(c2-4)CCCC5)CCCC3)c1.O=C(/N=C/c1ccccc1)c1ccccc1.Sc1ccccc1>>O=C(NC(Sc1ccccc1)c1ccccc1)c1ccccc1",
                  "COCc1cccc(-c2cc3ccccc3c3c2OP(=O)(O)Oc2c(-c4cccc(COC)c4)cc4ccccc4c2-3)c1.O=C(/N=C/c1cccc2ccccc12)c1ccccc1.SC1CCCCC1>>O=C(NC(SC1CCCCC1)c1cccc2ccccc12)c1ccccc1"]
thiol_add_sel_preds = reaction_prediction(thiol_add_model_path,rxn_smiles_lst,task_type="selectivity")
```

**Reaction synthesis planning prediction**

```python3
from rxngraphormer.eval import reaction_prediction

# retro-synthesis planning
uspto_50k_model_path = "./model_path/USPTO_50k"
pdt_smiles_lst = ['COC(=O)[C@H](CCCCN)NC(=O)Nc1cc(OC)cc(C(C)(C)C)c1O',
                  'O=C(Nc1cccc2cnccc12)c1cc([N+](=O)[O-])c(Sc2c(Cl)cncc2Cl)s1',
                  'CCN(CC)Cc1ccc(-c2nc(C)c(COc3ccc([C@H](CC(=O)N4C(=O)OC[C@@H]4Cc4ccccc4)c4ccon4)cc3)s2)cc1']
rct_preds = reaction_prediction(uspto_50k_model_path, pdt_smiles_lst, task_type="retro-synthesis")

# forward-synthesis planning
uspto_480k_model_path = "./model_path/USPTO_480k"
rct_smiles_lst = ['C1CCOC1.CC(C)C[Mg+].CON(C)C(=O)c1ccc(O)nc1.[Cl-]',
                  'CN.O.O=C(O)c1ccc(Cl)c([N+](=O)[O-])c1',
                  'CCn1cc(C(=O)O)c(=O)c2cc(F)c(-c3ccc(N)cc3)cc21.O=CO']
pdt_preds = reaction_prediction(uspto_480k_model_path, rct_smiles_lst, task_type="forward-synthesis")
```

**Reaction embeddings generation**

```python3
from rxngraphormer.rxn_emb import RXNEMB

# base on pretrained model
pretrain_model_path = "./model_path/pretrained_classification_model"
rxnemb_calc_pretrained = RXNEMB(pretrained_model_path=pretrain_model_path,model_type="classifier")
rxn_emb_pretrained = rxnemb_calc_pretrained.gen_rxn_emb(["C1CCCCC1.CCO.CS(=O)(=O)N1CCN(Cc2ccccc2)CC1.[OH-].[OH-].[Pd+2]>>CS(=O)(=O)N1CCNCC1",
                                              "CCOC(C)=O.Cc1cc([N+](=O)[O-])ccc1NC(=O)c1ccccc1.Cl[Sn]Cl.O.O.O=C([O-])O.[Na+]>>Cc1cc(N)ccc1NC(=O)c1ccccc1",
                                              "COc1ccc(-c2coc3ccc(-c4nnc(S)o4)cc23)cc1.COc1ccc(CCl)cc1F>>COc1ccc(-c2coc3ccc(-c4nnc(SCc5ccc(OC)c(F)c5)o4)cc23)cc1"])

# base on finetuned model
finetune_model_path = "./model_path/buchwald_harwig/seed0"
rxnemb_calc_finetuneed = RXNEMB(pretrained_model_path=finetune_model_path,model_type="regressor")
rxn_emb_finetuned = rxnemb_calc_finetuneed.gen_rxn_emb(["CC(C)c1cc(C(C)C)c(-c2ccccc2P(C2CCCCC2)C2CCCCC2)c(C(C)C)c1.CN(C)C(=NC(C)(C)C)N(C)C.Cc1ccc(N)cc1.FC(F)(F)c1ccc(I)cc1.c1ccc(-c2ccon2)cc1>>Cc1ccc(Nc2ccc(C(F)(F)F)cc2)cc1",
                                                        "CCN=P(N=P(N(C)C)(N(C)C)N(C)C)(N(C)C)N(C)C.COc1ccc(OC)c(P([C@]23C[C@H]4C[C@H](C[C@H](C4)C2)C3)[C@]23C[C@H]4C[C@H](C[C@H](C4)C2)C3)c1-c1c(C(C)C)cc(C(C)C)cc1C(C)C.Cc1ccc(N)cc1.Ic1cccnc1.c1ccc(-c2ccon2)cc1>>Cc1ccc(Nc2cccnc2)cc1",
                                                        "CC(C)c1cc(C(C)C)c(-c2ccccc2P(C2CCCCC2)C2CCCCC2)c(C(C)C)c1.CCc1ccc(I)cc1.CN1CCCN2CCCN=C12.Cc1ccc(N)cc1.c1ccc2oncc2c1>>CCc1ccc(Nc2ccc(C)cc2)cc1"])
```

## ⚛️ Model Training

Although the script automatically preprocesses the data when you start the model training program, we recommend that you perform data preprocessing separately to avoid errors when using multi-gpu training models. Multi-gpu is usually used for model pretraining and fine-tuning for reaction synthesis planning tasks.

In the pre-training task, multiple data sub-files are used to create training sets and validation sets. The `file_num_trunck` parameter in the json file can control how many sub-files are used. If it is set to `0`, all sub-files are used.

```bash
python data_preprocess.py --config_json ./config/pretrain_parameters.json                                                                         # pretrain data preprocessing
CUDA_VISIBLE_DEVICES="0,1" python -m torch.distributed.launch --nproc_per_node 2 train_model.py --config_json ./config/pretrain_parameters.json   # pretrain model with multi-gpu
python train_model.py --config_json ./config/C_H_func_parameters.json                                                                             # finetune regression model
python train_model.py --config_json ./config/uspto_50k_parameters.json                                                                            # finetune sequence genration model
```

## 🔢 Evaluation

```bash
python eval_model.py --config_json ./config/buchwald_hartwig_eval.json    # evaluate regression model
python eval_model.py --config_json ./config/uspto_50k_eval.json           # evaluate sequence generation model
```

## 📑 Some results in paper

Here we provide a [notebook](https://github.com/licheng-xu-echo/RXNGraphormer/blob/main/notebook/demo.ipynb) to show how to generate fictitious reactions based on real reactions, how to generate delta-mol SMILES based on reaction SMILES, how to evaluate the performance of RXNGraphormer, use the embeddings generated by pre-trained RXNGraphormer to measure the distance between different reaction types in USPTO-50k, and use the embeddings generated by fine-tuned RXNGraphormer to examine the model's understanding of reaction structure-activity relationship.

## 📚 Citation

This paper is currently under review.
