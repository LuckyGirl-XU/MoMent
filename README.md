# MoMent

Official implementation of **Unlocking Multi-Modal Potentials for Link Prediction on Dynamic Text-Attributed Graph Representation** (AAAI 2026).

## Installation

```bash
git clone https://github.com/LuckyGirl-XU/MoMent.git
cd MoMent
conda create -n moment python=3.7.13 -y
conda activate moment
pip install torch==1.13.1+cu116 --extra-index-url https://download.pytorch.org/whl/cu116
pip install -r requirements-reproduction.txt
```

## Datasets

Download the datasets from [DTGB](https://drive.google.com/drive/folders/1QFxHIjusLOFma30gF59_hcB19Ix3QZtk) and organize the prepared data as follows:

```text
DyLink_Datasets/
├── Enron/
│   ├── edge_list.csv
│   ├── e_feat.npy
│   └── r_feat.npy
├── Stack_elec/
├── ICEWS1819/
├── Googlemap_CT/
├── Stack_ubuntu/
├── Amazon_movies/
└── Yelp/
```

Training uses 768-dimensional BERT node and edge features (`e_feat.npy` and `r_feat.npy`), indexed by the dataset IDs with a zero padding row at index 0. Prepare these features before running training.

## Link Prediction

Run from the repository root:

```bash
python train_link_prediction.py --dataset_name Enron --gpu 0
```

Parameters are loaded automatically for each dataset from [configs/moment_link_prediction.json](configs/moment_link_prediction.json). No manual hyperparameter changes are needed. The default is one run with seed 0.

To specify configuration loading explicitly or run multiple seeds:

```bash
python train_link_prediction.py --dataset_name Stack_ubuntu --best-configs --gpu 0
python train_link_prediction.py --dataset_name Enron --num_runs 5 --gpu 0
```

`--load_best_configs` is an alias for `--best-configs`. Use `--no_best_configs` to supply custom hyperparameters.

## Edge Classification

```bash
python train_edge_classification.py --dataset_name Enron --model_name MoMent --num_runs 5 --gpu 0 --use_feature Bert --load_best_configs
```

## Acknowledgments

Our implementation builds on [DTGB](https://github.com/zjs123/DTGB) and [TPNet](https://github.com/lxd99/TPNet).
