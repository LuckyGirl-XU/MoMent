# MoMent

Code for our AAAI 2026 work, **Unlocking Multi-Modal Potentials for Link Prediction on Dynamic Text-Attributed Graph Representation**.

## Setup

The recorded GPU experiments used Python 3.7.13, PyTorch 1.13.1 with CUDA 11.6,
NumPy 1.21.5, pandas 1.3.5, scikit-learn 1.0.2, and Transformers 4.21.0 on
NVIDIA RTX A6000 GPUs. See [the recorded environment](reproduction/environment.json).

Install PyTorch for your CUDA environment, then the remaining dependencies:

```bash
conda create -n moment-reproduction python=3.7.13
conda activate moment-reproduction
python -m pip install torch==1.13.1+cu116 --extra-index-url https://download.pytorch.org/whl/cu116
python -m pip install -r requirements-reproduction.txt
```

## Datasets

Download the datasets from the [DTGB benchmark](https://drive.google.com/drive/folders/1QFxHIjusLOFma30gF59_hcB19Ix3QZtk).
The paper datasets are Enron, Stack_elec, ICEWS1819, Googlemap_CT, Stack_ubuntu,
Amazon_movies, and Yelp. Place each dataset under the repository root:

```text
DyLink_Datasets/
  Enron/
    edge_list.csv
    entity_text.csv
    relation_text.csv
    e_feat.npy
    r_feat.npy
  Stack_elec/
  ...
```

The fixed configurations use BERT node and edge features: 768-dimensional arrays
indexed by the CSV IDs, with zero padding at row 0. Use the same prepared features
as the recorded run when comparing results; changing text embeddings changes the
experiment. Recorded input fingerprints, when available, are in
[the data manifests](reproduction/data_manifests.json).

For Yelp, the resumable preparation helper uses a locally cached
`bert-base-uncased` model, pooler output, a maximum sequence length of 512, and
rounding to eight decimal places:

```bash
python scripts/prepare_yelp_features.py --data-root ./DyLink_Datasets --zip-file /path/to/Yelp.zip --bert-model /path/to/bert-base-uncased --gpu 0
```

## Link prediction

Run from the repository root. Dataset-specific parameters load automatically;
`--best-configs` makes that choice explicit:

```bash
python train_link_prediction.py --dataset_name Enron --best-configs --gpu 0
python train_link_prediction.py --dataset_name Stack_ubuntu --best-configs --gpu 0
```

`--load_best_configs` is an equivalent flag. The default is one run with seed 0.
Use `--num_runs 5` to request five seeds explicitly. Use `--no_best_configs` to
supply manual hyperparameters without the dataset overrides:

```bash
python train_link_prediction.py --dataset_name Enron --no_best_configs --use_feature Bert --learning_rate 0.00003 --gpu 0
```

The executable settings are in
[configs/moment_link_prediction.json](configs/moment_link_prediction.json).
Shared settings are `time_feat_dim=100`, `batch_size=512`, one layer, two heads,
20 recent neighbors, dropout 0.1, Adam, zero weight decay, and patience 5.
Training retains the original limit of 100 epochs and the original checkpoint
selection and evaluation behavior.

## Fixed parameters and recorded results

The table reports **one seed-0 run**, using the final selected checkpoint's
**new-node test** AP/AUC. The comparison criterion is an absolute difference of
at most two **percentage points** from the paper for **both** metrics, in either
direction. These measurements do not guarantee identical scores on different
hardware, data features, or random seeds.

| Dataset | loss_weight | learning_rate | AP (%) | AUC (%) | AP / AUC difference from paper (pp) | Within 2 pp |
| --- | ---: | ---: | ---: | ---: | --- | :---: |
| Enron | 0.2 | 1e-5 | 98.03 | 97.99 | +1.32 / +0.88 | Yes |
| Stack_elec | 0.1 | 1e-5 | 94.45 | 93.06 | -1.29 / -1.76 | Yes |
| ICEWS1819 | 0.2 | 1e-5 | 100.00 | 100.00 | +0.76 / +0.62 | Yes |
| Googlemap_CT | 0.2 | 1e-5 | 96.13 | 94.50 | -0.47 / -1.10 | Yes |
| Stack_ubuntu | 0.5 | 3e-5 | 95.04 | 94.04 | -0.22 / -1.95 | Yes |
| Amazon_movies | 0.2 | 1e-5 | 97.01 | 96.71 | -1.59 / -1.66 | Yes |
| Yelp | 0.2 | 1e-5 | 100.00 | 100.00 | +3.52 / +2.87 | **No** |

Six of seven datasets meet this criterion. Yelp remains above the range after
all three declared parameter groups; it is recorded as `outside_tolerance`.
Default parameters were preferred when they met the criterion. Otherwise the
three declared groups were compared by new-node validation AP/AUC mean, without
selecting random seeds or intermediate test scores. Yelp's default and 128-dimension
candidate tie in the saved validation scores, so the default is retained.

Machine-readable results and all completed candidate comparisons are in
[reproduction/seed0_results.json](reproduction/seed0_results.json).
[configs/moment_fixed_results.json](configs/moment_fixed_results.json) summarizes
the fixed settings, and
[configs/moment_verification.json](configs/moment_verification.json) records their
verification status. GDELT and Stack_english have default configurations but were
not part of this seven-dataset verification.

The original `models/MoMent.py`, including its existing `time_gap` and batch
behavior, is unchanged. Its SHA256 is
`eb182f1205566957d548ace13d93965b7618eb05a559e81735d266248c38ccc0`.

## Checks

```bash
python -m unittest discover -s tests -v
python train_link_prediction.py --help
```

These checks cover configuration loading, CLI aliases, the original model hash,
data preparation, and the reproduction runner. They do not replace GPU training.
Large datasets, embeddings, checkpoints, and runtime logs are excluded from Git.

## Edge classification

```bash
python train_edge_classification.py --dataset_name Enron --model_name MoMent --num_runs 5 --gpu 0 --use_feature Bert --load_best_configs
```

The fixed link-prediction configurations do not change edge-classification settings.

## Baselines and acknowledgments

Our implementations refer to [DTGB](https://github.com/zjs123/DTGB) and
[TPNet](https://github.com/lxd99/TPNet). The previously missing baseline import
modules `CAWN.py`, `EdgeBank.py`, `MemoryModel.py`, and `TCL.py` are included from
DTGB commit `840eb442` so the original training entry point can import successfully.
