# MoMent
This repo is for our AAAI 2026 work - 'Unlocking Multi-Modal Potentials for Link Prediction on Dynamic Text-Attributed Graph Representation'.

## Datasets
We collect dynamic text-attributed graphs from the DTGB benchmark [here](https://drive.google.com/drive/folders/1QFxHIjusLOFma30gF59_hcB19Ix3QZtk). Below we provide the details of the datasets:
- email network (Enron)
- knowledge graph (ICEWS1819)
-  multi-round dialogue (Stack elec and Stack ubuntu)
-  E-commerce network (Googlemap CT, Amazon movies, and Yelp)

### Dataset Usage

After downloading the datasets, they should be uncompressed into the DyLink_Datasets folder.

Run `get_pretrained_embeddings.py` to obtain the Bert-based node and edge text embeddings. They will be saved as e_feat.npy and r_feat.npy respectively.


## Reproduce the Results

### Future Link Prediction Task
  - Example of training MoMent on Enron dataset:
  ```
  python train_link_prediction.py --dataset_name Enron --model_name MoMent --num_runs 5 --gpu 0 --use_feature 'Bert' --load_best_configs
  ```

### Edge Classification Task

 - Example of training MoMent on Enron dataset:
  ```
  python train_edge_classification.py --dataset_name Enron --model_name MoMent --num_runs 5 --gpu 0 --use_feature 'Bert' --load_best_configs
  ```

## Baseline

  - Codes for DTGB baseline are available [here](https://github.com/zjs123/DTGB).
  - Codes for the TPNet are available [here](https://github.com/lxd99/TPNet).

## Acknowledge
Our model implementations are referred to the above projects. Thanks for their great contributions!
