# MoMent
This repo is for our KDD 2025 submission.

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
  - Example of training MoMent with GraphMixer on Enron dataset with text attributes:
  ```
  python train_link_prediction.py --dataset_name Enron --model_name GraphMixer --num_runs 5 --gpu 0 --use_feature 'Bert' --loss_weight 0.2 
  ```

### Edge Classification Task

 - Example of training MoMent with GraphMixer on Enron dataset with text attributes:
  ```
  python train_edge_classification.py --dataset_name Enron --model_name GraphMixer --num_runs 5 --gpu 0 --use_feature 'Bert' --loss_weight 0.2 
  ```
