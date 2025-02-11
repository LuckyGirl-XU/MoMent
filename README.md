# MoMent
This repo is for our KDD 2025 submission.

## Datasets
We collect dynamic text-attributed graphs from the DTGB benchmark [here](https://drive.google.com/drive/folders/1QFxHIjusLOFma30gF59_hcB19Ix3QZtk). Below we provide the details of the datasets:
- email network (Enron)
- knowledge graph (ICEWS1819)
-  multi-round dialogue (Stack elec and Stack ubuntu)
-  E-commerce network (Googlemap CT, Amazon movies, and Yelp)

After downloading the datasets, they should be uncompressed into the DyLink_Datasets folder.
Run `get_pretrained_embeddings.py` to obtain the Bert-based node and edge text embeddings. They will be saved as e_feat.npy and r_feat.npy respectively.
