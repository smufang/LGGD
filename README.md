# LGGD
Code for KDD'24 Paper: A Learned Generalized Geodesic Distance Function-Based Approach for Node Feature Augmentation on Graphs.

## Environment
The code was last tested with:
- Python 3.10.10
- Pytorch 1.13.0
- Torch-geometric 2.3.1
- Torch-diffeq 0.2.3


## Description
We used only public datasets. The default dataset is cora and the default backbone is gcn.

- **table1_src**: Source code for Table1 in the paper
- **robustness_src**: Jupyter-notebook to generate the images for the Figure 2 in the paper.
- **dynamic_src**: Soruce code for the Figure 5 bottom row in the paper.

## Training
```python
# to generate lggd node features and run on a gcn backbone
python table1_src/main.py

# to test dynamic inclusion of the new labels
python dynamic_src/main_val.py
```

## Cite
```
@inproceedings{10.1145/3637528.3671858,
author = {Azad, Amitoz and Fang, Yuan},
title = {A Learned Generalized Geodesic Distance Function-Based Approach for Node Feature Augmentation on Graphs},
year = {2024},
booktitle = {Proceedings of the 30th ACM SIGKDD Conference on Knowledge Discovery and Data Mining},
pages = {49–58},
numpages = {10},
series = {KDD '24}
}
```



