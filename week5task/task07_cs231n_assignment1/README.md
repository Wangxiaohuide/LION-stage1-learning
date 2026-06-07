# Week 5 Task 07: CS231n Assignment 1

## Goal

Complete CS231n 2026 Assignment 1 locally:

1. `knn.ipynb`: k-Nearest Neighbor classifier
2. `softmax.ipynb`: Softmax classifier
3. `two_layer_net.ipynb`: Two-layer neural network
4. `features.ipynb`: Image features
5. `FullyConnectedNets.ipynb`: Fully connected networks

## Dataset Status

The required datasets have been downloaded into:

`cs231n/datasets/`

Included files:

- `cifar-10-batches-py/`
- `cifar-10-python.tar.gz`
- `imagenet_val_25.npz`

Verified CIFAR-10 load result:

```text
X_train: (50000, 32, 32, 3)
y_train: (50000,)
X_test:  (10000, 32, 32, 3)
y_test:  (10000,)
```

## Local Environment

Verified local environment:

```text
Python 3.12.4
numpy 1.26.4
matplotlib 3.8.4
scipy 1.13.1
imageio 2.33.1
Jupyter available
```

Install dependencies if needed:

```powershell
python -m pip install -r requirements.txt
```

Run Jupyter locally from this directory:

```powershell
jupyter notebook
```

The original starter code has been backed up at:

`cs231n_starter_backup/`

The current `cs231n/` directory uses the existing week4 implementation so this week5 folder can reproduce the assignment locally.

## Verification

Run the core implementation check:

```powershell
python verify_assignment1_core.py
```

Latest verification result:

```text
PASS test_data_loads
PASS test_knn_distances_and_labels
PASS test_softmax_vectorized_matches_naive
PASS test_affine_relu_softmax_layers
PASS test_two_layer_net_gradients
PASS test_fully_connected_net_forward_backward_and_training
```

## Reproduction Results

Run a small local reproduction experiment:

```powershell
python run_week5_reproduction.py
```

Latest run summary saved at:

`results/week5_reproduction_summary.json`

Current quick-run results:

```text
kNN best validation accuracy: 0.275, best k: 1
Softmax raw-pixel best validation accuracy: 0.323, lr: 1e-6, reg: 2.5e4
TwoLayerNet raw-pixel best validation accuracy: 0.318
```

## Suggested Work Order

Start from `knn.ipynb`, then proceed in the notebook order listed above. The assignment skeleton still contains TODO sections in `cs231n/`; implement them while running the notebooks.
