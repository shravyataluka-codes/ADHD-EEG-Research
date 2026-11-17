# Deep-Learning Training and Evaluation Report

- Run mode: **full**
- Cell: **1 only**
- Input epochs: **553**
- Raw epoch shape: **[1024, 2]**
- Class mapping: **Control = 0, ADHD = 1**
- Device: **cpu**
- Seed: **42**

## Evaluation procedure

Five existing Cell 1 fold assignments were used as outer test folds. For each outer fold, the other four folds were split into training and inner validation groups using deterministic class-stratified group sampling. No group crossed training, validation, and test splits. Training-only per-channel mean and standard deviation were fitted on the inner-training epochs and then applied unchanged to validation and test epochs. Early stopping and checkpoint selection used inner validation loss only. The selected checkpoint was loaded before one prediction pass on the outer test fold.

Out-of-fold predictions were checked so each included epoch had exactly one prediction. ADHD is the positive class. Undefined metrics are recorded as `null` with an explanation rather than replaced by a misleading number.

## Data and identity limitations

The experiment uses raw Cell 1 epochs only; it does not use the 40-column handcrafted feature matrix or combine recording cells. The two channel slots remain unnamed because channel order is not independently verified. Group IDs are provisional source-file/storage-slot identifiers, not verified participant IDs. Consequently, this procedure provides provisional group separation only and must not be described as verified participant-independent evaluation. No clinical diagnostic performance is established.

## Models and results

| Model | Parameters | OOF epochs | Accuracy | Balanced accuracy | Precision | Sensitivity | Specificity | F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| eegnet | 1370 | 553 | 0.8373 | 0.8292 | 0.9333 | 0.7027 | 0.9558 | 0.8018 | 0.9200 |
| cnn_lstm | 15650 | 553 | 0.8734 | 0.8674 | 0.9479 | 0.7722 | 0.9626 | 0.8511 | 0.9445 |
| eeg_transformer | 22402 | 553 | 0.9005 | 0.9009 | 0.8835 | 0.9073 | 0.8946 | 0.8952 | 0.9451 |
| eeg_conformer | 13818 | 553 | 0.8590 | 0.8570 | 0.8664 | 0.8263 | 0.8878 | 0.8458 | 0.9339 |

## Confusion matrices

### eegnet

Rows are true labels `[Control, ADHD]`; columns are predicted labels `[Control, ADHD]`.

```json
[
  [
    281,
    13
  ],
  [
    77,
    182
  ]
]
```

### cnn_lstm

Rows are true labels `[Control, ADHD]`; columns are predicted labels `[Control, ADHD]`.

```json
[
  [
    283,
    11
  ],
  [
    59,
    200
  ]
]
```

### eeg_transformer

Rows are true labels `[Control, ADHD]`; columns are predicted labels `[Control, ADHD]`.

```json
[
  [
    263,
    31
  ],
  [
    24,
    235
  ]
]
```

### eeg_conformer

Rows are true labels `[Control, ADHD]`; columns are predicted labels `[Control, ADHD]`.

```json
[
  [
    261,
    33
  ],
  [
    45,
    214
  ]
]
```

## Hyperparameters

- Inner validation fraction: `0.2` (group-stratified).
- Optimization settings: `{"batch_size": 16, "early_stopping_patience": 12, "gradient_clip_norm": 1.0, "learning_rate": 0.001, "learning_rate_scheduler": {"factor": 0.5, "minimum_learning_rate": 1e-06, "patience": 3}, "maximum_epochs": 100, "minimum_validation_improvement": 0.0, "weight_decay": 0.0001}`
- Class weighting policy: `{"enabled": true, "imbalance_ratio_threshold": 1.1}`

Model-specific architecture settings:

- **eegnet:** `{"depth_multiplier": 2, "dropout": 0.25, "separable_filters": 16, "temporal_filters": 8, "temporal_kernel_size": 63}`
- **cnn_lstm:** `{"bidirectional": true, "conv_channels": [16, 32], "conv_kernel_sizes": [7, 5], "dropout": 0.25, "hidden_size": 24, "projection_dim": 32}`
- **eeg_transformer:** `{"dropout": 0.2, "embed_dim": 32, "feedforward_dim": 64, "num_heads": 4, "num_layers": 2, "token_kernel_size": 16, "token_stride": 8}`
- **eeg_conformer:** `{"dropout": 0.2, "embed_dim": 32, "feedforward_dim": 64, "num_heads": 4, "num_layers": 1, "pool_size": 8, "spatial_filters": 16, "temporal_filters": 8, "temporal_kernel_size": 25}`

## Configuration and artifacts

- Model configuration: `C:\Users\jyoth\Downloads\ADHD-HC mendeley dataset\configs\deep_models_cell1.yaml`
- Training configuration: `C:\Users\jyoth\Downloads\ADHD-HC mendeley dataset\configs\training_cell1.yaml`
- Output directory: `C:\Users\jyoth\Downloads\ADHD-HC mendeley dataset\results\deep_learning\cell_01\full\run_20261115T090201955591Z`
- Per-fold checkpoints, training histories, validation losses, group split records, metrics, out-of-fold predictions, and ROC curve data are saved under the output directory.
