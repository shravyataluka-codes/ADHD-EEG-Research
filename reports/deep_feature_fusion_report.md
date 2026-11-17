# Deep Feature-Fusion Experiment

- Run mode: **full**
- Cell: **Cell 1 only**
- Epochs in source Cell 1 data: **553**
- Epochs evaluated: **553** (Control=294, ADHD=259)
- Raw EEG shape: **[2, 1024]**; handcrafted input: **40 features**
- Class mapping: **Control = 0, ADHD = 1**; ADHD is positive.
- Threshold: **0.5**, fixed; no threshold tuning on test folds.
- Group IDs are provisional: **True**

## Methods and leakage controls

The existing completed EEG-Transformer full-run OOF predictions and fold checkpoints provide Arm A without modifying or retraining the baseline. Its saved group splits were checked against the Cell 1 metadata and deterministic group-aware split procedure. Arms B and C use those exact same inner and outer splits in a full run. The smoke run uses a small group subset and creates new group-aware inner splits, while retaining the same five outer-fold IDs.

Handcrafted features were generated independently per epoch using the existing feature extractor, preserving its 40 names and ordering. Raw EEG and feature rows, labels, groups, and folds were checked for exact epoch-order alignment. Both raw-channel normalization and feature-wise standardization were fitted on inner-training epochs only. The logistic-regression estimator is fit on inner-training rows only; the fusion model uses validation loss for scheduling, early stopping, and checkpoint selection. Each included epoch receives exactly one OOF prediction.

Group IDs are provisional filename/storage-slot identifiers, not verified participant IDs. Results are not verified participant-independent evaluation and do not establish clinical utility. The two channel slots remain unnamed.

## Ablation arms

- **A - Raw EEG Transformer:** existing full-run OOF reference. Checkpoint and history copies are stored in the fusion output; the original baseline files remain unchanged.
- **B - Handcrafted-only logistic regression:** 40 existing features, training-only feature normalization, fixed L2 logistic regression.
- **C - EEG Transformer + handcrafted fusion:** Full runs initialize from the matching fold's baseline Transformer checkpoint; smoke runs start from a seeded initialization to avoid using baseline training data in their newly formed validation splits. The model uses a compact 16-dimensional projection per modality and a 2-class head. Backbone freezing is configurable.

## Results

### Pooled OOF metrics

The following metrics are calculated from the combined 553 out-of-fold predictions for each arm. They are distinct from the unweighted mean and sample standard deviation across the five outer folds shown below.

| Arm | Accuracy | Balanced accuracy | Sensitivity | Specificity | F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|---:|
| A_raw_eeg_transformer | 0.9005 | 0.9009 | 0.9073 | 0.8946 | 0.8952 | 0.9451 |
| B_handcrafted_logistic | 0.9349 | 0.9344 | 0.9266 | 0.9422 | 0.9302 | 0.9644 |
| C_transformer_plus_handcrafted | 0.9295 | 0.9295 | 0.9305 | 0.9286 | 0.9251 | 0.9668 |

### Fold mean and sample standard deviation

| Arm | Metric | Mean | Sample SD (ddof=1) |
|---|---|---:|---:|
| A_raw_eeg_transformer | accuracy | 0.9013 | 0.0323 |
| A_raw_eeg_transformer | balanced_accuracy | 0.9021 | 0.0346 |
| A_raw_eeg_transformer | precision | 0.8903 | 0.0699 |
| A_raw_eeg_transformer | recall_sensitivity | 0.9117 | 0.1041 |
| A_raw_eeg_transformer | specificity | 0.8925 | 0.0704 |
| A_raw_eeg_transformer | f1_score | 0.8947 | 0.0392 |
| A_raw_eeg_transformer | roc_auc | 0.9664 | 0.0289 |
| B_handcrafted_logistic | accuracy | 0.9348 | 0.0411 |
| B_handcrafted_logistic | balanced_accuracy | 0.9344 | 0.0388 |
| B_handcrafted_logistic | precision | 0.9403 | 0.0822 |
| B_handcrafted_logistic | recall_sensitivity | 0.9276 | 0.0434 |
| B_handcrafted_logistic | specificity | 0.9413 | 0.0836 |
| B_handcrafted_logistic | f1_score | 0.9313 | 0.0395 |
| B_handcrafted_logistic | roc_auc | 0.9690 | 0.0330 |
| C_transformer_plus_handcrafted | accuracy | 0.9293 | 0.0257 |
| C_transformer_plus_handcrafted | balanced_accuracy | 0.9297 | 0.0240 |
| C_transformer_plus_handcrafted | precision | 0.9236 | 0.0677 |
| C_transformer_plus_handcrafted | recall_sensitivity | 0.9332 | 0.0535 |
| C_transformer_plus_handcrafted | specificity | 0.9262 | 0.0701 |
| C_transformer_plus_handcrafted | f1_score | 0.9256 | 0.0241 |
| C_transformer_plus_handcrafted | roc_auc | 0.9775 | 0.0218 |

### Per-fold sensitivity and specificity

| Arm | Fold | Sensitivity | Specificity | Balanced accuracy |
|---|---:|---:|---:|---:|
| A_raw_eeg_transformer | 0 | 0.9286 | 0.8730 | 0.9008 |
| A_raw_eeg_transformer | 1 | 0.7321 | 1.0000 | 0.8661 |
| A_raw_eeg_transformer | 2 | 1.0000 | 0.8929 | 0.9464 |
| A_raw_eeg_transformer | 3 | 0.9388 | 0.8036 | 0.8712 |
| A_raw_eeg_transformer | 4 | 0.9592 | 0.8929 | 0.9260 |
| B_handcrafted_logistic | 0 | 0.9643 | 0.9206 | 0.9425 |
| B_handcrafted_logistic | 1 | 0.8571 | 1.0000 | 0.9286 |
| B_handcrafted_logistic | 2 | 0.9592 | 0.9821 | 0.9707 |
| B_handcrafted_logistic | 3 | 0.9388 | 0.8036 | 0.8712 |
| B_handcrafted_logistic | 4 | 0.9184 | 1.0000 | 0.9592 |
| C_transformer_plus_handcrafted | 0 | 0.9107 | 0.9524 | 0.9315 |
| C_transformer_plus_handcrafted | 1 | 0.8571 | 1.0000 | 0.9286 |
| C_transformer_plus_handcrafted | 2 | 1.0000 | 0.8929 | 0.9464 |
| C_transformer_plus_handcrafted | 3 | 0.9592 | 0.8214 | 0.8903 |
| C_transformer_plus_handcrafted | 4 | 0.9388 | 0.9643 | 0.9515 |

### Fusion comparison

- accuracy: fusion was higher by 0.0280 on average across paired outer folds (fold differences: +0.0336, +0.0588, +0.0000, +0.0190, +0.0286).
- balanced_accuracy: fusion was higher by 0.0276 on average across paired outer folds (fold differences: +0.0308, +0.0625, +0.0000, +0.0191, +0.0255).
- recall_sensitivity: fusion was higher by 0.0214 on average across paired outer folds (fold differences: -0.0179, +0.1250, +0.0000, +0.0204, -0.0204).
- specificity: fusion was higher by 0.0337 on average across paired outer folds (fold differences: +0.0794, +0.0000, +0.0000, +0.0179, +0.0714).
- f1_score: fusion was higher by 0.0308 on average across paired outer folds (fold differences: +0.0307, +0.0777, +0.0000, +0.0189, +0.0269).
- roc_auc: fusion was higher by 0.0111 on average across paired outer folds (fold differences: +0.0130, +0.0176, +0.0022, +0.0211, +0.0015).
This is a descriptive five-fold comparison. Improvement is not claimed unless the paired fold results and pooled OOF metrics support it; fold variability and provisional group identities limit generalization.

Confusion matrices and ROC data are saved per arm. Metric undefinedness is preserved as `null` with the explanations from the existing metrics implementation. Fold means and standard deviations are descriptive only; no significance testing or threshold selection was performed.

## Configuration and artifacts

- Baseline source (read-only): `C:\Users\jyoth\Downloads\ADHD-HC mendeley dataset\results\deep_learning\cell_01\full\run_20261115T090201955591Z`
- Run output: `C:\Users\jyoth\Downloads\ADHD-HC mendeley dataset\results\deep_learning\cell_01\feature_fusion\full\run_20261115T110540461128Z`
- Feature order preserved: `["channel_slot_1_absolute_delta_power", "channel_slot_1_absolute_theta_power", "channel_slot_1_absolute_alpha_power", "channel_slot_1_absolute_beta_power", "channel_slot_1_absolute_low_gamma_power", "channel_slot_1_relative_delta_power", "channel_slot_1_relative_theta_power", "channel_slot_1_relative_alpha_power", "channel_slot_1_relative_beta_power", "channel_slot_1_relative_low_gamma_power", "channel_slot_1_theta_to_beta_power_ratio", "channel_slot_1_mean", "channel_slot_1_standard_deviation", "channel_slot_1_variance", "channel_slot_1_root_mean_square", "channel_slot_1_skewness", "channel_slot_1_kurtosis_excess", "channel_slot_1_hjorth_activity", "channel_slot_1_hjorth_mobility", "channel_slot_1_hjorth_complexity", "channel_slot_2_absolute_delta_power", "channel_slot_2_absolute_theta_power", "channel_slot_2_absolute_alpha_power", "channel_slot_2_absolute_beta_power", "channel_slot_2_absolute_low_gamma_power", "channel_slot_2_relative_delta_power", "channel_slot_2_relative_theta_power", "channel_slot_2_relative_alpha_power", "channel_slot_2_relative_beta_power", "channel_slot_2_relative_low_gamma_power", "channel_slot_2_theta_to_beta_power_ratio", "channel_slot_2_mean", "channel_slot_2_standard_deviation", "channel_slot_2_variance", "channel_slot_2_root_mean_square", "channel_slot_2_skewness", "channel_slot_2_kurtosis_excess", "channel_slot_2_hjorth_activity", "channel_slot_2_hjorth_mobility", "channel_slot_2_hjorth_complexity"]`
- Transformer configuration: `{"dropout": 0.2, "embed_dim": 32, "feedforward_dim": 64, "num_heads": 4, "num_layers": 2, "token_kernel_size": 16, "token_stride": 8}`
- Handcrafted baseline configuration: `{"C": 1.0, "class_weighting_uses_training_fold": true, "maximum_iterations": 1000, "model": "logistic_regression", "random_state": 42}`
- Optimization configuration: `{"batch_size": 16, "early_stopping_patience": 12, "gradient_clip_norm": 1.0, "learning_rate": 0.001, "learning_rate_scheduler": {"factor": 0.5, "minimum_learning_rate": 1e-06, "patience": 3}, "maximum_epochs": 100, "minimum_validation_improvement": 0.0, "weight_decay": 0.0001}`
- Fusion configuration: `{"classifier_hidden_dim": 16, "dropout": 0.3, "feature_count": 40, "freeze_transformer_backbone": false, "projection_dim": 16, "warm_start_from_fold_baseline": true}`
- Per-fold checkpoints, normalization parameters, training histories, group-split records, OOF predictions, metrics, confusion matrices, ROC data, and comparison tables are under the run output directory.

## Commands

```powershell
python -m src.train_deep_fusion --smoke-test --device cpu
python -m src.train_deep_fusion --device cpu
```
