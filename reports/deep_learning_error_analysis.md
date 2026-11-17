# Deep-Learning Error Analysis - Full Cell 1 Run

- Source run: `C:\Users\jyoth\Downloads\ADHD-HC mendeley dataset\results\deep_learning\cell_01\full\run_20261115T090201955591Z`
- Models: EEGNet-inspired, CNN-LSTM, EEG-Transformer-inspired, EEG Conformer.
- Scope: existing Cell 1 outer-fold OOF predictions and saved training histories; no retraining.
- Input: 553 epochs, shape `[553, 1024, 2]`; Control=294, ADHD=259; five folds; 79 provisional groups.
- Group identifiers are provisional filename/storage-slot keys, not verified participant IDs.

## Methods and interpretation limits

Per-fold metrics were read from the existing full-run outputs and independently recalculated from each model's saved OOF predictions. Pooled confusion matrices below are computed from the combined OOF rows and checked against the original results JSON. Fold summaries are unweighted mean and **sample standard deviation (ddof=1)** over the five folds; fold sizes are 119, 119, 105, 105, and 105 epochs. These fold summaries are not confidence intervals and do not account for dependence among epochs from a provisional group.

ADHD is treated as positive (1); Control is 0. The evaluation uses provisional group separation only. Storage-slot IDs do not establish participant identity across source files, so this is not verified participant-independent evaluation. The two channel slots remain unnamed. No clinical claim is made.

## Per-fold metrics

| Model | Fold | N | Accuracy | Balanced accuracy | Sensitivity | Specificity | F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| eegnet | 0 | 119 | 0.8487 | 0.8462 | 0.8036 | 0.8889 | 0.8333 | 0.9359 |
| eegnet | 1 | 119 | 0.8992 | 0.8938 | 0.8036 | 0.9841 | 0.8824 | 0.9674 |
| eegnet | 2 | 105 | 0.7905 | 0.7768 | 0.5714 | 0.9821 | 0.7179 | 0.9501 |
| eegnet | 3 | 105 | 0.8190 | 0.8074 | 0.6327 | 0.9821 | 0.7654 | 0.8841 |
| eegnet | 4 | 105 | 0.8190 | 0.8099 | 0.6735 | 0.9464 | 0.7765 | 0.8761 |
| cnn_lstm | 0 | 119 | 0.8908 | 0.8839 | 0.7679 | 1.0000 | 0.8687 | 0.9583 |
| cnn_lstm | 1 | 119 | 0.8571 | 0.8482 | 0.6964 | 1.0000 | 0.8211 | 0.9850 |
| cnn_lstm | 2 | 105 | 0.9524 | 0.9490 | 0.8980 | 1.0000 | 0.9462 | 0.9989 |
| cnn_lstm | 3 | 105 | 0.8952 | 0.8941 | 0.8776 | 0.9107 | 0.8866 | 0.9694 |
| cnn_lstm | 4 | 105 | 0.7714 | 0.7628 | 0.6327 | 0.8929 | 0.7209 | 0.8069 |
| eeg_transformer | 0 | 119 | 0.8992 | 0.9008 | 0.9286 | 0.8730 | 0.8966 | 0.9674 |
| eeg_transformer | 1 | 119 | 0.8739 | 0.8661 | 0.7321 | 1.0000 | 0.8454 | 0.9694 |
| eeg_transformer | 2 | 105 | 0.9429 | 0.9464 | 1.0000 | 0.8929 | 0.9423 | 0.9880 |
| eeg_transformer | 3 | 105 | 0.8667 | 0.8712 | 0.9388 | 0.8036 | 0.8679 | 0.9180 |
| eeg_transformer | 4 | 105 | 0.9238 | 0.9260 | 0.9592 | 0.8929 | 0.9216 | 0.9894 |
| eeg_conformer | 0 | 119 | 0.8824 | 0.8780 | 0.8036 | 0.9524 | 0.8654 | 0.9427 |
| eeg_conformer | 1 | 119 | 0.8319 | 0.8224 | 0.6607 | 0.9841 | 0.7872 | 0.9493 |
| eeg_conformer | 2 | 105 | 0.9714 | 0.9707 | 0.9592 | 0.9821 | 0.9691 | 0.9993 |
| eeg_conformer | 3 | 105 | 0.8762 | 0.8839 | 1.0000 | 0.7679 | 0.8829 | 0.9413 |
| eeg_conformer | 4 | 105 | 0.7333 | 0.7334 | 0.7347 | 0.7321 | 0.7200 | 0.8929 |

## Mean and sample standard deviation across folds

| Model | Accuracy | Balanced accuracy | Sensitivity | Specificity | F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|---:|
| eegnet | 0.8353 +/- 0.0412 | 0.8268 +/- 0.0448 | 0.6969 +/- 0.1039 | 0.9567 +/- 0.0411 | 0.7951 +/- 0.0637 | 0.9227 +/- 0.0406 |
| cnn_lstm | 0.8734 +/- 0.0665 | 0.8676 +/- 0.0689 | 0.7745 +/- 0.1142 | 0.9607 +/- 0.0542 | 0.8487 +/- 0.0843 | 0.9437 +/- 0.0780 |
| eeg_transformer | 0.9013 +/- 0.0323 | 0.9021 +/- 0.0346 | 0.9117 +/- 0.1041 | 0.8925 +/- 0.0704 | 0.8947 +/- 0.0392 | 0.9664 +/- 0.0289 |
| eeg_conformer | 0.8590 +/- 0.0866 | 0.8577 +/- 0.0874 | 0.8316 +/- 0.1449 | 0.8837 +/- 0.1234 | 0.8449 +/- 0.0951 | 0.9451 +/- 0.0378 |

Fold standard deviations reflect variation across only five particular splits. They should be interpreted descriptively, not as inferential uncertainty.

## Pooled out-of-fold confusion matrices

Rows are true labels and columns are predicted labels. The pooled matrices use all 553 saved predictions per model.

### eegnet

| True class | Predicted Control | Predicted ADHD |
|---|---:|---:|
| Control | 281 | 13 |
| ADHD | 77 | 182 |

False positives (Control -> ADHD): **13**; false negatives (ADHD -> Control): **77**.

### cnn_lstm

| True class | Predicted Control | Predicted ADHD |
|---|---:|---:|
| Control | 283 | 11 |
| ADHD | 59 | 200 |

False positives (Control -> ADHD): **11**; false negatives (ADHD -> Control): **59**.

### eeg_transformer

| True class | Predicted Control | Predicted ADHD |
|---|---:|---:|
| Control | 263 | 31 |
| ADHD | 24 | 235 |

False positives (Control -> ADHD): **31**; false negatives (ADHD -> Control): **24**.

### eeg_conformer

| True class | Predicted Control | Predicted ADHD |
|---|---:|---:|
| Control | 261 | 33 |
| ADHD | 45 | 214 |

False positives (Control -> ADHD): **33**; false negatives (ADHD -> Control): **45**.

## Misclassification patterns

| Model | False positives | False negatives | Dominant error type |
|---|---:|---:|---|
| eegnet | 13 | 77 | False negatives |
| cnn_lstm | 11 | 59 | False negatives |
| eeg_transformer | 31 | 24 | False positives |
| eeg_conformer | 33 | 45 | False negatives |
| **Sum across model predictions** | **88** | **205** | **Not independent observations** |

Across the four model outputs there are **205 false-negative model/epoch errors** and **88 false-positive model/epoch errors**. These are counts across model predictions, not independent observations. False negatives dominate for EEGNet-inspired, CNN-LSTM, and Conformer; the Transformer trades some false negatives for more false positives.

### Shared errors across models

The table counts epochs by how many of the four models misclassified them. Shared errors identify difficult epochs in this dataset/run, not verified participants.

| Number of models misclassifying an epoch | Number of epochs |
|---:|---:|
| 0 | 399 |
| 1 | 75 |
| 2 | 32 |
| 3 | 34 |
| 4 | 13 |

### Most error-prone provisional groups

Top groups below are storage-slot keys only. They are not participant identities. Error totals sum model-by-epoch errors across the four models.

| Provisional group key | Fold | True class | Epochs | Models with errors | Model/epoch errors | FP errors | FN errors |
|---|---:|---|---:|---:|---:|---:|---:|
| MADHD.mat:slot-020 | 4 | ADHD | 7 | 3 | 21 | 0 | 21 |
| FADHD.mat:slot-001 | 0 | ADHD | 7 | 4 | 19 | 0 | 19 |
| FADHD.mat:slot-008 | 1 | ADHD | 7 | 4 | 19 | 0 | 19 |
| MADHD.mat:slot-015 | 4 | ADHD | 7 | 4 | 19 | 0 | 19 |
| MADHD.mat:slot-017 | 1 | ADHD | 7 | 4 | 15 | 0 | 15 |

## Sensitivity-specificity trade-off

On pooled OOF predictions, EEGNet has the highest specificity (0.9558) but lower ADHD sensitivity (0.7027). CNN-LSTM similarly has high specificity (0.9626) with sensitivity 0.7722. The EEG-Transformer has the highest sensitivity (0.9073), with specificity 0.8946 and more pooled false positives (31). The Conformer is intermediate (sensitivity 0.8263, specificity 0.8878). These are threshold-0.5 results; no threshold was tuned on the test folds.

## EEG-Transformer fold consistency

| Metric | Fold mean | Sample SD | Range |
|---|---:|---:|---:|
| Accuracy | 0.9013 | 0.0323 | 0.8667-0.9429 |
| Balanced accuracy | 0.9021 | 0.0346 | 0.8661-0.9464 |
| Sensitivity | 0.9117 | 0.1041 | 0.7321-1.0000 |
| Specificity | 0.8925 | 0.0704 | 0.8036-1.0000 |
| ROC-AUC | 0.9664 | 0.0289 | 0.9180-0.9894 |

Transformer accuracy and ROC-AUC are relatively steady across these five folds, but performance is not uniform: sensitivity is lower in fold 1 (0.7321), and specificity is lower in fold 3 (0.8036). The spread is meaningful given the small number of folds; it does not demonstrate generalization to verified participants.

## Training versus validation loss

Each history ends when early stopping fired. An increase in validation loss means the final validation loss exceeded that fold's best validation loss; a decrease in training loss means training loss continued to fall after the best validation epoch. This pattern is consistent with late-epoch overfitting, although the inner validation sets are small and the curves can also reflect split variability.

| Model | Folds with final validation loss above best | Folds with training loss below best-epoch value | Folds showing both signals |
|---|---:|---:|---:|
| eegnet | 5/5 | 5/5 | 5/5 |
| cnn_lstm | 5/5 | 4/5 | 4/5 |
| eeg_transformer | 5/5 | 5/5 | 5/5 |
| eeg_conformer | 5/5 | 5/5 | 5/5 |

| Model | Fold | Epochs | Best epoch | Best val loss | Final val loss | Validation loss change (final minus best) | Training loss change (final minus best epoch) |
|---|---:|---:|---:|---:|---:|---:|---:|
| eegnet | 0 | 38 | 26 | 0.2968 | 0.3669 | +0.0700 | -0.0448 |
| eegnet | 1 | 40 | 28 | 0.1199 | 0.1582 | +0.0383 | -0.0650 |
| eegnet | 2 | 22 | 10 | 0.4439 | 0.4549 | +0.0109 | -0.0509 |
| eegnet | 3 | 35 | 23 | 0.2738 | 0.3321 | +0.0583 | -0.0637 |
| eegnet | 4 | 15 | 3 | 0.6012 | 0.6752 | +0.0741 | -0.0970 |
| cnn_lstm | 0 | 38 | 26 | 0.2633 | 0.3229 | +0.0596 | -0.0239 |
| cnn_lstm | 1 | 27 | 15 | 0.3881 | 0.4699 | +0.0818 | -0.0661 |
| cnn_lstm | 2 | 52 | 40 | 0.1924 | 0.2051 | +0.0127 | -0.0119 |
| cnn_lstm | 3 | 43 | 31 | 0.1649 | 0.2039 | +0.0389 | +0.0018 |
| cnn_lstm | 4 | 13 | 1 | 0.6364 | 0.7711 | +0.1347 | -0.4904 |
| eeg_transformer | 0 | 15 | 3 | 0.2282 | 0.4642 | +0.2360 | -0.2136 |
| eeg_transformer | 1 | 14 | 2 | 0.2166 | 0.3448 | +0.1282 | -0.2951 |
| eeg_transformer | 2 | 16 | 4 | 0.3455 | 0.5412 | +0.1958 | -0.1866 |
| eeg_transformer | 3 | 25 | 13 | 0.1103 | 0.1303 | +0.0201 | -0.0767 |
| eeg_transformer | 4 | 17 | 5 | 0.3765 | 1.0559 | +0.6793 | -0.1132 |
| eeg_conformer | 0 | 16 | 4 | 0.2373 | 0.2995 | +0.0622 | -0.0419 |
| eeg_conformer | 1 | 21 | 9 | 0.1453 | 0.2231 | +0.0778 | -0.2688 |
| eeg_conformer | 2 | 41 | 29 | 0.1369 | 0.1964 | +0.0595 | -0.0287 |
| eeg_conformer | 3 | 57 | 45 | 0.0093 | 0.0677 | +0.0584 | -0.0436 |
| eeg_conformer | 4 | 14 | 2 | 0.5818 | 1.0891 | +0.5074 | -0.1391 |

![Training and validation loss curves](../results/deep_learning/cell_01/error_analysis/run_20261115T090201955591Z/training_validation_loss_curves.png)

## Saved analysis artifacts

- Analysis directory: `../results/deep_learning/cell_01/error_analysis/run_20261115T090201955591Z/`
- [Per-fold metrics CSV](../results/deep_learning/cell_01/error_analysis/run_20261115T090201955591Z/per_fold_metrics.csv)
- [Fold metric summary CSV](../results/deep_learning/cell_01/error_analysis/run_20261115T090201955591Z/fold_metric_summary.csv)
- [Pooled OOF confusion matrix CSV](../results/deep_learning/cell_01/error_analysis/run_20261115T090201955591Z/pooled_oof_confusion_matrices.csv)
- [Fold false-positive/false-negative counts CSV](../results/deep_learning/cell_01/error_analysis/run_20261115T090201955591Z/fold_misclassification_counts.csv)
- [Learning-curve summary CSV](../results/deep_learning/cell_01/error_analysis/run_20261115T090201955591Z/learning_curve_summary.csv)
- [Provisional-group error summary CSV](../results/deep_learning/cell_01/error_analysis/run_20261115T090201955591Z/provisional_group_error_summary.csv)
- [Pooled confusion matrix plot](../results/deep_learning/cell_01/error_analysis/run_20261115T090201955591Z/pooled_oof_confusion_matrices.png)
- [Per-fold metric comparison plot](../results/deep_learning/cell_01/error_analysis/run_20261115T090201955591Z/per_fold_metric_comparison.png)
- [Sensitivity-specificity plot](../results/deep_learning/cell_01/error_analysis/run_20261115T090201955591Z/sensitivity_specificity_by_fold.png)
- [Training/validation loss curves](../results/deep_learning/cell_01/error_analysis/run_20261115T090201955591Z/training_validation_loss_curves.png)

## Conclusion

Across this run, the Transformer has the highest pooled accuracy, balanced accuracy, sensitivity, and ROC-AUC, while CNN-LSTM has the highest pooled specificity. Fold-level variability and late validation-loss increases warrant further validation. Because participant IDs are not verified and only five provisional group folds are available, these results are exploratory model comparisons for this dataset configuration, not evidence of clinical utility or confirmed participant-independent performance.
