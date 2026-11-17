# Comparative Model Evaluation: ADHD vs Healthy Control

## Scope and evaluation

- **Cell:** 1 only; 553 epochs (294 Control, 259 ADHD).
- **Experiments:** four raw-EEG deep architectures, handcrafted-feature-only logistic regression, raw EEG Transformer baseline, and Transformer plus handcrafted fusion.
- **Source runs:**
  - Deep learning: `results/deep_learning/cell_01/full/run_20261115T090201955591Z`
  - Feature fusion: `results/deep_learning/cell_01/feature_fusion/full/run_20261115T110540461128Z`
- The existing saved five outer-fold assignments and OOF predictions were used; no model was retrained and no threshold was tuned on test folds.
- Class mapping is Control = 0 and ADHD = 1. Sensitivity is recall for ADHD. All reported metrics were recalculated from saved predictions.
- Pooled OOF metrics and unweighted five-fold mean ± sample SD are distinct summaries. The fold SD is descriptive (ddof=1), not a confidence interval.

## Pooled out-of-fold metrics

Ranking is by pooled balanced accuracy. A duplicate Transformer baseline is retained under its requested label; see the note below.

| Rank | Experiment | Accuracy | Balanced accuracy | ADHD sensitivity | Specificity | Precision | F1 | ROC-AUC |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | Handcrafted features only (Logistic Regression) | 0.9349 | 0.9344 | 0.9266 | 0.9422 | 0.9339 | 0.9302 | 0.9644 |
| 2 | Transformer + handcrafted feature fusion | 0.9295 | 0.9295 | 0.9305 | 0.9286 | 0.9198 | 0.9251 | 0.9668 |
| 3 | EEG-Transformer-inspired | 0.9005 | 0.9009 | 0.9073 | 0.8946 | 0.8835 | 0.8952 | 0.9451 |
| 3 | Raw EEG Transformer baseline (fusion Arm A) | 0.9005 | 0.9009 | 0.9073 | 0.8946 | 0.8835 | 0.8952 | 0.9451 |
| 5 | CNN-LSTM | 0.8734 | 0.8674 | 0.7722 | 0.9626 | 0.9479 | 0.8511 | 0.9445 |
| 6 | EEG Conformer | 0.8590 | 0.8570 | 0.8263 | 0.8878 | 0.8664 | 0.8458 | 0.9339 |
| 7 | EEGNet-inspired | 0.8373 | 0.8292 | 0.7027 | 0.9558 | 0.9333 | 0.8018 | 0.9200 |

## Five-fold mean ± sample standard deviation

Each cell below is the unweighted mean ± sample SD of the five outer-fold metrics; these are not pooled OOF values.

| Experiment | Accuracy | Balanced accuracy | ADHD sensitivity | Specificity | Precision | F1 | ROC-AUC |
| --- | --- | --- | --- | --- | --- | --- | --- |
| EEGNet-inspired | 0.8353 ± 0.0412 | 0.8268 ± 0.0448 | 0.6969 ± 0.1039 | 0.9567 ± 0.0411 | 0.9389 ± 0.0476 | 0.7951 ± 0.0637 | 0.9227 ± 0.0406 |
| CNN-LSTM | 0.8734 ± 0.0665 | 0.8676 ± 0.0689 | 0.7745 ± 0.1142 | 0.9607 ± 0.0542 | 0.9467 ± 0.0758 | 0.8487 ± 0.0843 | 0.9437 ± 0.0780 |
| EEG-Transformer-inspired | 0.9013 ± 0.0323 | 0.9021 ± 0.0346 | 0.9117 ± 0.1041 | 0.8925 ± 0.0704 | 0.8903 ± 0.0699 | 0.8947 ± 0.0392 | 0.9664 ± 0.0289 |
| EEG Conformer | 0.8590 ± 0.0866 | 0.8577 ± 0.0874 | 0.8316 ± 0.1449 | 0.8837 ± 0.1234 | 0.8773 ± 0.1227 | 0.8449 ± 0.0951 | 0.9451 ± 0.0378 |
| Handcrafted features only (Logistic Regression) | 0.9348 ± 0.0411 | 0.9344 ± 0.0388 | 0.9276 ± 0.0434 | 0.9413 ± 0.0836 | 0.9403 ± 0.0822 | 0.9313 ± 0.0395 | 0.9690 ± 0.0330 |
| Raw EEG Transformer baseline (fusion Arm A) | 0.9013 ± 0.0323 | 0.9021 ± 0.0346 | 0.9117 ± 0.1041 | 0.8925 ± 0.0704 | 0.8903 ± 0.0699 | 0.8947 ± 0.0392 | 0.9664 ± 0.0289 |
| Transformer + handcrafted feature fusion | 0.9293 ± 0.0257 | 0.9297 ± 0.0240 | 0.9332 ± 0.0535 | 0.9262 ± 0.0701 | 0.9236 ± 0.0677 | 0.9256 ± 0.0241 | 0.9775 ± 0.0218 |

## All five outer-fold results

| Experiment | Fold | Epochs | Accuracy | Balanced accuracy | ADHD sensitivity | Specificity | Precision | F1 | ROC-AUC |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| EEGNet-inspired | 0 | 119 | 0.8487 | 0.8462 | 0.8036 | 0.8889 | 0.8654 | 0.8333 | 0.9359 |
| EEGNet-inspired | 1 | 119 | 0.8992 | 0.8938 | 0.8036 | 0.9841 | 0.9783 | 0.8824 | 0.9674 |
| EEGNet-inspired | 2 | 105 | 0.7905 | 0.7768 | 0.5714 | 0.9821 | 0.9655 | 0.7179 | 0.9501 |
| EEGNet-inspired | 3 | 105 | 0.8190 | 0.8074 | 0.6327 | 0.9821 | 0.9688 | 0.7654 | 0.8841 |
| EEGNet-inspired | 4 | 105 | 0.8190 | 0.8099 | 0.6735 | 0.9464 | 0.9167 | 0.7765 | 0.8761 |
| CNN-LSTM | 0 | 119 | 0.8908 | 0.8839 | 0.7679 | 1.0000 | 1.0000 | 0.8687 | 0.9583 |
| CNN-LSTM | 1 | 119 | 0.8571 | 0.8482 | 0.6964 | 1.0000 | 1.0000 | 0.8211 | 0.9850 |
| CNN-LSTM | 2 | 105 | 0.9524 | 0.9490 | 0.8980 | 1.0000 | 1.0000 | 0.9462 | 0.9989 |
| CNN-LSTM | 3 | 105 | 0.8952 | 0.8941 | 0.8776 | 0.9107 | 0.8958 | 0.8866 | 0.9694 |
| CNN-LSTM | 4 | 105 | 0.7714 | 0.7628 | 0.6327 | 0.8929 | 0.8378 | 0.7209 | 0.8069 |
| EEG-Transformer-inspired | 0 | 119 | 0.8992 | 0.9008 | 0.9286 | 0.8730 | 0.8667 | 0.8966 | 0.9674 |
| EEG-Transformer-inspired | 1 | 119 | 0.8739 | 0.8661 | 0.7321 | 1.0000 | 1.0000 | 0.8454 | 0.9694 |
| EEG-Transformer-inspired | 2 | 105 | 0.9429 | 0.9464 | 1.0000 | 0.8929 | 0.8909 | 0.9423 | 0.9880 |
| EEG-Transformer-inspired | 3 | 105 | 0.8667 | 0.8712 | 0.9388 | 0.8036 | 0.8070 | 0.8679 | 0.9180 |
| EEG-Transformer-inspired | 4 | 105 | 0.9238 | 0.9260 | 0.9592 | 0.8929 | 0.8868 | 0.9216 | 0.9894 |
| EEG Conformer | 0 | 119 | 0.8824 | 0.8780 | 0.8036 | 0.9524 | 0.9375 | 0.8654 | 0.9427 |
| EEG Conformer | 1 | 119 | 0.8319 | 0.8224 | 0.6607 | 0.9841 | 0.9737 | 0.7872 | 0.9493 |
| EEG Conformer | 2 | 105 | 0.9714 | 0.9707 | 0.9592 | 0.9821 | 0.9792 | 0.9691 | 0.9993 |
| EEG Conformer | 3 | 105 | 0.8762 | 0.8839 | 1.0000 | 0.7679 | 0.7903 | 0.8829 | 0.9413 |
| EEG Conformer | 4 | 105 | 0.7333 | 0.7334 | 0.7347 | 0.7321 | 0.7059 | 0.7200 | 0.8929 |
| Handcrafted features only (Logistic Regression) | 0 | 119 | 0.9412 | 0.9425 | 0.9643 | 0.9206 | 0.9153 | 0.9391 | 0.9773 |
| Handcrafted features only (Logistic Regression) | 1 | 119 | 0.9328 | 0.9286 | 0.8571 | 1.0000 | 1.0000 | 0.9231 | 0.9824 |
| Handcrafted features only (Logistic Regression) | 2 | 105 | 0.9714 | 0.9707 | 0.9592 | 0.9821 | 0.9792 | 0.9691 | 0.9825 |
| Handcrafted features only (Logistic Regression) | 3 | 105 | 0.8667 | 0.8712 | 0.9388 | 0.8036 | 0.8070 | 0.8679 | 0.9107 |
| Handcrafted features only (Logistic Regression) | 4 | 105 | 0.9619 | 0.9592 | 0.9184 | 1.0000 | 1.0000 | 0.9574 | 0.9920 |
| Raw EEG Transformer baseline (fusion Arm A) | 0 | 119 | 0.8992 | 0.9008 | 0.9286 | 0.8730 | 0.8667 | 0.8966 | 0.9674 |
| Raw EEG Transformer baseline (fusion Arm A) | 1 | 119 | 0.8739 | 0.8661 | 0.7321 | 1.0000 | 1.0000 | 0.8454 | 0.9694 |
| Raw EEG Transformer baseline (fusion Arm A) | 2 | 105 | 0.9429 | 0.9464 | 1.0000 | 0.8929 | 0.8909 | 0.9423 | 0.9880 |
| Raw EEG Transformer baseline (fusion Arm A) | 3 | 105 | 0.8667 | 0.8712 | 0.9388 | 0.8036 | 0.8070 | 0.8679 | 0.9180 |
| Raw EEG Transformer baseline (fusion Arm A) | 4 | 105 | 0.9238 | 0.9260 | 0.9592 | 0.8929 | 0.8868 | 0.9216 | 0.9894 |
| Transformer + handcrafted feature fusion | 0 | 119 | 0.9328 | 0.9315 | 0.9107 | 0.9524 | 0.9444 | 0.9273 | 0.9804 |
| Transformer + handcrafted feature fusion | 1 | 119 | 0.9328 | 0.9286 | 0.8571 | 1.0000 | 1.0000 | 0.9231 | 0.9870 |
| Transformer + handcrafted feature fusion | 2 | 105 | 0.9429 | 0.9464 | 1.0000 | 0.8929 | 0.8909 | 0.9423 | 0.9902 |
| Transformer + handcrafted feature fusion | 3 | 105 | 0.8857 | 0.8903 | 0.9592 | 0.8214 | 0.8246 | 0.8868 | 0.9391 |
| Transformer + handcrafted feature fusion | 4 | 105 | 0.9524 | 0.9515 | 0.9388 | 0.9643 | 0.9583 | 0.9485 | 0.9909 |

## Confusion matrices and ROC-AUC

Confusion matrix counts are pooled OOF values with rows = true [Control, ADHD] and columns = predicted [Control, ADHD]. TN/FP/FN/TP correspond to that ordering.

| Experiment | TN | FP (Control → ADHD) | FN (ADHD → Control) | TP |
| --- | --- | --- | --- | --- |
| EEGNet-inspired | 281 | 13 | 77 | 182 |
| CNN-LSTM | 283 | 11 | 59 | 200 |
| EEG-Transformer-inspired | 263 | 31 | 24 | 235 |
| EEG Conformer | 261 | 33 | 45 | 214 |
| Handcrafted features only (Logistic Regression) | 277 | 17 | 19 | 240 |
| Raw EEG Transformer baseline (fusion Arm A) | 263 | 31 | 24 | 235 |
| Transformer + handcrafted feature fusion | 273 | 21 | 18 | 241 |

By pooled OOF ROC-AUC, the leader is **Transformer + handcrafted feature fusion** (0.9668); the ROC plot compares saved OOF curves. The pooled AUC ranking is separate from the balanced-accuracy ranking.

![Pooled OOF ROC curves](../results/deep_learning/cell_01/final_comparison/plots/pooled_roc_curves.png)

![Pooled OOF confusion matrices](../results/deep_learning/cell_01/final_comparison/plots/pooled_confusion_matrices.png)

## Model ranking and interpretation

- **Highest pooled balanced accuracy:** Handcrafted features only (Logistic Regression) (0.9344). **Highest pooled ROC-AUC:** Transformer + handcrafted feature fusion (0.9668). These criteria do not identify one universally superior model.
- Among the four raw-EEG deep architectures, EEG-Transformer-inspired has the highest pooled balanced accuracy (0.9009); handcrafted-only Logistic Regression (0.9344) and fusion (0.9295) are higher on this comparison. This is descriptive evidence for these saved splits only.
- Handcrafted-only Logistic Regression versus the raw Transformer reference: pooled accuracy +0.0344, balanced accuracy +0.0335, sensitivity +0.0193, specificity +0.0476, precision +0.0504, F1 +0.0350, and ROC-AUC +0.0193. All seven pooled point metrics and all seven fold means are higher for this handcrafted-only arm than for the raw Transformer reference in these saved splits; this does not establish a general advantage for handcrafted features.
- Transformer + handcrafted fusion versus the raw Transformer reference: pooled accuracy +0.0289, balanced accuracy +0.0286, sensitivity +0.0232, specificity +0.0340, precision +0.0364, F1 +0.0299, and ROC-AUC +0.0218. All seven pooled point metrics and all seven fold means are higher for fusion than the raw Transformer reference in this run; with five folds, this is descriptive rather than inferential.
- Against handcrafted-only Logistic Regression, fusion changes pooled balanced accuracy by -0.0049, sensitivity by +0.0039, specificity by -0.0136, and ROC-AUC by +0.0025. Thus fusion's relative advantage depends on metric: it is not uniformly better than the handcrafted-only arm.
- **Sensitivity/specificity trade-off:** pooled ADHD sensitivity is highest for Transformer + handcrafted feature fusion (0.9305); specificity is highest for CNN-LSTM (0.9626). At the fixed 0.5 threshold, false-positive errors exceed false negatives for EEG-Transformer-inspired, Raw EEG Transformer baseline (fusion Arm A), Transformer + handcrafted feature fusion, while false negatives exceed false positives for EEGNet-inspired, CNN-LSTM, EEG Conformer, Handcrafted features only (Logistic Regression). These are descriptive classification trade-offs, not diagnostic recommendations.
- The EEG-Transformer-inspired model and the fusion-run Raw EEG Transformer baseline have identical saved OOF rows, including probabilities. They are shown under both requested names but are one baseline result, not independent evidence.

![Fold-mean sensitivity and specificity](../results/deep_learning/cell_01/final_comparison/plots/sensitivity_specificity_fold_means.png)

![Pooled metric comparison](../results/deep_learning/cell_01/final_comparison/plots/pooled_metric_comparison.png)

![Fold balanced accuracy and ROC-AUC](../results/deep_learning/cell_01/final_comparison/plots/fold_metric_means.png)

## Training-history and variability evidence

For neural models, a final validation loss above the best validation loss together with training loss below its best-validation-epoch value is a descriptive pattern consistent with possible late-epoch overfitting. It is not proof of overfitting. The logistic-regression estimator has no epoch-wise history.

| Experiment | Folds final validation > best | Folds training loss fell after best | Interpretation |
| --- | --- | --- | --- |
| EEGNet-inspired | 5/5 | 5/5 | 5/5 folds show both indicators |
| CNN-LSTM | 5/5 | 4/5 | 4/5 folds show both indicators |
| EEG-Transformer-inspired | 5/5 | 5/5 | 5/5 folds show both indicators |
| EEG Conformer | 5/5 | 5/5 | 5/5 folds show both indicators |
| Handcrafted features only (Logistic Regression) | — | — | Not applicable: classical fit; no epoch-wise training history. |
| Raw EEG Transformer baseline (fusion Arm A) | 5/5 | 5/5 | 5/5 folds show both indicators |
| Transformer + handcrafted feature fusion | 5/5 | 5/5 | 5/5 folds show both indicators |

Variation across only five folds is substantial for some metrics, especially sensitivity/specificity for several neural models. Fold SDs are descriptive and do not quantify uncertainty for new participants. The scatter plot shows fold means with sample-SD error bars.

## Existing-report checks

The established training, error-analysis, and fusion reports were numerically checked against metrics independently recomputed from saved OOF predictions.

| Report | Values/cells checked | Result |
| --- | --- | --- |
| deep_learning_training_report.md | 28 | PASS |
| deep_learning_error_analysis.md | 24 | PASS |
| deep_feature_fusion_report.md | 12 | PASS |
| deep_feature_fusion_report.md | 28 | PASS |
| deep_learning_error_analysis.md | 16 | PASS |

## Limitations

- Only Cell 1 was used in this comparison.
- Participant identities were not independently verified. Group IDs are provisional filename/storage-slot identifiers, not verified participant IDs.
- Consequently, the saved group folds do not establish participant-independent generalization.
- Results do not establish clinical diagnostic performance or utility.
- ADHD subtype classification is not part of this objective.
- Five-fold summaries are descriptive; no significance testing or universal ranking is supported by these saved experiments.
- The EEG-Transformer-inspired model and the fusion-run Raw EEG Transformer baseline have identical saved OOF rows, including probabilities. They are shown under both requested names but are one baseline result, not independent evidence.

## Reproducibility and generated artifacts

All source prediction, metric, history, and report artifacts were left unchanged. This comparison creates a separate JSON, CSV tables, and plots under `results/deep_learning/cell_01/final_comparison/`.
