"""Tests for the Cell 1 EEG and handcrafted feature-fusion experiment."""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from src.deep_fusion import EEGTransformerFusion
from src.features import FeatureDataset
from src.modeling_data import ModelingData
from src.train_deep_fusion import (
    _build_fusion_model,
    _load_fusion_checkpoint,
    _save_fusion_checkpoint,
    _train_logistic_fold,
    fit_feature_normalizer,
    validate_feature_alignment,
    validate_group_partition,
)
from src.train_deep_models import (
    fit_channel_normalizer,
    group_aware_inner_split,
    validate_prediction_coverage,
)


def make_aligned_data():
    rng = np.random.default_rng(42)
    labels = []
    groups = []
    folds = []
    epochs = []
    features = []
    for fold in range(5):
        for label in (0, 1):
            labels.append(label)
            groups.append(f"class-{label}-fold-{fold}")
            folds.append(fold)
            epochs.append(rng.normal(loc=label * 0.1, size=(1024, 2)))
            features.append(rng.normal(loc=label * 0.2, size=40))
    labels = np.asarray(labels, dtype=np.int64)
    groups = np.asarray(groups)
    folds = np.asarray(folds, dtype=np.int64)
    data = ModelingData(
        X=np.asarray(epochs, dtype=np.float64),
        y=labels,
        groups=groups,
        folds=folds,
        class_mapping={"Control": 0, "ADHD": 1},
    )
    feature_dataset = FeatureDataset(
        X=np.asarray(features, dtype=np.float64),
        feature_names=tuple(f"feature_{index}" for index in range(40)),
        y=labels.copy(),
        groups=groups.copy(),
        folds=folds.copy(),
        class_mapping={"Control": 0, "ADHD": 1},
    )
    return data, feature_dataset


class FusionModelTests(unittest.TestCase):
    def test_model_returns_logits_and_separate_representations(self):
        model = EEGTransformerFusion(
            transformer_config={
                "embed_dim": 8,
                "num_heads": 2,
                "num_layers": 1,
                "feedforward_dim": 16,
                "token_stride": 16,
            },
            fusion_dim=4,
            head_hidden_dim=4,
        )
        raw = torch.randn(3, 2, 1024)
        features = torch.randn(3, 40)

        logits = model(raw, features)
        representations = model.forward_representations(raw, features)

        self.assertEqual(logits.shape, (3, 2))
        self.assertTrue(torch.isfinite(logits).all().item())
        self.assertEqual(representations["raw_eeg"].shape, (3, 8))
        self.assertEqual(representations["handcrafted_features"].shape, (3, 40))
        self.assertEqual(representations["raw_projected"].shape, (3, 4))
        self.assertEqual(representations["handcrafted_projected"].shape, (3, 4))
        self.assertEqual(representations["fused"].shape, (3, 8))

    def test_batch_size_one_forward_and_backward(self):
        model = EEGTransformerFusion(
            transformer_config={
                "embed_dim": 8,
                "num_heads": 2,
                "num_layers": 1,
                "feedforward_dim": 16,
                "token_stride": 16,
            },
            fusion_dim=4,
            head_hidden_dim=4,
        )
        logits = model(torch.randn(1, 2, 1024), torch.randn(1, 40))
        loss = torch.nn.functional.cross_entropy(logits, torch.tensor([1]))
        loss.backward()

        self.assertEqual(logits.shape, (1, 2))
        self.assertTrue(torch.isfinite(logits).all().item())
        self.assertTrue(
            any(parameter.grad is not None for parameter in model.parameters())
        )

    def test_invalid_modality_shapes_are_rejected(self):
        model = EEGTransformerFusion(
            transformer_config={
                "embed_dim": 8,
                "num_heads": 2,
                "num_layers": 1,
                "feedforward_dim": 16,
            }
        )
        with self.assertRaisesRegex(ValueError, "Raw EEG dimensions"):
            model(torch.randn(2, 3, 1024), torch.randn(2, 40))
        with self.assertRaisesRegex(ValueError, "one 40-value row"):
            model(torch.randn(2, 2, 1024), torch.randn(2, 39))
        invalid_features = torch.randn(2, 40)
        invalid_features[0, 0] = torch.nan
        with self.assertRaisesRegex(ValueError, "non-finite"):
            model(torch.randn(2, 2, 1024), invalid_features)

    def test_backbone_freezing_is_configurable(self):
        model = EEGTransformerFusion(
            transformer_config={
                "embed_dim": 8,
                "num_heads": 2,
                "num_layers": 1,
                "feedforward_dim": 16,
            },
            freeze_transformer_backbone=True,
        )
        self.assertTrue(
            all(
                not parameter.requires_grad
                for parameter in model.backbone.parameters()
            )
        )
        self.assertTrue(
            any(parameter.requires_grad for parameter in model.classifier.parameters())
        )
        model.train()
        self.assertFalse(model.backbone.training)
        self.assertTrue(model.classifier.training)


class AlignmentAndNormalizationTests(unittest.TestCase):
    def test_raw_and_handcrafted_metadata_must_match_exactly(self):
        data, features = make_aligned_data()
        validate_feature_alignment(data, features)

        features.y[0] = 1 - features.y[0]
        with self.assertRaisesRegex(ValueError, "y are not exactly aligned"):
            validate_feature_alignment(data, features)

    def test_training_only_feature_normalization(self):
        training = np.arange(120, dtype=np.float64).reshape(3, 40)
        training[:, 0] = 7.0
        validation = np.full((1, 40), 1e6)
        test = np.full((1, 40), -1e6)

        normalizer = fit_feature_normalizer(training)
        normalized_validation = normalizer.transform(validation)
        normalized_test = normalizer.transform(test)

        np.testing.assert_allclose(normalizer.mean, training.mean(axis=0))
        np.testing.assert_allclose(
            normalizer.scale[1:],
            training.std(axis=0)[1:],
        )
        self.assertEqual(normalizer.constant_feature_indices, (0,))
        self.assertEqual(float(normalizer.transform(training)[:, 0].max()), 0.0)
        self.assertGreater(float(np.abs(normalized_validation).mean()), 1e3)
        self.assertGreater(float(np.abs(normalized_test).mean()), 1e3)

    def test_rejects_nonfinite_or_misaligned_feature_matrices(self):
        data, features = make_aligned_data()
        features.X[0, 0] = np.nan
        with self.assertRaisesRegex(ValueError, "non-finite"):
            validate_feature_alignment(data, features)


class SplitAndCheckpointTests(unittest.TestCase):
    def test_group_split_separates_outer_test_and_inner_validation(self):
        data, _ = make_aligned_data()
        outer_training = np.flatnonzero(data.folds != 0)
        test_indices = np.flatnonzero(data.folds == 0)
        relative_training, relative_validation = group_aware_inner_split(
            data.groups[outer_training],
            data.y[outer_training],
            validation_fraction=0.25,
            seed=42,
        )
        training_indices = outer_training[relative_training]
        validation_indices = outer_training[relative_validation]

        record = validate_group_partition(
            data,
            training_indices,
            validation_indices,
            test_indices,
        )
        self.assertFalse(
            set(record["training_group_ids"])
            & set(record["validation_group_ids"])
        )
        self.assertFalse(
            set(record["training_group_ids"]) & set(record["test_group_ids"])
        )
        self.assertFalse(
            set(record["validation_group_ids"]) & set(record["test_group_ids"])
        )
        self.assertFalse(record["group_ids_are_verified_participant_ids"])

    def test_group_partition_rejects_outer_test_leakage(self):
        data, _ = make_aligned_data()
        outer_training = np.flatnonzero(data.folds != 0)
        test_indices = np.flatnonzero(data.folds == 0)
        relative_training, relative_validation = group_aware_inner_split(
            data.groups[outer_training],
            data.y[outer_training],
            validation_fraction=0.25,
            seed=42,
        )
        training_indices = np.append(outer_training[relative_training], test_indices[0])
        validation_indices = outer_training[relative_validation]

        with self.assertRaisesRegex(RuntimeError, "outer-test group"):
            validate_group_partition(
                data,
                training_indices,
                validation_indices,
                test_indices,
            )

    def test_fusion_checkpoint_round_trip_preserves_weights_and_normalizers(self):
        torch.manual_seed(42)
        model = EEGTransformerFusion(
            transformer_config={
                "embed_dim": 8,
                "num_heads": 2,
                "num_layers": 1,
                "feedforward_dim": 16,
                "token_stride": 16,
            },
            fusion_dim=4,
            head_hidden_dim=4,
        )
        raw_normalizer = fit_channel_normalizer(
            np.random.default_rng(1).normal(size=(3, 1024, 2))
        )
        feature_normalizer = fit_feature_normalizer(
            np.random.default_rng(2).normal(size=(3, 40))
        )
        model.eval()
        raw = torch.randn(1, 2, 1024)
        features = torch.randn(1, 40)
        expected = model(raw, features).detach()
        parameters = {
            "models": {
                "eeg_transformer": {
                    "embed_dim": 8,
                    "num_heads": 2,
                    "num_layers": 1,
                    "feedforward_dim": 16,
                    "token_stride": 16,
                }
            }
        }
        fusion_configuration = {
            "feature_count": 40,
            "projection_dim": 4,
            "classifier_hidden_dim": 4,
            "dropout": 0.3,
            "freeze_transformer_backbone": False,
        }
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "fusion.pt"
            _save_fusion_checkpoint(
                path,
                model,
                raw_normalizer,
                feature_normalizer,
                {"outer_fold": 0},
            )
            restored = _build_fusion_model(
                parameters,
                fusion_configuration,
                initial_checkpoint=None,
                device=torch.device("cpu"),
            )
            checkpoint = _load_fusion_checkpoint(
                path,
                restored,
                torch.device("cpu"),
            )
            restored.eval()
            actual = restored(raw, features).detach()

        self.assertEqual(checkpoint["metadata"]["outer_fold"], 0)
        np.testing.assert_array_equal(
            checkpoint["raw_normalization_mean"],
            raw_normalizer.mean,
        )
        np.testing.assert_array_equal(
            checkpoint["feature_normalization_mean"],
            feature_normalizer.mean,
        )
        torch.testing.assert_close(actual, expected)

    def test_prediction_coverage_requires_one_oof_prediction(self):
        validate_prediction_coverage(np.ones(5, dtype=np.int64))
        with self.assertRaisesRegex(ValueError, "missing indices"):
            validate_prediction_coverage(np.asarray([1, 0, 1]))
        with self.assertRaisesRegex(ValueError, "duplicated indices"):
            validate_prediction_coverage(np.asarray([1, 2, 1]))


class HandcraftedBaselineTests(unittest.TestCase):
    def test_logistic_baseline_saves_and_reloads_fold_checkpoint(self):
        data, features = make_aligned_data()
        outer_training = np.flatnonzero(data.folds != 0)
        test_indices = np.flatnonzero(data.folds == 0)
        relative_training, relative_validation = group_aware_inner_split(
            data.groups[outer_training],
            data.y[outer_training],
            validation_fraction=0.25,
            seed=42,
        )
        training_indices = outer_training[relative_training]
        validation_indices = outer_training[relative_validation]
        split = {
            "training_indices": training_indices,
            "validation_indices": validation_indices,
            "test_indices": test_indices,
        }
        configuration = {
            "C": 1.0,
            "maximum_iterations": 200,
            "class_weighting_uses_training_fold": True,
            "random_state": 42,
        }
        training_config = {
            "class_weighting": {
                "enabled": True,
                "imbalance_ratio_threshold": 1.1,
            }
        }
        with tempfile.TemporaryDirectory() as temporary_directory:
            probabilities, metrics = _train_logistic_fold(
                data,
                features,
                split,
                0,
                configuration,
                training_config,
                Path(temporary_directory) / "fold",
            )
            checkpoint_path = (
                Path(temporary_directory)
                / "fold"
                / "classical_checkpoint.joblib"
            )
            self.assertTrue(checkpoint_path.exists())
            self.assertEqual(probabilities.shape, (test_indices.size,))
            self.assertTrue(np.isfinite(probabilities).all())
            self.assertIn("confusion_matrix", metrics)
            self.assertIn("roc_auc", metrics)


if __name__ == "__main__":
    unittest.main()
