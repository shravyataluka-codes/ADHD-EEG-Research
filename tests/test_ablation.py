"""Synthetic-only tests for the Sprint 7 model registry and matched runner."""

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

try:
    import torch
    import torch.nn as nn
except ImportError:
    torch = None

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

if torch is not None:
    from adhd_eeg import ablation
    from adhd_eeg.ablation import (
        MODEL_VARIANT_REGISTRY,
        MatchedSyntheticConfig,
        ModelVariant,
        build_matched_synthetic_manifest,
        build_model,
        run_matched_synthetic_experiment,
    )
    from adhd_eeg.classifier import (
        CLASS_ORDER,
        EEGConformerClassifier,
        EEGNetBaseline,
        EEGNetConfig,
        POOLING_MEAN,
        POOLING_MEAN_MAX,
    )
    from adhd_eeg.conformer import EXPECTED_CHANNELS, EXPECTED_TIME_STEPS, MODEL_DIM
    from adhd_eeg.training import TrainingConfig


@unittest.skipIf(torch is None, "Install the optional 'pytorch' extra to run model tests")
class ModelVariantTests(unittest.TestCase):
    def test_registry_dispatch_and_rejects_unregistered_variants(self) -> None:
        self.assertEqual(set(MODEL_VARIANT_REGISTRY), {variant.value for variant in ModelVariant})
        self.assertIsInstance(build_model("conformer_reference"), EEGConformerClassifier)
        self.assertIsInstance(build_model(ModelVariant.EEGNET), EEGNetBaseline)
        with self.assertRaisesRegex(ValueError, "unregistered"):
            build_model("unbounded_variant")

    def test_ablation_components_change_only_the_named_component(self) -> None:
        reference = build_model("conformer_reference")
        no_attention = build_model("conformer_no_attention")
        no_conv = build_model("conformer_no_conv")
        mean_only = build_model("conformer_mean_pool")

        reference_keys = set(reference.state_dict())
        attention_keys = {
            key for key in reference_keys if key.startswith("encoder.blocks.") and ".attn." in key
        }
        convolution_keys = {
            key for key in reference_keys if key.startswith("encoder.blocks.") and ".conv." in key
        }
        self.assertTrue(attention_keys)
        self.assertTrue(convolution_keys)
        self.assertEqual(
            reference_keys - set(no_attention.state_dict()), attention_keys
        )
        self.assertEqual(reference_keys - set(no_conv.state_dict()), convolution_keys)
        self.assertEqual(set(no_attention.state_dict()) - reference_keys, set())
        self.assertEqual(set(no_conv.state_dict()) - reference_keys, set())

        for block in no_attention.encoder.blocks:
            self.assertIsNone(block.attn)
            self.assertIsNotNone(block.conv)
        for block in no_conv.encoder.blocks:
            self.assertIsNotNone(block.attn)
            self.assertIsNone(block.conv)
        self.assertEqual(reference.pooling_mode, POOLING_MEAN_MAX)
        self.assertEqual(mean_only.pooling_mode, POOLING_MEAN)
        self.assertEqual(
            set(reference.state_dict()), set(mean_only.state_dict())
        )
        for key in reference.state_dict():
            if key == "head.linear1.weight":
                self.assertEqual(reference.state_dict()[key].shape, (64, 128))
                self.assertEqual(mean_only.state_dict()[key].shape, (64, 64))
            else:
                self.assertEqual(
                    reference.state_dict()[key].shape,
                    mean_only.state_dict()[key].shape,
                    key,
                )
        self.assertEqual(
            mean_only.pool(torch.randn(2, 97, MODEL_DIM)).shape, (2, MODEL_DIM)
        )

    def test_variant_forward_shapes_finiteness_and_gradients(self) -> None:
        signals = torch.randn(2, EXPECTED_TIME_STEPS, EXPECTED_CHANNELS)
        targets = torch.tensor([0, 1], dtype=torch.long)
        for variant in MODEL_VARIANT_REGISTRY:
            with self.subTest(variant=variant):
                model = build_model(variant)
                model.train()
                logits = model(signals)
                self.assertEqual(logits.shape, (2, len(CLASS_ORDER)))
                self.assertTrue(torch.isfinite(logits).all())
                nn.CrossEntropyLoss()(logits, targets).backward()
                gradients = [
                    parameter.grad
                    for parameter in model.parameters()
                    if parameter.requires_grad
                ]
                self.assertTrue(gradients)
                self.assertTrue(all(gradient is not None for gradient in gradients))
                self.assertTrue(
                    all(torch.isfinite(gradient).all() for gradient in gradients)
                )

    def test_parameter_counts_and_eegnet_input_configuration(self) -> None:
        reference = build_model("conformer_reference")
        no_attention = build_model("conformer_no_attention")
        no_conv = build_model("conformer_no_conv")
        mean_only = build_model("conformer_mean_pool")
        eegnet = build_model("eegnet")
        count = lambda model: sum(parameter.numel() for parameter in model.parameters())
        self.assertGreater(count(reference), count(no_attention))
        self.assertGreater(count(reference), count(no_conv))
        self.assertGreater(count(reference), count(mean_only))
        self.assertGreater(count(eegnet), 0)
        self.assertEqual(eegnet.config, EEGNetConfig())
        eegnet.eval()
        with torch.no_grad():
            output = eegnet(torch.randn(1, EXPECTED_TIME_STEPS, EXPECTED_CHANNELS))
        self.assertEqual(output.shape, (1, len(CLASS_ORDER)))
        self.assertTrue(torch.isfinite(output).all())
        with self.assertRaises(ValueError):
            EEGNetConfig(expected_channels=55).validate()

    def test_matched_configuration_and_manifest_validation(self) -> None:
        training = TrainingConfig(max_epochs=1, batch_size=3)
        config = MatchedSyntheticConfig(
            training=training,
            variants=("conformer_reference", "eegnet"),
            participants_per_class=3,
            trials_per_participant=1,
        )
        manifest = build_matched_synthetic_manifest(config)
        self.assertEqual(manifest, build_matched_synthetic_manifest(config))
        self.assertEqual(manifest["protocol_purpose"], "SYNTHETIC_SMOKE_TEST_ONLY")
        self.assertEqual(manifest["result_kind"], "synthetic engineering validation")
        self.assertEqual(manifest["class_order"], list(CLASS_ORDER))
        self.assertEqual(manifest["model_input_fields"], ["signal_tensor"])
        self.assertFalse(manifest["model_selection"]["test_partition_used_for_selection"])
        single_variant_manifest = build_matched_synthetic_manifest(
            MatchedSyntheticConfig(
                training=training,
                variants=("eegnet",),
                participants_per_class=3,
                trials_per_participant=1,
            )
        )
        self.assertEqual(
            manifest["sample_ids_by_partition"],
            single_variant_manifest["sample_ids_by_partition"],
        )
        partitions = manifest["sample_ids_by_partition"]
        self.assertFalse(set(partitions["train"]) & set(partitions["validation"]))
        self.assertFalse(set(partitions["train"]) & set(partitions["test"]))
        self.assertFalse(set(partitions["validation"]) & set(partitions["test"]))
        config_copy = MatchedSyntheticConfig.from_dict(config.to_dict())
        self.assertEqual(config, config_copy)
        with self.assertRaises(ValueError):
            MatchedSyntheticConfig(variants=("conformer_reference", "conformer_reference"))
        with self.assertRaises(ValueError):
            MatchedSyntheticConfig(variants=("real_data_model",))
        with self.assertRaises(ValueError):
            MatchedSyntheticConfig(protocol_purpose="REAL_COHORT_TRAINING")
        with self.assertRaises(ValueError):
            MatchedSyntheticConfig(participants_per_class=2)

    def test_runner_uses_validation_only_for_selection_and_returns_test_artifact(self) -> None:
        config = MatchedSyntheticConfig(
            training=TrainingConfig(
                seed=19,
                max_epochs=1,
                batch_size=3,
                early_stopping_patience=None,
            ),
            variants=("eegnet",),
            participants_per_class=3,
            trials_per_participant=1,
        )
        fixture = ablation._build_fixture(config)
        validation_indices = ablation._indices_for_samples(
            fixture, fixture.split.validation_sample_ids
        )
        test_indices = ablation._indices_for_samples(
            fixture, fixture.split.test_sample_ids
        )
        validation_sample_counts: list[int] = []
        training_tensor_shapes: list[tuple[tuple[int, ...], tuple[int, ...]]] = []
        original_evaluate = ablation.evaluate_epoch
        original_train = ablation.train_one_epoch

        def capture_validation(model, loader, device=None):
            validation_sample_counts.append(len(loader.dataset))
            self.assertIsInstance(loader.dataset, torch.utils.data.TensorDataset)
            self.assertTrue(
                torch.equal(
                    loader.dataset.tensors[0], fixture.signals[validation_indices]
                )
            )
            self.assertFalse(
                torch.equal(loader.dataset.tensors[0], fixture.signals[test_indices])
            )
            return original_evaluate(model, loader, device=device)

        def capture_training(model, loader, optimizer, **kwargs):
            self.assertIsInstance(loader.dataset, torch.utils.data.TensorDataset)
            self.assertEqual(len(loader.dataset.tensors), 2)
            signals, targets = loader.dataset.tensors
            self.assertEqual(signals.dtype, torch.float32)
            self.assertEqual(signals.shape[1:], (EXPECTED_TIME_STEPS, EXPECTED_CHANNELS))
            self.assertEqual(targets.dtype, torch.long)
            training_tensor_shapes.append((tuple(signals.shape), tuple(targets.shape)))
            return original_train(model, loader, optimizer, **kwargs)

        with (
            patch.object(ablation, "evaluate_epoch", side_effect=capture_validation),
            patch.object(ablation, "train_one_epoch", side_effect=capture_training),
        ):
            result = run_matched_synthetic_experiment(config)

        self.assertEqual(len(result.variants), 1)
        run = result.variants[0]
        self.assertEqual(run.variant, "eegnet")
        self.assertGreater(run.parameter_count, 0)
        self.assertEqual(run.parameter_count, run.trainable_parameter_count)
        self.assertEqual(run.compute_cost_measurement, "not_measured")
        self.assertEqual(run.selection_partition, "validation")
        self.assertEqual(run.selected_epoch, 1)
        self.assertEqual(validation_sample_counts, [3])
        self.assertEqual(len(training_tensor_shapes), 1)
        self.assertEqual(training_tensor_shapes[0][0][1:], (EXPECTED_TIME_STEPS, EXPECTED_CHANNELS))
        self.assertEqual(len(training_tensor_shapes[0][1]), 1)
        manifest = result.manifest
        sample_ids = manifest["sample_ids_by_partition"]
        all_ids = set(sample_ids["train"] + sample_ids["validation"] + sample_ids["test"])
        self.assertEqual(len(all_ids), 9)
        self.assertEqual(len(run.predictions.records), len(sample_ids["test"]))
        self.assertTrue(
            all(
                record.sample_id in set(sample_ids["test"])
                for record in run.predictions.records
            )
        )
        self.assertTrue(
            all(
                record.participant_id.startswith("synthetic-fixture:")
                for record in run.predictions.records
            )
        )


if __name__ == "__main__":
    unittest.main()
