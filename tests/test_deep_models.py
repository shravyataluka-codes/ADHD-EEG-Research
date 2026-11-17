"""Synthetic shape, gradient, and reproducibility tests for deep EEG models."""

import unittest

import torch

from src.deep_models import (
    CNNLSTM,
    EEGConformer,
    EEGNet,
    EEGTransformer,
    MODEL_REGISTRY,
)


MODEL_CLASSES = (EEGNet, CNNLSTM, EEGTransformer, EEGConformer)


class DeepModelTests(unittest.TestCase):
    def test_all_architectures_instantiate_and_return_finite_logits(self):
        for model_class in MODEL_CLASSES:
            with self.subTest(model=model_class.__name__):
                model = model_class()
                inputs = torch.randn(3, 2, 1024)
                logits = model(inputs)

                self.assertEqual(logits.shape, (3, 2))
                self.assertTrue(torch.isfinite(logits).all().item())

    def test_backward_pass_produces_finite_gradients(self):
        for model_class in MODEL_CLASSES:
            with self.subTest(model=model_class.__name__):
                model = model_class()
                inputs = torch.randn(3, 2, 1024)
                labels = torch.tensor([0, 1, 0], dtype=torch.long)
                loss = torch.nn.functional.cross_entropy(model(inputs), labels)
                loss.backward()

                gradients = [
                    parameter.grad
                    for parameter in model.parameters()
                    if parameter.requires_grad
                ]
                self.assertTrue(any(gradient is not None for gradient in gradients))
                self.assertTrue(
                    all(
                        gradient is None or torch.isfinite(gradient).all().item()
                        for gradient in gradients
                    )
                )

    def test_batch_size_one_is_supported(self):
        for model_class in MODEL_CLASSES:
            with self.subTest(model=model_class.__name__):
                model = model_class()
                logits = model(torch.randn(1, 2, 1024))

                self.assertEqual(logits.shape, (1, 2))
                self.assertTrue(torch.isfinite(logits).all().item())

    def test_batch_size_one_supports_training_backward(self):
        for model_class in MODEL_CLASSES:
            with self.subTest(model=model_class.__name__):
                model = model_class()
                logits = model(torch.randn(1, 2, 1024))
                loss = torch.nn.functional.cross_entropy(
                    logits,
                    torch.tensor([1], dtype=torch.long),
                )
                loss.backward()

                self.assertTrue(
                    any(parameter.grad is not None for parameter in model.parameters())
                )

    def test_parameter_counts_are_positive_and_reported(self):
        counts = {}
        for model_name, model_class in MODEL_REGISTRY.items():
            model = model_class()
            counts[model_name] = sum(
                parameter.numel()
                for parameter in model.parameters()
                if parameter.requires_grad
            )

        self.assertEqual(set(counts), set(MODEL_REGISTRY))
        self.assertTrue(all(count > 0 for count in counts.values()))
        print("Trainable parameter counts:", counts)

    def test_initialization_is_reproducible_with_fixed_seed(self):
        for model_class in MODEL_CLASSES:
            with self.subTest(model=model_class.__name__):
                torch.manual_seed(2026)
                first = model_class()
                first_state = {
                    key: value.detach().clone()
                    for key, value in first.state_dict().items()
                }

                torch.manual_seed(2026)
                second = model_class()

                self.assertEqual(first_state.keys(), second.state_dict().keys())
                for key, value in second.state_dict().items():
                    self.assertTrue(torch.equal(first_state[key], value))

    def test_rejects_invalid_channel_count_and_sample_count(self):
        for model_class in MODEL_CLASSES:
            with self.subTest(model=model_class.__name__, invalid="channels"):
                with self.assertRaisesRegex(ValueError, "input channel slots"):
                    model_class()(torch.randn(2, 3, 1024))
            with self.subTest(model=model_class.__name__, invalid="samples"):
                with self.assertRaisesRegex(ValueError, "samples per epoch"):
                    model_class()(torch.randn(2, 2, 1000))

    def test_rejects_invalid_constructor_dimensions(self):
        with self.assertRaisesRegex(ValueError, "divisible by num_heads"):
            EEGTransformer(embed_dim=30, num_heads=4)
        with self.assertRaisesRegex(ValueError, "divisible by num_heads"):
            EEGConformer(embed_dim=30, num_heads=4)


if __name__ == "__main__":
    unittest.main()
