import sys
import unittest
from collections import Counter
from decimal import Decimal
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from adhd_eeg import grouped_train_validation_test_split


class GroupedEvaluationSplitTests(unittest.TestCase):
    def setUp(self):
        self.sample_ids = ("s0", "s1", "s2", "s3", "s4", "s5", "s6")
        self.group_ids = ("g0", "g0", "g1", "g2", "g2", "g3", "g3")
        self.group_counts = {"train": 2, "validation": 1, "test": 1}
        self.random_seed = 41

    def _split(self, **overrides):
        arguments = {
            "sample_ids": self.sample_ids,
            "group_ids": self.group_ids,
            "group_counts": self.group_counts,
            "random_seed": self.random_seed,
        }
        arguments.update(overrides)
        return grouped_train_validation_test_split(**arguments)

    def test_no_group_overlap_and_complete_sample_coverage(self):
        split = self._split()
        train_groups = set(split.train_group_ids)
        validation_groups = set(split.validation_group_ids)
        test_groups = set(split.test_group_ids)

        self.assertFalse(train_groups & validation_groups)
        self.assertFalse(train_groups & test_groups)
        self.assertFalse(validation_groups & test_groups)
        self.assertEqual(len(train_groups), 2)
        self.assertEqual(len(validation_groups), 1)
        self.assertEqual(len(test_groups), 1)

        assignments = (
            split.train_sample_ids
            + split.validation_sample_ids
            + split.test_sample_ids
        )
        self.assertCountEqual(assignments, self.sample_ids)

        group_by_sample = dict(zip(self.sample_ids, self.group_ids))
        for sample_ids, assigned_groups in (
            (split.train_sample_ids, train_groups),
            (split.validation_sample_ids, validation_groups),
            (split.test_sample_ids, test_groups),
        ):
            self.assertTrue(
                all(group_by_sample[sample_id] in assigned_groups for sample_id in sample_ids)
            )

    def test_same_seed_reproduces_assignments(self):
        self.assertEqual(self._split(), self._split())

    def test_class_counts_are_reported_per_partition_when_labels_are_supplied(self):
        labels = ("class_a", "class_b", "class_a", "class_b", "class_b", "class_a", "class_a")
        split = self._split(labels=labels)
        self.assertIsNotNone(split.class_counts)
        labels_by_sample = dict(zip(self.sample_ids, labels))

        for partition, sample_ids in (
            ("train", split.train_sample_ids),
            ("validation", split.validation_sample_ids),
            ("test", split.test_sample_ids),
        ):
            expected_counts = Counter(labels_by_sample[sample_id] for sample_id in sample_ids)
            self.assertEqual(
                split.class_counts[partition],
                {
                    label: expected_counts.get(label, 0)
                    for label in dict.fromkeys(labels)
                },
            )
            self.assertEqual(
                sum(split.class_counts[partition].values()), len(sample_ids)
            )

    def test_class_counts_are_none_when_labels_are_not_supplied(self):
        self.assertIsNone(self._split().class_counts)

    def test_missing_and_invalid_group_identifiers_are_rejected(self):
        for invalid_group in (None, "", "   ", float("nan"), []):
            with self.subTest(invalid_group=invalid_group):
                group_ids = list(self.group_ids)
                group_ids[0] = invalid_group
                with self.assertRaises((TypeError, ValueError)):
                    self._split(group_ids=group_ids)

    def test_missing_and_duplicate_sample_identifiers_are_rejected(self):
        missing_ids = list(self.sample_ids)
        missing_ids[0] = None
        with self.assertRaisesRegex(ValueError, "must not be missing"):
            self._split(sample_ids=missing_ids)

        duplicate_ids = list(self.sample_ids)
        duplicate_ids[1] = duplicate_ids[0]
        with self.assertRaisesRegex(ValueError, "duplicate sample identifier"):
            self._split(sample_ids=duplicate_ids)

    def test_common_missing_value_identifiers_are_rejected(self):
        missing_values = (
            float("nan"),
            np.float32("nan"),
            Decimal("NaN"),
            np.datetime64("NaT"),
            np.timedelta64("NaT", "ns"),
        )
        for missing_value in missing_values:
            with self.subTest(missing_value=repr(missing_value)):
                sample_ids = list(self.sample_ids)
                sample_ids[0] = missing_value
                with self.assertRaisesRegex(ValueError, "must not be missing"):
                    self._split(sample_ids=sample_ids)

    def test_misaligned_inputs_and_invalid_group_counts_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "equal lengths"):
            self._split(group_ids=self.group_ids[:-1])
        with self.assertRaisesRegex(ValueError, "same length"):
            self._split(labels=("class_a",))
        with self.assertRaisesRegex(ValueError, "exactly train"):
            self._split(group_counts={"train": 2, "test": 2})
        with self.assertRaisesRegex(ValueError, "positive integer"):
            self._split(
                group_counts={"train": 2, "validation": 0, "test": 2}
            )
        with self.assertRaisesRegex(ValueError, "number of unique"):
            self._split(
                group_counts={"train": 1, "validation": 1, "test": 1}
            )


if __name__ == "__main__":
    unittest.main()
