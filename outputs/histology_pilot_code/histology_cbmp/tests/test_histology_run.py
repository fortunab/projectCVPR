"""Protocol checks using tiny generated fixtures, never research data."""
from __future__ import annotations

import csv
import hashlib
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image

from histology_cbmp import run
from histology_cbmp.data import CLASS_NAMES


class SplitProtocolTests(unittest.TestCase):
    @staticmethod
    def rows():
        return [
            {"sample_id": f"{name}/tile-{group}.tif", "label": label, "class_name": name,
             "sha256": f"bytes-{label}-{group}", "pixel_sha256": f"pixels-{label}-{group}"}
            for label, name in enumerate(CLASS_NAMES) for group in range(10)
        ]

    def assert_disjoint(self, rows, split):
        membership = [int(i) for indices in split.values() for i in indices]
        self.assertEqual(sorted(membership), list(range(len(rows))))
        for first, second in (("train", "validation"), ("train", "test"), ("validation", "test")):
            for key in ("sha256", "pixel_sha256"):
                self.assertTrue({rows[int(i)][key] for i in split[first]}.isdisjoint(
                    {rows[int(i)][key] for i in split[second]}))

    def test_splits_are_stable_stratified_and_sixty_twenty_twenty(self):
        rows = self.rows()
        split = run.make_splits(rows, seed=17)
        repeated = run.make_splits(rows, seed=17)
        self.assertEqual({name: len(indices) for name, indices in split.items()},
                         {"train": 48, "validation": 16, "test": 16})
        for name, indices in split.items():
            np.testing.assert_array_equal(indices, repeated[name])
            counts = np.bincount([rows[int(i)]["label"] for i in indices], minlength=len(CLASS_NAMES))
            np.testing.assert_array_equal(counts, np.full(len(CLASS_NAMES), 6 if name == "train" else 2))
        self.assert_disjoint(rows, split)

    def test_pixel_identical_different_encodings_share_one_partition(self):
        rows = self.rows()
        duplicate = {**rows[0], "sample_id": "01_TUMOR/reencoded.png", "sha256": "different-file-bytes"}
        rows.append(duplicate)
        split = run.make_splits(rows, seed=17)
        original_partition = next(name for name, indices in split.items() if 0 in indices)
        self.assertIn(len(rows) - 1, split[original_partition])
        self.assert_disjoint(rows, split)

    def test_pixel_identical_conflicting_labels_fail(self):
        rows = self.rows()
        rows.append({**rows[0], "sample_id": "02_STROMA/conflicting.png", "sha256": "different-file-bytes",
                     "label": 1, "class_name": CLASS_NAMES[1]})
        with self.assertRaises(ValueError):
            run.make_splits(rows, seed=17)

    def test_manifest_fingerprint_includes_pixel_hash(self):
        rows = self.rows()
        original = run.manifest_digest(rows)
        changed = [dict(row) for row in rows]
        changed[0]["pixel_sha256"] = "different-decoded-pixels"
        self.assertNotEqual(original, run.manifest_digest(changed))

    def test_exported_splits_include_pixel_identity(self):
        rows = self.rows()
        split = run.make_splits(rows)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run.save_splits(rows, split, root)
            with (root / "splits.csv").open(newline="", encoding="utf-8") as stream:
                exported = list(csv.DictReader(stream))
        self.assertEqual(len(exported), len(rows))
        by_id = {row["sample_id"]: row for row in exported}
        for row in rows:
            self.assertEqual(by_id[row["sample_id"]]["pixel_sha256"], row["pixel_sha256"])
            self.assertIn(by_id[row["sample_id"]]["split"], {"train", "validation", "test"})


class FileVerificationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.image = self.root / "tiny-fixture.png"
        Image.new("RGB", (8, 8), (12, 34, 56)).save(self.image)
        self.rows = [{"sample_id": "tiny-fixture.png", "image_path": str(self.image),
                      "sha256": hashlib.sha256(self.image.read_bytes()).hexdigest()}]

    def test_current_files_matching_audit_are_accepted(self):
        run.verify_manifest_files(self.rows)

    def test_changed_file_is_rejected_against_audited_hash(self):
        Image.new("RGB", (8, 8), (90, 80, 70)).save(self.image)
        with self.assertRaises(ValueError):
            run.verify_manifest_files(self.rows)

    def test_mutated_file_blocks_cache_hit_before_torch_import(self):
        self.image.write_bytes(b"mutated after data audit")
        cache = self.root / "existing-feature-cache.npz"
        cache.write_bytes(b"cache must not be opened before image verification")
        # Import blocking makes this an inexpensive preflight test: a stale
        # image must fail before either importing torch or reading a cache.
        with patch.dict(sys.modules, {"torch": None, "torchvision": None}):
            with self.assertRaises(ValueError):
                run.extract_features(self.rows, cache, self.root / "unused-weights.pth")


class CalibrationTests(unittest.TestCase):
    def test_scalar_temperature_is_positive_and_preserves_argmax(self):
        rng = np.random.default_rng(71)
        scores = rng.normal(size=(40, len(CLASS_NAMES)))
        labels = np.arange(40) % len(CLASS_NAMES)
        temperature = run.select_temperature(scores, labels)
        self.assertTrue(np.isfinite(temperature))
        self.assertGreater(temperature, 0)
        probabilities = run.softmax(scores, temperature)
        np.testing.assert_array_equal(probabilities.argmax(axis=1), scores.argmax(axis=1))
        np.testing.assert_allclose(probabilities.sum(axis=1), 1.0)


if __name__ == "__main__":
    unittest.main()
