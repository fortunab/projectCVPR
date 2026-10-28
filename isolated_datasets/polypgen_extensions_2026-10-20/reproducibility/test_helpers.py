"""Small local portability/safety fixtures; no experimental data or training."""
from __future__ import annotations
import csv
import hashlib
import io
import json
import os
import stat
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch
import numpy as np
import reproduce
from metrics_from_predictions import read_paired


class ReproductionHelpers(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="repro_fixture_", dir=reproduce.HERE)
        self.root = Path(self.temporary.name).resolve()
        assert self.root.is_relative_to(reproduce.HERE.resolve())

    def tearDown(self):
        # Both the absolute temporary root and its workspace containment were checked above.
        self.temporary.cleanup()

    def test_zip_rejects_cross_platform_traversal_and_links(self):
        for name in ("../outside.txt", "C:/outside.txt", "/absolute.txt", "a\\..\\outside.txt"):
            content = io.BytesIO()
            with zipfile.ZipFile(content, "w") as archive:
                archive.writestr(name, b"fixture")
            content.seek(0)
            with zipfile.ZipFile(content) as archive, self.assertRaises(ValueError):
                reproduce.checked_zip_members(archive, self.root / "extract")
        content = io.BytesIO()
        with zipfile.ZipFile(content, "w") as archive:
            info = zipfile.ZipInfo("link")
            info.external_attr = (stat.S_IFLNK | 0o777) << 16
            archive.writestr(info, "../outside")
        content.seek(0)
        with zipfile.ZipFile(content) as archive, self.assertRaises(ValueError):
            reproduce.checked_zip_members(archive, self.root / "extract")

    def test_stage_only_changes_new_copy_checkpoint_locations(self):
        bundle = self.root / "bundle"
        bundle.mkdir()
        (bundle / "sources_manifest.json").write_text("{}")
        for scope in ("baseline", "followups", "extensions", "histology"):
            (bundle / "sources" / scope).mkdir(parents=True)
        ext = bundle / "sources/extensions"
        original4 = 'from pathlib import Path\nMODEL_DIR = Path.home() / ".cache/huggingface/hub/models--facebook--dinov2-base/snapshots" / SNAPSHOT_ID\n'
        original5 = 'def f():\n    weights_path = Path.home()/".cache/torch/hub/checkpoints/resnet18-f37072fd.pth"\n'
        (ext / "exp4_dinov2.py").write_text(original4)
        (ext / "exp5_camelyon.py").write_text(original5)
        archive = self.root / "fixture.zip"
        with zipfile.ZipFile(archive, "w") as stream:
            stream.writestr("PolypGen2021_MultiCenterData_v3/fixture.txt", "fixture")
        metadata = {"archive_bytes": archive.stat().st_size, "archive_sha256": reproduce.sha256(archive)}
        destination = self.root / "fresh_workspace"
        with patch.object(reproduce, "HERE", bundle), patch.object(reproduce, "verify_bundle", return_value=metadata), \
                patch.object(reproduce.shutil, "disk_usage", return_value=type("D", (), {"free": 100 * 1024**3})()):
            reproduce.stage(destination, archive)
            with self.assertRaises(FileExistsError):
                reproduce.stage(destination, archive)
        self.assertEqual((ext / "exp4_dinov2.py").read_text(), original4)
        self.assertEqual((ext / "exp5_camelyon.py").read_text(), original5)
        staged = destination / "isolated_datasets" / reproduce.EXT_NAME
        self.assertIn('os.environ["CBMP_DINO_MODEL_DIR"]', (staged / "exp4_dinov2.py").read_text())
        record = json.loads((destination / "reproduction/stage.json").read_text())
        self.assertEqual(record["core_classifier_changes"], [])
        self.assertEqual(len(record["patches"]), 2)

    def test_order_normalization_preserves_labels_groups_and_rebuilds_csv_hash(self):
        columns = ("sample_id", "sha256", "pixel_sha256", "center", "group_id", "label", "label_source", "mask_sha256", "bbox_sha256")
        rows = [{k: f"{k}_{sample}" for k in columns} for sample in ("a", "b")]
        for i, row in enumerate(rows):
            row.update(sample_id=("a", "b")[i], label=i, image_path=f"{self.root}/local/{i}.jpg")
        payload = [{k: r[k] for k in columns} for r in reversed(rows)]
        digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
        reference = {"sample_ids": ["b", "a"], "manifest_digest": digest}
        (self.root / "cohort_order.json").write_text(json.dumps(reference))
        base = self.root / "baseline"
        (base / "audit").mkdir(parents=True)
        path = base / "audit/manifest.csv"
        with path.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        path.with_suffix(".audit.json").write_text("{}")
        with patch.object(reproduce, "HERE", self.root):
            reproduce.normalize_manifest(base)
        with path.open(newline="") as stream:
            rebuilt = list(csv.DictReader(stream))
        self.assertEqual([r["sample_id"] for r in rebuilt], ["b", "a"])
        self.assertEqual({r["sample_id"]: int(r["label"]) for r in rebuilt}, {"a": 0, "b": 1})
        self.assertEqual(json.loads(path.with_suffix(".audit.json").read_text())["manifest_sha256"], reproduce.sha256(path))

    def test_prediction_support_is_paired_and_rejects_metadata_changes(self):
        path = self.root / "predictions.csv"
        rows = []
        for seed in (17, 29, 43):
            for method in ("a", "b"):
                for center in ("C2", "C4"):
                    for label in (0, 1):
                        rows.append({"seed": seed, "method": method, "center": center,
                                     "sample_id": f"{center}_{label}", "group_id": f"{center}_group",
                                     "label": label, "prediction": label if method == "a" else 1-label,
                                     "probability_class_1": .9 if label else .1})
        def save():
            with path.open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
        save()
        seeds, methods, centers, groups, labels, predictions, probabilities = read_paired([str(path)])
        self.assertEqual(seeds, [17, 29, 43])
        self.assertEqual(predictions.shape, (3, 2, 4))
        self.assertTrue(np.array_equal(predictions[0, 0], labels))
        rows[-1]["label"] = 0
        save()
        with self.assertRaisesRegex(ValueError, "differ"):
            read_paired([str(path)])


if __name__ == "__main__":
    unittest.main(verbosity=2)
