"""Audit policy and content-integrity checks with tiny artificial fixtures only."""
from __future__ import annotations

import csv
import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

from PIL import Image, PngImagePlugin

import audit_cbmp_data
from cbmp_data import make_outer_splits
from polypgen_manifest import MANIFEST_SCHEMA_VERSION, build_manifest, load_manifest


class AuditPolicyFixtures(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.root = self.base / "PolypGen"
        self.root.mkdir()
        self.manifest = self.base / "manifest.csv"

    def image(self, relative, color, *, comment=None, compression=6):
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        metadata = PngImagePlugin.PngInfo()
        if comment is not None:
            metadata.add_text("fixture_comment", comment)
        Image.new("RGB", (8, 8), color).save(
            path, pnginfo=metadata, compress_level=compression)
        return path

    def annotated_frame(self, center, sequence, name, color, present, **kwargs):
        container = f"data_{center}/{sequence}"
        image = self.image(f"{container}/images_{sequence}/{name}.png", color, **kwargs)
        mask_path = self.root / f"{container}/masks_{sequence}/{name}.png"
        mask_path.parent.mkdir(parents=True, exist_ok=True)
        mask = Image.new("L", (8, 8), 0)
        if present:
            mask.putpixel((3, 3), 255)
        mask.save(mask_path)
        return image

    def metadata(self, sequences):
        path = self.base / "reviewed.csv"
        with path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=["sequence_id", "center", "source"])
            writer.writeheader()
            writer.writerows({"sequence_id": seq, "center": center,
                              "source": "artificial fixture provenance"}
                             for seq, center in sequences)
        return path

    def read_rows(self):
        with self.manifest.open(newline="", encoding="utf-8") as stream:
            return list(csv.DictReader(stream))

    def test_verified_subset_requires_both_sequence_scope_and_nonempty_review(self):
        self.image("negativeOnly/seq1_neg/frame.png", (10, 20, 30))
        reviewed = self.metadata([("seq1_neg", "C1")])
        with self.assertRaisesRegex(ValueError, "sequence_only.*nonempty"):
            build_manifest(self.root, self.manifest, sequence_metadata=reviewed,
                           verified_sequences_only=True)
        with self.assertRaisesRegex(ValueError, "sequence_only.*nonempty"):
            build_manifest(self.root, self.manifest, sequence_only=True,
                           verified_sequences_only=True)
        empty_review = self.metadata([])
        with self.assertRaisesRegex(ValueError, "sequence_only.*nonempty"):
            build_manifest(self.root, self.manifest, sequence_only=True,
                           sequence_metadata=empty_review, verified_sequences_only=True)

    def test_unknown_sequences_block_by_default_and_require_explicit_subset_opt_in(self):
        self.image("negativeOnly/seq1_neg/reviewed.png", (10, 20, 30))
        self.image("negativeOnly/seq2_neg/unresolved.png", (40, 50, 60))
        reviewed = self.metadata([("seq1_neg", "C1")])
        full_audit = build_manifest(self.root, self.manifest, sequence_only=True,
                                    sequence_metadata=reviewed)
        self.assertEqual(full_audit["samples_indexed"], 1)
        self.assertEqual(full_audit["blocking_rejections"], 1)
        self.assertEqual(full_audit["unresolved_sequences"], ["seq2_neg"])
        self.assertFalse(full_audit["verified_sequences_only"])
        self.assertIsNone(full_audit["included_sequence_ids"])

        subset_audit = build_manifest(self.root, self.manifest, sequence_only=True,
                                      sequence_metadata=reviewed, verified_sequences_only=True)
        self.assertEqual(subset_audit["samples_indexed"], 1)
        self.assertEqual(subset_audit["blocking_rejections"], 0)
        self.assertEqual(subset_audit["unresolved_sequences"], [])
        self.assertEqual(subset_audit["rejection_counts"][
            "sequence_excluded_by_verified_metadata_policy"], 1)
        self.assertEqual(subset_audit["included_sequence_ids"], ["seq1_neg"])
        self.assertIn("not a complete-release evaluation", subset_audit["scope"])
        self.assertEqual(self.read_rows()[0]["sequence_id"], "seq1_neg")
        saved = json.loads(self.manifest.with_suffix(".audit.json").read_text())
        self.assertEqual(saved["schema_version"], MANIFEST_SCHEMA_VERSION)
        self.assertEqual(saved["manifest_sha256"], subset_audit["manifest_sha256"])

    def test_subset_policy_excludes_unreviewed_sequence_even_with_center_in_path(self):
        self.annotated_frame("C1", "seq1", "reviewed", (1, 2, 3), True)
        self.annotated_frame("C2", "seq2", "unreviewed", (4, 5, 6), False)
        reviewed = self.metadata([("seq1", "C1")])
        full_audit = build_manifest(self.root, self.manifest, sequence_only=True,
                                    sequence_metadata=reviewed)
        self.assertEqual(full_audit["samples_indexed"], 2)
        subset_audit = build_manifest(self.root, self.manifest, sequence_only=True,
                                      sequence_metadata=reviewed, verified_sequences_only=True)
        self.assertEqual(subset_audit["samples_indexed"], 1)
        self.assertEqual(subset_audit["blocking_rejections"], 0)
        self.assertEqual({row["sequence_id"] for row in self.read_rows()}, {"seq1"})

    def test_different_png_encodings_of_same_pixels_union_entire_sequences(self):
        first = self.annotated_frame("C1", "seq1", "shared", (20, 40, 60), True,
                                     comment="first encoding", compression=0)
        second = self.annotated_frame("C2", "seq2", "shared", (20, 40, 60), True,
                                      comment="different metadata", compression=9)
        self.assertNotEqual(first.read_bytes(), second.read_bytes())
        self.annotated_frame("C1", "seq1", "unique", (30, 50, 70), False)
        self.annotated_frame("C2", "seq2", "unique", (35, 55, 75), False)
        self.annotated_frame("C3", "seq3", "negative", (40, 60, 80), False)
        self.annotated_frame("C3", "seq3", "positive", (45, 65, 85), True)
        audit = build_manifest(self.root, self.manifest)
        rows = load_manifest(self.manifest)
        shared = [row for row in rows if Path(row["image_path"]).stem == "shared"]
        self.assertEqual(len({row["sha256"] for row in shared}), 2)
        self.assertEqual(len({row["pixel_sha256"] for row in shared}), 1)
        connected_sequences = [row for row in rows if row["sequence_id"] in {"seq1", "seq2"}]
        self.assertEqual(len({row["group_id"] for row in connected_sequences}), 1)
        self.assertEqual({row["group_id_original"] for row in connected_sequences},
                         {"sequence:seq1", "sequence:seq2"})
        self.assertEqual(len(audit["duplicate_content_groups"]), 1)
        specs = {spec["heldout_center"]: spec for spec in make_outer_splits(rows)}
        self.assertEqual(specs["C1"]["status"], "excluded_group_cross_center")
        self.assertEqual(specs["C2"]["status"], "excluded_group_cross_center")
        for spec in specs.values():
            if spec["status"] == "ready":
                train_pixels = {rows[i]["pixel_sha256"] for i in spec["outer_train"]}
                test_pixels = {rows[i]["pixel_sha256"] for i in spec["outer_test"]}
                self.assertTrue(train_pixels.isdisjoint(test_pixels))

    def test_same_pixels_with_conflicting_labels_fail_despite_different_file_bytes(self):
        self.annotated_frame("C1", "seq1", "one", (15, 25, 35), True, comment="one")
        self.annotated_frame("C2", "seq2", "two", (15, 25, 35), False, comment="two")
        with self.assertRaisesRegex(ValueError, "contradictory labels"):
            build_manifest(self.root, self.manifest)

    def test_mutated_manifest_is_rejected_instead_of_reusing_old_audit(self):
        self.annotated_frame("C1", "seq1", "negative", (1, 3, 5), False)
        self.annotated_frame("C1", "seq1", "positive", (2, 4, 6), True)
        build_manifest(self.root, self.manifest)
        self.assertEqual(len(load_manifest(self.manifest)), 2)
        rows = self.read_rows()
        rows[0]["center"] = "C6"
        with self.manifest.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        with self.assertRaisesRegex(ValueError, "Manifest changed"):
            load_manifest(self.manifest)

    def test_cli_forwards_explicit_subset_flag_and_keeps_default_false(self):
        reviewed = self.metadata([("seq1_neg", "C1")])
        for subset in (False, True):
            with self.subTest(subset=subset):
                arguments = ["audit_cbmp_data.py", str(self.root), "--sequence-only",
                             "--sequence-metadata", str(reviewed)]
                if subset:
                    arguments.append("--verified-sequences-only")
                with patch("sys.argv", arguments), patch.object(
                        audit_cbmp_data, "build_manifest", return_value={"samples_indexed": 2}) as build, \
                        redirect_stdout(io.StringIO()):
                    audit_cbmp_data.main()
                self.assertEqual(build.call_args.kwargs["verified_sequences_only"], subset)
                self.assertTrue(build.call_args.kwargs["sequence_only"])


if __name__ == "__main__":
    unittest.main()
