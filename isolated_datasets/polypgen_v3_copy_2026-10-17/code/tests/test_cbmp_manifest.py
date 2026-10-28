"""Local format and leakage checks using tiny artificial image fixtures.

These fixtures exercise indexing rules only; they are not experimental data.
Run from polypgen/: python -m unittest discover -s tests -p test_cbmp_manifest.py -v
"""
from __future__ import annotations

import csv
import shutil
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from cbmp_data import make_outer_splits
from polypgen_manifest import build_manifest, load_manifest


class ManifestFixtures(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / "PolypGen"
        self.root.mkdir()
        self.manifest = Path(self.temp.name) / "out" / "manifest.csv"
        self.image_counter = 0

    def image(self, relative, color=None):
        self.image_counter += 1
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        color = color if color is not None else (self.image_counter, 27, 53)
        Image.new("RGB", (8, 8), color).save(path)
        return path

    def mask(self, relative, present):
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        mask = Image.new("L", (8, 8), 0)
        if present:
            mask.putpixel((2, 2), 255)
        mask.save(path)
        return path

    def bbox(self, relative, present):
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("polyp 1 1 5 5\n" if present else "", encoding="utf-8")
        return path

    def sequence_metadata(self, rows, fields=None):
        path = Path(self.temp.name) / "sequence_metadata.csv"
        fields = fields or ["sequence_id", "center", "source", "patient_id"]
        with path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        return path

    def read_rows(self):
        with self.manifest.open(newline="", encoding="utf-8") as stream:
            return list(csv.DictReader(stream))

    def rejections(self):
        with self.manifest.with_suffix(".rejected.csv").open(newline="", encoding="utf-8") as stream:
            return list(csv.DictReader(stream))

    def test_original_images_exclude_masks_overlays_and_pooled_copies(self):
        self.image("data_C1/images_C1/frame.png")
        self.mask("data_C1/masks_C1/frame.png", True)
        self.image("data_C1/bbox_image_C1/frame.png")
        self.image("imagesAll_positive/frame.png")
        audit = build_manifest(self.root, self.manifest)
        rows = self.read_rows()
        self.assertEqual(audit["samples_indexed"], 1)
        self.assertEqual(rows[0]["relative_path"], "data_C1/images_C1/frame.png")
        self.assertEqual(rows[0]["label"], "1")
        self.assertEqual(audit["samples_rejected"], 3)
        self.assertTrue(all(r["reason"] == "not_original_image_leaf" for r in self.rejections()))
        self.assertEqual(set(audit["center_class_counts"]), {"C1"})
        self.assertEqual(audit["center_class_counts"]["C1"], {"0": 0, "1": 1})

    def test_positive_sequence_membership_does_not_imply_positive_frame(self):
        base = "data_C1/Positive/seq1"
        self.image(f"{base}/images_seq1/empty.png")
        self.image(f"{base}/images_seq1/present.png")
        self.mask(f"{base}/masks_seq1/empty.png", False)
        self.mask(f"{base}/masks_seq1/present.png", True)
        audit = build_manifest(self.root, self.manifest)
        rows = load_manifest(self.manifest)
        self.assertEqual(audit["class_counts"], {"0": 1, "1": 1})
        self.assertEqual({Path(r["image_path"]).stem: r["label"] for r in rows},
                         {"empty": 0, "present": 1})
        self.assertEqual({r["group_id"] for r in rows}, {"sequence:seq1"})

    def test_macos_resource_forks_are_normal_exclusions_not_samples(self):
        self.image("data_C1/images_C1/frame.png")
        self.mask("data_C1/masks_C1/frame.png", True)
        resource_paths = [
            "__MACOSX/data_C1/images_C1/frame.jpg",
            "data_C1/images_C1/._frame.jpg",
        ]
        for relative in resource_paths:
            path = self.root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            # Resource forks carry metadata, not decodable image pixels.
            path.write_bytes(b"fixture resource fork; not an image")
        audit = build_manifest(self.root, self.manifest)
        self.assertEqual(audit["samples_indexed"], 1)
        self.assertEqual(self.read_rows()[0]["relative_path"], "data_C1/images_C1/frame.png")
        self.assertEqual(audit["blocking_rejections"], 0)
        self.assertEqual(audit["rejection_counts"].get("system_metadata"), 2)
        rejected_metadata = [r["path"] for r in self.rejections() if r["reason"] == "system_metadata"]
        self.assertEqual(set(rejected_metadata), set(resource_paths))

    def test_mask_and_bbox_presence_agree(self):
        for present, name in ((False, "empty"), (True, "present")):
            self.image(f"data_C2/images_C2/{name}.png")
            self.mask(f"data_C2/masks_C2/{name}.png", present)
            self.bbox(f"data_C2/bbox_C2/{name}.txt", present)
        build_manifest(self.root, self.manifest)
        rows = load_manifest(self.manifest)
        self.assertEqual({r["label_source"] for r in rows}, {"bbox+mask"})
        self.assertTrue(all(r["mask_sha256"] and r["bbox_sha256"] for r in rows))

    def test_bbox_only_annotations_provide_both_labels(self):
        for present, name in ((False, "empty"), (True, "present")):
            self.image(f"data_C2/images_C2/{name}.png")
            self.bbox(f"data_C2/bbox_C2/{name}.txt", present)
        build_manifest(self.root, self.manifest)
        rows = load_manifest(self.manifest)
        self.assertEqual({r["label"] for r in rows}, {0, 1})
        self.assertEqual({r["label_source"] for r in rows}, {"bbox"})

    def test_bbox_mask_mismatch_is_rejected_and_audited_with_zero_rows(self):
        self.image("data_C1/images_C1/conflict.png")
        self.mask("data_C1/masks_C1/conflict.png", False)
        self.bbox("data_C1/bbox_C1/conflict.txt", True)
        audit = build_manifest(self.root, self.manifest)
        self.assertEqual(audit["samples_indexed"], 0)
        self.assertIn("Mask/bounding-box presence disagreement", audit["rejection_counts"])
        self.assertTrue(self.manifest.with_suffix(".audit.json").exists())
        self.assertEqual(self.read_rows(), [])

    def test_missing_annotation_is_not_inferred_as_negative(self):
        self.image("data_C1/images_C1/unannotated.png")
        audit = build_manifest(self.root, self.manifest)
        self.assertEqual(audit["samples_indexed"], 0)
        self.assertTrue(any("Missing annotation" in r["reason"] for r in self.rejections()))

    def test_negative_only_sequence_needs_reviewed_center_mapping(self):
        self.image("negativeOnly/seq003_neg/frame.png")
        audit = build_manifest(self.root, self.manifest)
        self.assertEqual(audit["samples_indexed"], 0)
        self.assertEqual(audit["unresolved_sequences"], ["seq3_neg"])
        metadata = self.sequence_metadata([
            {"sequence_id": "SEQ_003_NEG", "center": "c4", "source": "reviewed test mapping",
             "patient_id": "patient-4"},
        ])
        audit = build_manifest(self.root, self.manifest, sequence_metadata=metadata)
        row = self.read_rows()[0]
        self.assertEqual(audit["samples_indexed"], 1)
        self.assertEqual(row["center"], "C4")
        self.assertEqual(row["sequence_id"], "seq3_neg")
        self.assertEqual(row["label"], "0")
        self.assertEqual(row["label_source"], "documented_negative_only_sequence")
        self.assertEqual(row["group_id"], "patient:patient-4")
        self.assertEqual(row["center_source"], "reviewed test mapping")

    def test_patient_id_optional_but_provenance_column_required(self):
        self.image("negativeOnly/seq2_neg/frame.png")
        metadata = self.sequence_metadata([
            {"sequence_id": "seq2_neg", "center": "C2", "source": "reviewed mapping"},
        ], fields=["sequence_id", "center", "source"])
        build_manifest(self.root, self.manifest, sequence_metadata=metadata)
        self.assertEqual(self.read_rows()[0]["group_id"], "sequence:seq2_neg")
        metadata = self.sequence_metadata([
            {"sequence_id": "seq2_neg", "center": "C2"},
        ], fields=["sequence_id", "center"])
        with self.assertRaisesRegex(ValueError, "source"):
            build_manifest(self.root, self.manifest, sequence_metadata=metadata)

    def test_sequence_metadata_conflicts_and_canonical_duplicates_fail(self):
        self.image("data_C1/seq1/images_seq1/frame.png")
        self.mask("data_C1/seq1/masks_seq1/frame.png", True)
        metadata = self.sequence_metadata([
            {"sequence_id": "seq01", "center": "C2", "source": "reviewed", "patient_id": ""},
        ])
        audit = build_manifest(self.root, self.manifest, sequence_metadata=metadata)
        self.assertEqual(audit["samples_indexed"], 0)
        self.assertIn("Sequence mapping conflicts with folder center", audit["rejection_counts"])
        metadata = self.sequence_metadata([
            {"sequence_id": "seq01", "center": "C1", "source": "reviewed", "patient_id": ""},
            {"sequence_id": "seq-1", "center": "C1", "source": "reviewed", "patient_id": ""},
        ])
        with self.assertRaisesRegex(ValueError, "Duplicate sequence mapping"):
            build_manifest(self.root, self.manifest, sequence_metadata=metadata)

    def test_negative_only_positive_annotation_is_rejected(self):
        self.image("data_C1/negativeOnly/seq1_neg/images/frame.png")
        self.mask("data_C1/negativeOnly/seq1_neg/masks/frame.png", True)
        audit = build_manifest(self.root, self.manifest)
        self.assertEqual(audit["samples_indexed"], 0)
        self.assertIn("Negative-only sequence has a positive annotation", audit["rejection_counts"])

    def test_duplicate_images_union_every_frame_of_connected_sequences(self):
        first = self.image("data_C1/seq1/images_seq1/duplicate.png")
        second = self.image("data_C2/seq2/images_seq2/duplicate.png")
        shutil.copyfile(first, second)
        self.image("data_C1/seq1/images_seq1/unique.png")
        self.image("data_C2/seq2/images_seq2/unique.png")
        for center, seq in (("C1", "seq1"), ("C2", "seq2")):
            for name in ("duplicate", "unique"):
                self.mask(f"data_{center}/{seq}/masks_{seq}/{name}.png", True)
        audit = build_manifest(self.root, self.manifest)
        rows = self.read_rows()
        self.assertEqual(len(rows), 4)
        self.assertEqual(len({r["group_id"] for r in rows}), 1)
        self.assertEqual({r["group_id_original"] for r in rows}, {"sequence:seq1", "sequence:seq2"})
        self.assertEqual(len(audit["duplicate_content_groups"]), 1)
        self.assertEqual(len(audit["duplicate_content_groups"][0]), 2)

    def test_identical_image_with_contradictory_labels_fails(self):
        first = self.image("data_C1/images_C1/one.png")
        second = self.image("data_C2/images_C2/two.png")
        shutil.copyfile(first, second)
        self.mask("data_C1/masks_C1/one.png", True)
        self.mask("data_C2/masks_C2/two.png", False)
        with self.assertRaisesRegex(ValueError, "contradictory labels"):
            build_manifest(self.root, self.manifest)

    def test_sequence_only_excludes_still_frames(self):
        self.image("data_C1/images_C1/still.png")
        self.mask("data_C1/masks_C1/still.png", True)
        self.image("data_C1/seq1/images_seq1/frame.png")
        self.mask("data_C1/seq1/masks_seq1/frame.png", True)
        audit = build_manifest(self.root, self.manifest, sequence_only=True)
        self.assertEqual(audit["samples_indexed"], 1)
        self.assertEqual(self.read_rows()[0]["sequence_id"], "seq1")

    def test_loading_legacy_or_single_class_manifest_fails(self):
        self.image("data_C1/images_C1/frame.png")
        self.mask("data_C1/masks_C1/frame.png", True)
        build_manifest(self.root, self.manifest)
        with self.assertRaisesRegex(ValueError, "both labels"):
            load_manifest(self.manifest)
        self.manifest.write_text("sample_id,label\none,1\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "Legacy/incomplete"):
            load_manifest(self.manifest)


class SplitLeakageFixtures(unittest.TestCase):
    @staticmethod
    def rows():
        return [
            {"sample_id": f"{center}-{label}-{group}", "center": center, "label": label,
             "group_id": f"{center}-group-{group}", "sha256": f"{center}-{label}-{group}"}
            for center in ("C1", "C2", "C3") for label in (0, 1) for group in (0, 1)
        ]

    def assert_partitions_disjoint(self, rows, specs):
        for spec in specs:
            if spec["status"] != "ready":
                continue
            outer_train, outer_test = spec["outer_train"], spec["outer_test"]
            self.assertEqual({rows[i]["center"] for i in outer_test}, {spec["heldout_center"]})
            self.assertNotIn(spec["heldout_center"], {rows[i]["center"] for i in outer_train})
            for key in ("group_id", "sha256"):
                self.assertTrue({rows[i][key] for i in outer_train}.isdisjoint({rows[i][key] for i in outer_test}))
            for inner in spec["inner_folds"]:
                self.assertTrue(set(inner["train"]).issubset(outer_train))
                self.assertTrue(set(inner["validation"]).issubset(outer_train))
                self.assertTrue(set(inner["train"]).isdisjoint(inner["validation"]))
                for key in ("group_id", "sha256"):
                    self.assertTrue({rows[i][key] for i in inner["train"]}.isdisjoint(
                        {rows[i][key] for i in inner["validation"]}))

    def test_three_centers_support_ready_disjoint_splits(self):
        rows = self.rows()
        specs = make_outer_splits(rows)
        self.assertEqual(len(specs), 3)
        self.assertTrue(all(spec["status"] == "ready" for spec in specs))
        self.assert_partitions_disjoint(rows, specs)

    def test_cross_center_duplicate_content_is_excluded(self):
        rows = self.rows()
        rows[4]["sha256"] = rows[0]["sha256"]
        specs = make_outer_splits(rows)
        by_center = {spec["heldout_center"]: spec for spec in specs}
        self.assertNotEqual(by_center["C1"]["status"], "ready")
        self.assertNotEqual(by_center["C2"]["status"], "ready")
        self.assert_partitions_disjoint(rows, specs)

    def test_entire_group_cannot_cross_outer_or_inner_center_boundaries(self):
        rows = self.rows()
        rows[4]["group_id"] = rows[0]["group_id"]
        specs = make_outer_splits(rows)
        by_center = {spec["heldout_center"]: spec for spec in specs}
        self.assertNotEqual(by_center["C1"]["status"], "ready")
        self.assertNotEqual(by_center["C2"]["status"], "ready")
        self.assert_partitions_disjoint(rows, specs)

    def test_globally_missing_class_rejected_before_experiments(self):
        rows = [row for row in self.rows() if row["label"] == 1]
        with self.assertRaises(ValueError):
            make_outer_splits(rows)

    def test_target_missing_class_is_reported_without_inventing_coverage(self):
        rows = [row for row in self.rows() if row["center"] != "C1" or row["label"] == 1]
        specs = make_outer_splits(rows)
        by_center = {spec["heldout_center"]: spec for spec in specs}
        self.assertEqual(by_center["C1"]["status"], "excluded_target_missing_class")
        self.assert_partitions_disjoint(rows, specs)


if __name__ == "__main__":
    unittest.main()
