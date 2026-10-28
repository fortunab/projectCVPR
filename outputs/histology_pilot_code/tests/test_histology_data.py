"""Small generated fixtures test auditing only; these are never experiment data."""
import hashlib
import io
import json
from pathlib import Path
import stat
import sys
import tempfile
import unittest
from unittest.mock import patch
import zipfile

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from histology_cbmp.data import (  # noqa: E402
    CLASS_NAMES, build_manifest, download_archive, load_manifest,
    safe_extract_zip, verify_archive, import_tfds_records, TFDS_LABEL_NAMES,
)


class HistologyDataTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.data = self.root / "images"
        for index, name in enumerate(CLASS_NAMES):
            folder = self.data / name
            folder.mkdir(parents=True)
            Image.new("RGB", (150, 150), (index * 20, 50, 80)).save(folder / "tile.tif")
        self.manifest = self.root / "manifest.csv"

    def tearDown(self):
        self.temporary.cleanup()

    def test_subset_requires_explicit_opt_in_and_provides_honest_metadata(self):
        with self.assertRaisesRegex(ValueError, "audit failed"):
            build_manifest(self.data, self.manifest)
        audit = build_manifest(self.data, self.manifest, require_complete=False)
        self.assertEqual(audit["sample_count"], 8)
        self.assertFalse(audit["patient_metadata_available"])
        self.assertFalse(audit["center_metadata_available"])
        with self.assertRaisesRegex(ValueError, "incomplete"):
            load_manifest(self.manifest)
        rows = load_manifest(self.manifest, require_complete=False)
        self.assertEqual([row["label"] for row in rows], list(range(8)))
        first = self.manifest.read_bytes()
        build_manifest(self.data, self.manifest, require_complete=False)
        self.assertEqual(first, self.manifest.read_bytes())

    def test_mac_resource_forks_are_excluded_without_opening(self):
        (self.data / CLASS_NAMES[0] / "._tile.tif").write_bytes(b"not an image")
        resource = self.data / "__MACOSX" / CLASS_NAMES[0]
        resource.mkdir(parents=True)
        (resource / "tile.tif").write_bytes(b"not an image")
        audit = build_manifest(self.data, self.manifest, require_complete=False)
        self.assertEqual(audit["sample_count"], 8)
        self.assertEqual([r["reason"] for r in audit["excluded"]], ["system_metadata"] * 2)

    def test_wrong_size_and_mode_are_blocking(self):
        path = self.data / CLASS_NAMES[0] / "tile.tif"
        Image.new("L", (149, 150)).save(path)
        with self.assertRaises(ValueError):
            build_manifest(self.data, self.manifest, require_complete=False)
        audit = json.loads(self.manifest.with_suffix(".audit.json").read_text())
        self.assertFalse(audit["valid"])
        self.assertIn("expected RGB 150x150", audit["errors"][0])

    def test_identical_pixels_with_different_labels_are_blocking(self):
        original = self.data / CLASS_NAMES[0] / "tile.tif"
        with Image.open(original) as image:
            image.save(self.data / CLASS_NAMES[1] / "same_pixels.png")
        with self.assertRaises(ValueError):
            build_manifest(self.data, self.manifest, require_complete=False)
        audit = json.loads(self.manifest.with_suffix(".audit.json").read_text())
        self.assertTrue(any("pixel-identical" in issue for issue in audit["errors"]))

    def test_same_class_duplicate_is_reported_for_split_grouping(self):
        folder = self.data / CLASS_NAMES[0]
        (folder / "duplicate.tif").write_bytes((folder / "tile.tif").read_bytes())
        audit = build_manifest(self.data, self.manifest, require_complete=False)
        self.assertEqual(audit["byte_duplicate_count"], 1)
        self.assertEqual(audit["pixel_duplicate_count"], 1)

    def test_modified_manifest_is_rejected(self):
        build_manifest(self.data, self.manifest, require_complete=False)
        self.manifest.write_text(self.manifest.read_text().replace("01_TUMOR", "wrong", 1))
        with self.assertRaisesRegex(ValueError, "changed after auditing"):
            load_manifest(self.manifest, require_complete=False)

    def test_checksum_rejects_bad_archive_without_network(self):
        archive = self.root / "release.zip"
        archive.write_bytes(b"existing corrupted bytes")
        with patch("urllib.request.urlopen") as request:
            with self.assertRaisesRegex(ValueError, "MD5 mismatch"):
                download_archive(archive)
            request.assert_not_called()
        self.assertEqual(verify_archive(archive, hashlib.md5(archive.read_bytes()).hexdigest()), hashlib.md5(archive.read_bytes()).hexdigest())

    def test_zip_traversal_preflight_prevents_any_output(self):
        archive = self.root / "bad.zip"
        with zipfile.ZipFile(archive, "w") as handle:
            handle.writestr("safe.txt", "safe")
            handle.writestr("../escape.txt", "bad")
        destination = self.root / "extracted"
        with self.assertRaisesRegex(ValueError, "Unsafe"):
            safe_extract_zip(archive, destination)
        self.assertFalse(destination.exists())
        self.assertFalse((self.root / "escape.txt").exists())

    def test_zip_rejects_windows_paths_symlinks_and_duplicate_names(self):
        for name in ("C:/outside.txt", "folder\\outside.txt", "/absolute.txt"):
            with self.subTest(name=name):
                archive = self.root / "bad.zip"
                with zipfile.ZipFile(archive, "w") as handle:
                    # ZipInfo.__init__ normalizes Windows separators on Windows;
                    # assign the raw member name afterward to exercise hostile ZIPs.
                    member = zipfile.ZipInfo("placeholder")
                    member.filename = name
                    handle.writestr(member, "bad")
                with self.assertRaises(ValueError):
                    safe_extract_zip(archive, self.root / "out")
        archive = self.root / "bad.zip"
        with zipfile.ZipFile(archive, "w") as handle:
            symlink = zipfile.ZipInfo("link")
            symlink.create_system = 3
            symlink.external_attr = (stat.S_IFLNK | 0o777) << 16
            handle.writestr(symlink, "target")
        with self.assertRaises(ValueError):
            safe_extract_zip(archive, self.root / "out")
        with zipfile.ZipFile(archive, "w") as handle:
            handle.writestr("IMAGE.tif", "one")
            handle.writestr("image.tif", "two")
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            safe_extract_zip(archive, self.root / "out")

    def test_zip_extracts_without_overwriting_and_checks_file_parent_conflict(self):
        archive = self.root / "good.zip"
        with zipfile.ZipFile(archive, "w") as handle:
            handle.writestr("wrapper/a.txt", "value")
        destination = safe_extract_zip(archive, self.root / "out")
        self.assertEqual((destination / "wrapper/a.txt").read_text(), "value")
        with self.assertRaises(FileExistsError):
            safe_extract_zip(archive, destination)
        with zipfile.ZipFile(archive, "w") as handle:
            handle.writestr("parent", "file")
            handle.writestr("parent/child.txt", "bad")
        with self.assertRaisesRegex(ValueError, "file/directory conflict"):
            safe_extract_zip(archive, self.root / "out2")
        self.assertFalse((self.root / "out2").exists())


class TFDSImportTests(unittest.TestCase):
    """Tiny TFRecords exercise import safety, never represent the real dataset."""

    @classmethod
    def setUpClass(cls):
        try:
            import tensorflow as tf
        except ImportError:
            raise unittest.SkipTest("TensorFlow is optional outside TFRecord import")
        cls.tf = tf

    def setUp(self):
        # Windows app-container Temp disallows directory rename, even for a
        # plain empty fixture; use the writable workspace to exercise commit.
        self.temporary = tempfile.TemporaryDirectory(
            prefix="_test_tfds_", dir=Path(__file__).resolve().parents[1] / "histology_cbmp"
        )
        self.root = Path(self.temporary.name)
        self.cache = self.root / "cache"
        self.cache.mkdir()
        self.destination = self.root / "imported"

    def tearDown(self):
        self.temporary.cleanup()

    def _cache(self, *, unsafe_filename=False, invalid_rgb=False):
        # Only this private fixture patches record counts: production CLI and
        # public importer always require the full 2500+2500 / 625-per-class cache.
        metadata = {
            "name": "colorectal_histology", "version": "2.0.0", "fileFormat": "tfrecord",
            "splits": [{"name": "train", "shardLengths": ["4", "4"]}],
        }
        (self.cache / "dataset_info.json").write_text(json.dumps(metadata))
        features = {"featuresDict": {"features": {
            "filename": {},
            "image": {"image": {"dtype": "uint8", "shape": {"dimensions": ["150", "150", "3"]}}},
            "label": {"classLabel": {"numClasses": "8"}},
        }}}
        (self.cache / "features.json").write_text(json.dumps(features))
        (self.cache / "label.labels.txt").write_text("\n".join(TFDS_LABEL_NAMES) + "\n")
        for shard in range(2):
            path = self.cache / f"colorectal_histology-train.tfrecord-{shard:05d}-of-00002"
            with self.tf.io.TFRecordWriter(str(path)) as writer:
                for label in range(shard * 4, shard * 4 + 4):
                    encoded = io.BytesIO()
                    mode = "L" if invalid_rgb and label == 0 else "RGB"
                    Image.new(mode, (150, 150), label * 20).save(encoded, format="PNG")
                    filename = "../escape.png" if unsafe_filename and label == 0 else f"tile_{label}.tif"
                    feature = {
                        "filename": self.tf.train.Feature(bytes_list=self.tf.train.BytesList(value=[filename.encode()])),
                        "image": self.tf.train.Feature(bytes_list=self.tf.train.BytesList(value=[encoded.getvalue()])),
                        "label": self.tf.train.Feature(int64_list=self.tf.train.Int64List(value=[label])),
                    }
                    writer.write(self.tf.train.Example(features=self.tf.train.Features(feature=feature)).SerializeToString())

    def _import_fixture(self):
        with patch.multiple("histology_cbmp.data", TFDS_SHARD_LENGTHS=(4, 4), TFDS_PER_CLASS=1):
            return import_tfds_records(self.cache, self.destination)

    def test_complete_fixture_import_saves_provenance_and_mapping(self):
        self._cache()
        path = self._import_fixture()
        self.assertEqual(len(list(path.glob("*/*.png"))), 8)
        provenance = json.loads((path / "tfds_import_provenance.json").read_text())
        self.assertEqual(provenance["record_count"], 8)
        self.assertEqual(len(provenance["shard_sha256"]), 2)
        self.assertEqual(len(provenance["metadata_sha256"]), 3)
        self.assertIn("not an official archive MD5", provenance["checksum_scope"])
        with self.assertRaises(FileExistsError):
            self._import_fixture()
        audit = build_manifest(path, self.root / "manifest.csv", require_complete=False)
        self.assertEqual(audit["provenance"]["import_format"], "TFDS_TFRecord")

    def test_missing_shard_is_rejected_before_import(self):
        self._cache()
        (self.cache / "colorectal_histology-train.tfrecord-00001-of-00002").unlink()
        with self.assertRaisesRegex(ValueError, "Both complete"):
            self._import_fixture()
        self.assertFalse(self.destination.exists())

    def test_unsafe_source_filename_and_bad_image_leave_no_dataset(self):
        for options in ({"unsafe_filename": True}, {"invalid_rgb": True}):
            with self.subTest(options=options):
                self._cache(**options)
                with self.assertRaises(ValueError):
                    self._import_fixture()
                self.assertFalse(self.destination.exists())
                self.assertFalse((self.root / "escape.png").exists())

    def test_corrupted_crc_is_rejected_without_committing_partial_import(self):
        self._cache()
        path = self.cache / "colorectal_histology-train.tfrecord-00001-of-00002"
        damaged = bytearray(path.read_bytes())
        damaged[-1] ^= 1
        path.write_bytes(damaged)
        with self.assertRaisesRegex(ValueError, "Corrupt/unreadable"):
            self._import_fixture()
        self.assertFalse(self.destination.exists())


if __name__ == "__main__":
    unittest.main()
