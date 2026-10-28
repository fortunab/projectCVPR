"""Audit CLI prerequisites and concise failures; no real images or downloads."""
from __future__ import annotations

import io
import json
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import patch

import audit_cbmp_data


class AuditCliTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.root = self.base / "extracted_polypgen"
        self.root.mkdir()
        self.manifest = self.base / "results" / "manifest.csv"

    def invoke(self, arguments, *, result=None, error=None):
        stdout, stderr = io.StringIO(), io.StringIO()
        result = result if result is not None else {"samples_indexed": 2, "samples_rejected": 0}
        with patch("sys.argv", ["audit_cbmp_data.py", *map(str, arguments)]), \
                patch.object(audit_cbmp_data, "build_manifest", return_value=result,
                             side_effect=error) as build, \
                redirect_stdout(stdout), redirect_stderr(stderr):
            try:
                audit_cbmp_data.main()
            except SystemExit as exc:
                exit_code = exc.code
            else:
                exit_code = 0
        return exit_code, stdout.getvalue(), stderr.getvalue(), build

    def test_missing_dataset_explains_download_extract_prerequisite(self):
        missing = self.base / "not_downloaded"
        code, stdout, stderr, build = self.invoke([missing])
        self.assertEqual(code, 2)
        build.assert_not_called()
        self.assertEqual(stdout, "")
        self.assertIn(str(missing), stderr)
        self.assertIn("download", stderr.lower())
        self.assertIn("extract", stderr.lower())
        self.assertNotIn("Traceback", stderr)

    def test_missing_metadata_suggests_initial_audit_without_mapping(self):
        missing = self.base / "sequence_metadata.csv"
        code, stdout, stderr, build = self.invoke([
            self.root, "--sequence-metadata", missing,
        ])
        self.assertEqual(code, 2)
        build.assert_not_called()
        self.assertEqual(stdout, "")
        self.assertIn(str(missing), stderr)
        self.assertIn("--sequence-metadata", stderr)
        self.assertIn("without", stderr.lower())
        self.assertNotIn("Traceback", stderr)

    def test_manifest_value_error_is_concise_and_has_nonzero_exit(self):
        message = "Identical image content has contradictory labels"
        code, stdout, stderr, build = self.invoke([self.root], error=ValueError(message))
        self.assertEqual(code, 2)
        build.assert_called_once()
        self.assertEqual(stdout, "")
        self.assertIn(message, stderr)
        self.assertNotIn("Traceback", stderr)

    def test_missing_images_error_is_concise_and_has_nonzero_exit(self):
        message = "No images under the extracted directory"
        code, stdout, stderr, build = self.invoke([self.root], error=FileNotFoundError(message))
        self.assertEqual(code, 2)
        build.assert_called_once()
        self.assertEqual(stdout, "")
        self.assertIn(message, stderr)
        self.assertNotIn("Traceback", stderr)

    def test_valid_default_call_prints_json_and_preserves_defaults(self):
        audit = {"samples_indexed": 12, "samples_rejected": 3, "class_counts": {"0": 5, "1": 7}}
        code, stdout, stderr, build = self.invoke([self.root], result=audit)
        self.assertEqual(code, 0)
        self.assertEqual(stderr, "")
        self.assertEqual(json.loads(stdout), audit)
        build.assert_called_once()
        arguments, keywords = build.call_args
        self.assertEqual(Path(arguments[0]), self.root)
        self.assertEqual(Path(arguments[1]), Path("results/cbmp/manifest.csv"))
        self.assertIsNone(keywords["sequence_metadata"])
        self.assertFalse(keywords["sequence_only"])
        self.assertEqual(keywords["mask_threshold"], 127)
        self.assertEqual(keywords["min_foreground_pixels"], 1)

    def test_valid_explicit_arguments_are_forwarded(self):
        metadata = self.base / "reviewed_sequence_metadata.csv"
        metadata.write_text("sequence_id,center,source\n", encoding="utf-8")
        code, stdout, stderr, build = self.invoke([
            self.root, "--manifest", self.manifest, "--sequence-metadata", metadata,
            "--sequence-only", "--mask-threshold", "100", "--min-foreground-pixels", "4",
        ])
        self.assertEqual(code, 0)
        self.assertEqual(stderr, "")
        self.assertEqual(json.loads(stdout)["samples_indexed"], 2)
        arguments, keywords = build.call_args
        self.assertEqual(Path(arguments[0]), self.root)
        self.assertEqual(Path(arguments[1]), self.manifest)
        self.assertEqual(Path(keywords["sequence_metadata"]), metadata)
        self.assertTrue(keywords["sequence_only"])
        self.assertEqual(keywords["mask_threshold"], 100)
        self.assertEqual(keywords["min_foreground_pixels"], 4)

    def test_zero_rows_preserves_audit_and_mapping_template_report(self):
        template = self.manifest.with_suffix(".sequence_metadata_template.csv")
        audit = {"samples_indexed": 0, "samples_rejected": 2,
                 "unresolved_sequences": ["seq1", "seq2_neg"],
                 "sequence_metadata_template": str(template), "manifest": str(self.manifest)}
        code, stdout, stderr, build = self.invoke([
            self.root, "--manifest", self.manifest,
        ], result=audit)
        self.assertNotEqual(code, 0)
        self.assertIn("No usable rows", str(code))
        build.assert_called_once()
        reported = json.loads(stdout)
        self.assertEqual(reported, audit)
        self.assertEqual(reported["sequence_metadata_template"], str(template))
        self.assertNotIn("Traceback", stderr)


if __name__ == "__main__":
    unittest.main()
