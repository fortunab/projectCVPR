"""Offline downloader checks: no dataset access or network requests."""

from __future__ import annotations

import hashlib
import io
import json
import shutil
import stat
import sys
import tempfile
import unittest
import warnings
import zipfile
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import download_synapse_polypgen as downloader
from download_synapse_polypgen import download_polypgen_from_synapse, safe_extract_zip


ENTITY = "syn45200214"
ARCHIVE = "PolypGen2021_MultiCenterData_v3.zip"


class DownloaderTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.destination = self.root / "downloads"
        self.fixture = self.root / "fixture.zip"
        with zipfile.ZipFile(self.fixture, "w") as zipped:
            zipped.writestr("PolypGen/Positive/images_C1/frame.jpg", b"local fixture")
        self.md5 = hashlib.md5(self.fixture.read_bytes()).hexdigest()
        self.client = Mock()
        self.client.restGET.return_value = {
            "id": ENTITY, "versionNumber": 1, "name": ARCHIVE,
            "dataFileHandleId": "fixture-handle",
            "concreteType": "org.sagebionetworks.repo.model.FileEntity",
        }
        self.client.get.side_effect = self.get_file

    def get_file(self, entity, **kwargs):
        self.assertEqual(entity, ENTITY)
        self.assertEqual(kwargs["version"], 1)
        result = {"md5": self.md5, "fileSize": self.fixture.stat().st_size}
        if kwargs["downloadFile"]:
            target = Path(kwargs["downloadLocation"]) / ARCHIVE
            shutil.copyfile(self.fixture, target)
            result["path"] = str(target)
        return result

    def run_downloader(self, **kwargs):
        output = io.StringIO()
        with redirect_stdout(output):
            ok = download_polypgen_from_synapse(self.destination, client=self.client, **kwargs)
        return ok, output.getvalue()

    def report(self):
        return json.loads((self.destination / f"{ENTITY}.v1.metadata.json").read_text())

    def test_metadata_only_never_logs_in_or_downloads(self):
        ok, output = self.run_downloader(metadata_only=True)
        self.assertTrue(ok)
        self.client.login.assert_not_called()
        self.client.get.assert_called_once_with(ENTITY, version=1, downloadFile=False)
        self.assertFalse((self.destination / ARCHIVE).exists())
        self.assertEqual(self.report()["expected_md5"], self.md5)
        self.assertIn("no archive downloaded", output)

    def test_public_metadata_still_works_when_file_metadata_is_denied(self):
        self.client.get.side_effect = PermissionError("not public")
        ok, _ = self.run_downloader(metadata_only=True)
        self.assertTrue(ok)
        self.assertIsNone(self.report()["expected_md5"])
        self.assertIn("permissions", self.report()["file_metadata_note"])

    def test_pinned_single_file_and_environment_auth(self):
        with patch.dict("os.environ", {"SYNAPSE_AUTH_TOKEN": "fixture-secret"}):
            ok, output = self.run_downloader(extract=True)
        self.assertTrue(ok)
        self.client.login.assert_called_once_with(authToken="fixture-secret", silent=True)
        self.assertNotIn("fixture-secret", output)
        self.client.restGET.assert_called_once_with(f"/entity/{ENTITY}/version/1")
        self.assertEqual(len(self.client.get.call_args_list), 2)
        report = self.report()
        self.assertTrue(report["checksum_verified"])
        self.assertEqual(report["status"], "extracted")
        self.assertEqual(report["extracted_files"], 1)
        self.assertTrue((self.destination / "extracted/PolypGen/Positive/images_C1/frame.jpg").is_file())

    def test_cached_login_and_no_automatic_extraction(self):
        with patch.dict("os.environ", {}, clear=True):
            ok, _ = self.run_downloader()
        self.assertTrue(ok)
        self.client.login.assert_called_once_with(silent=True)
        self.assertFalse((self.destination / "extracted").exists())

    def test_checksum_mismatch_prevents_extraction(self):
        original = self.get_file
        def wrong_md5(entity, **kwargs):
            result = original(entity, **kwargs)
            result["md5"] = "0" * 32
            return result
        self.client.get.side_effect = wrong_md5
        ok, _ = self.run_downloader(extract=True)
        self.assertFalse(ok)
        self.assertEqual(self.report()["status"], "checksum_mismatch")
        self.assertFalse((self.destination / "extracted").exists())

    def test_failed_login_does_not_print_exception_or_token(self):
        self.client.login.side_effect = RuntimeError("secret-token-in-request")
        ok, output = self.run_downloader(auth_token="secret-token-in-request")
        self.assertFalse(ok)
        self.assertNotIn("secret-token-in-request", output)
        self.client.restGET.assert_not_called()

    def test_failed_request_does_not_print_exception_headers(self):
        self.client.restGET.side_effect = ValueError("Authorization: secret-request")
        ok, output = self.run_downloader(metadata_only=True)
        self.assertFalse(ok)
        self.assertNotIn("secret-request", output)

    def test_release_version_mismatch_stops_download(self):
        self.client.restGET.return_value["versionNumber"] = 2
        ok, _ = self.run_downloader()
        self.assertFalse(ok)
        self.client.get.assert_not_called()


class CommandLineTests(unittest.TestCase):
    """Exercise the hidden token prompt without terminal input or network calls."""

    def invoke_cli(self, arguments, *, token="fixture-cli-secret", prompt_error=None):
        output, errors = io.StringIO(), io.StringIO()
        with (
            patch.object(sys, "argv", ["download_synapse_polypgen.py", *arguments]),
            patch("getpass.getpass", return_value=token, side_effect=prompt_error) as prompt,
            patch.object(downloader, "download_polypgen_from_synapse", return_value=True) as download,
            redirect_stdout(output),
            redirect_stderr(errors),
        ):
            try:
                status = downloader.main()
            except SystemExit as error:
                status = error.code
        return status, prompt, download, output.getvalue(), errors.getvalue()

    def test_normal_cli_uses_existing_authentication(self):
        status, prompt, download, _, _ = self.invoke_cli(["--dest-dir", "fixture-download"])
        self.assertEqual(status, 0)
        prompt.assert_not_called()
        download.assert_called_once_with(
            Path("fixture-download"), auth_token=None,
            entity_id=ENTITY, version=1, metadata_only=False, extract=False,
        )

    def test_hidden_prompt_strips_and_forwards_token_without_printing_it(self):
        secret = "fixture-cli-secret"
        status, prompt, download, output, errors = self.invoke_cli(
            ["--dest-dir", "fixture-download", "--prompt-token", "--extract"],
            token=f"  {secret}\n",
        )
        self.assertEqual(status, 0)
        prompt.assert_called_once()
        download.assert_called_once_with(
            Path("fixture-download"), auth_token=secret,
            entity_id=ENTITY, version=1, metadata_only=False, extract=True,
        )
        self.assertNotIn(secret, output)
        self.assertNotIn(secret, errors)

    def test_blank_prompt_refuses_download(self):
        for token in ("", "   \t\n"):
            with self.subTest(token=repr(token)):
                status, prompt, download, _, _ = self.invoke_cli(["--prompt-token"], token=token)
                self.assertNotEqual(status, 0)
                prompt.assert_called_once()
                download.assert_not_called()

    def test_prompt_eof_refuses_download(self):
        status, prompt, download, _, _ = self.invoke_cli(
            ["--prompt-token"], prompt_error=EOFError,
        )
        self.assertNotEqual(status, 0)
        prompt.assert_called_once()
        download.assert_not_called()

    def test_prompt_keyboard_interrupt_refuses_download(self):
        status, prompt, download, _, _ = self.invoke_cli(
            ["--prompt-token"], prompt_error=KeyboardInterrupt,
        )
        self.assertNotEqual(status, 0)
        prompt.assert_called_once()
        download.assert_not_called()

    def test_metadata_conflict_refuses_before_prompt_or_download(self):
        status, prompt, download, _, _ = self.invoke_cli(["--metadata-only", "--prompt-token"])
        self.assertNotEqual(status, 0)
        prompt.assert_not_called()
        download.assert_not_called()

    def test_unavailable_hidden_terminal_refuses_insecure_fallback(self):
        def insecure_prompt(*_args, **_kwargs):
            warnings.warn("Terminal cannot hide input", downloader.getpass.GetPassWarning)
            return "fixture-cli-secret"
        status, prompt, download, output, errors = self.invoke_cli(
            ["--prompt-token"], prompt_error=insecure_prompt,
        )
        self.assertNotEqual(status, 0)
        prompt.assert_called_once()
        download.assert_not_called()
        self.assertNotIn("fixture-cli-secret", output + errors)


class ExtractionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.archive = self.root / "fixture.zip"
        self.destination = self.root / "extracted"

    def test_rejects_traversal_and_windows_absolute_paths_before_writes(self):
        for path in ("../escape.jpg", "safe/../../escape.jpg", "/absolute.jpg", "C:/escape.jpg", "C:escape.jpg", "\\\\host\\file.jpg", "safe/file.jpg:stream"):
            with self.subTest(path=path):
                with zipfile.ZipFile(self.archive, "w") as zipped:
                    zipped.writestr("safe.jpg", b"first")
                    zipped.writestr(path, b"escape")
                with self.assertRaises(ValueError):
                    safe_extract_zip(self.archive, self.destination, workspace_root=self.root)
                self.assertFalse(self.destination.exists())

    def test_rejects_symlink_member(self):
        entry = zipfile.ZipInfo("linked.jpg")
        entry.create_system = 3
        entry.external_attr = (stat.S_IFLNK | 0o777) << 16
        with zipfile.ZipFile(self.archive, "w") as zipped:
            zipped.writestr(entry, "../../elsewhere")
        with self.assertRaises(ValueError):
            safe_extract_zip(self.archive, self.destination, workspace_root=self.root)
        self.assertFalse(self.destination.exists())

    def test_rejects_destination_outside_workspace(self):
        with self.assertRaises(ValueError):
            safe_extract_zip(self.archive, self.root.parent / "outside", workspace_root=self.root)

    def test_does_not_overwrite_existing_file(self):
        self.destination.mkdir()
        (self.destination / "frame.jpg").write_bytes(b"existing")
        with zipfile.ZipFile(self.archive, "w") as zipped:
            zipped.writestr("frame.jpg", b"new")
        with self.assertRaises(FileExistsError):
            safe_extract_zip(self.archive, self.destination, workspace_root=self.root)
        self.assertEqual((self.destination / "frame.jpg").read_bytes(), b"existing")


if __name__ == "__main__":
    unittest.main()
