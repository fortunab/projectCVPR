"""Download the pinned PolypGen v3 archive, rather than an entire Synapse project.

Credentials come from SYNAPSE_AUTH_TOKEN or the standard Synapse configuration.
Use --metadata-only to inspect public release metadata without downloading data.
Extraction is opt-in and confined below --dest-dir.
"""

from __future__ import annotations

import argparse
import getpass
import hashlib
import json
import os
import re
import shutil
import stat
import sys
import warnings
import zipfile
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath, PureWindowsPath

DEFAULT_ENTITY = "syn45200214"
DEFAULT_VERSION = 1


def _within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def safe_extract_zip(archive: Path, destination: Path, *, workspace_root: Path) -> int:
    """Preflight every ZIP path, reject links, and never overwrite existing files."""
    root = Path(workspace_root).resolve()
    destination = Path(destination).resolve()
    if not _within(destination, root) or destination == root:
        raise ValueError("Extraction destination must be below the download workspace")
    plans, seen = [], set()
    with zipfile.ZipFile(archive) as zipped:
        for member in zipped.infolist():
            name = member.filename.replace("\\", "/")
            relative = PurePosixPath(name)
            if (not relative.parts or relative.is_absolute()
                    or PureWindowsPath(name).drive or ".." in relative.parts
                    or any(":" in part for part in relative.parts)):
                raise ValueError("Archive contains an unsafe member path")
            file_type = stat.S_IFMT(member.external_attr >> 16)
            if file_type not in (0, stat.S_IFREG, stat.S_IFDIR):
                raise ValueError("Archive contains a symlink or special file")
            target = destination.joinpath(*relative.parts).resolve()
            if not _within(target, destination) or target == destination:
                raise ValueError("Archive member escapes the extraction directory")
            key = str(target).casefold()
            if key in seen:
                raise ValueError("Archive contains duplicate destination paths")
            seen.add(key)
            if target.exists() and (not member.is_dir() or not target.is_dir()):
                raise FileExistsError("Extraction would overwrite an existing path")
            plans.append((member, target))
        # A malicious later entry must not cause earlier writes.
        destination.mkdir(parents=True, exist_ok=True)
        for member, target in plans:
            if member.is_dir():
                target.mkdir(parents=True, exist_ok=True)
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                with zipped.open(member) as source, target.open("xb") as output:
                    shutil.copyfileobj(source, output)
    return sum(not member.is_dir() for member, _ in plans)


def _field(entity, key, default=None):
    if isinstance(entity, Mapping):
        return entity.get(key, default)
    return getattr(entity, key, default)


def _file_metadata(entity) -> dict:
    handle = _field(entity, "_file_handle", {}) or {}
    return {
        "expected_md5": _field(entity, "md5") or _field(handle, "contentMd5"),
        "size_bytes": _field(entity, "fileSize") or _field(handle, "contentSize"),
    }


def _checksums(path: Path) -> dict:
    md5, sha256 = hashlib.md5(), hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            md5.update(block)
            sha256.update(block)
    return {"actual_md5": md5.hexdigest(), "sha256": sha256.hexdigest()}


def _save_report(destination: Path, report: dict) -> Path:
    path = destination / f"{report['entity_id']}.v{report['version']}.metadata.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return path


def download_polypgen_from_synapse(
    dest_dir: Path, auth_token: str | None = None, *,
    entity_id: str = DEFAULT_ENTITY, version: int = DEFAULT_VERSION,
    metadata_only: bool = False, extract: bool = False, client=None,
) -> bool:
    """Retrieve one release, with injectable client for offline checks.

    Tokens are deliberately not accepted as CLI arguments. ``auth_token`` is
    retained for existing Python callers, with the environment preferred for CLI.
    """
    try:
        if not re.fullmatch(r"syn[0-9]+", entity_id) or version < 1:
            raise ValueError("Use a Synapse entity ID and a positive integer version")
        if metadata_only and extract:
            raise ValueError("--metadata-only cannot be combined with --extract")
        destination = Path(dest_dir).expanduser().resolve()
        destination.mkdir(parents=True, exist_ok=True)
        if client is None:
            try:
                import synapseclient
            except ImportError:
                print("Install the dependency with: python -m pip install synapseclient")
                return False
            client = synapseclient.Synapse(
                skip_checks=True, silent=True,
                cache_root_dir=str(destination / ".synapse-cache"),
            )
        if not metadata_only:
            try:
                token = auth_token or os.environ.get("SYNAPSE_AUTH_TOKEN")
                if token:
                    client.login(authToken=token, silent=True)
                else:
                    client.login(silent=True)
            except Exception:
                print("Synapse authentication failed. For hidden token entry, rerun with --prompt-token.")
                print("Create a Personal Access Token in Synapse Settings with View and Download permissions.")
                print("Sign in at https://www.synapse.org and check the dataset's access requirements.")
                return False
        # READ metadata is public even when anonymous archive download is denied.
        metadata = client.restGET(f"/entity/{entity_id}/version/{version}")
        if metadata.get("id") != entity_id or metadata.get("versionNumber") != version:
            raise ValueError("Returned release does not match the requested entity/version")
        if metadata.get("concreteType") != "org.sagebionetworks.repo.model.FileEntity":
            raise ValueError("Requested entity is not a file; project-wide downloads are disabled")
        name = metadata.get("name", "")
        if (not name or Path(name).name != name or PureWindowsPath(name).name != name
                or ":" in name or not name.lower().endswith(".zip")):
            raise ValueError("Release metadata must name a safe ZIP archive")
        report = {
            "entity_id": entity_id, "version": version, "archive_name": name,
            "file_handle_id": metadata.get("dataFileHandleId"),
            "source_url": f"https://www.synapse.org/Synapse:{entity_id}",
            "checked_at_utc": datetime.now(timezone.utc).isoformat(),
            "expected_md5": None, "size_bytes": None,
            "status": "metadata_only" if metadata_only else "metadata_verified",
        }
        try:
            info = client.get(entity_id, version=version, downloadFile=False)
            report.update(_file_metadata(info))
        except Exception:
            report["file_metadata_note"] = "File checksum/size metadata is unavailable with current permissions"
        report_path = _save_report(destination, report)
        print(f"Synapse release: {entity_id}, version {version}, archive {name}")
        print(f"Expected MD5: {report['expected_md5'] or 'not available'}")
        print(f"Release metadata: {report_path}")
        if metadata_only:
            print("Metadata inspection complete; no archive downloaded.")
            return True
        print(f"Downloading the pinned archive into {destination}")
        downloaded = client.get(
            entity_id, version=version, downloadLocation=str(destination),
            downloadFile=True, ifcollision="keep.local",
        )
        archive = Path(_field(downloaded, "path", "")).resolve()
        if not _within(archive, destination) or not archive.is_file() or archive.name != name:
            raise ValueError("Downloader returned a file outside the requested archive destination")
        later_metadata = _file_metadata(downloaded)
        if later_metadata["expected_md5"]:
            report["expected_md5"] = later_metadata["expected_md5"]
        if later_metadata["size_bytes"] is not None:
            report["size_bytes"] = later_metadata["size_bytes"]
        report.update(_checksums(archive))
        report["archive_path"] = str(archive)
        expected_md5 = report.get("expected_md5")
        if expected_md5 and report["actual_md5"].lower() != str(expected_md5).lower():
            report["status"] = "checksum_mismatch"
            _save_report(destination, report)
            raise ValueError("Archive MD5 does not match Synapse metadata; extraction stopped")
        report["checksum_verified"] = bool(expected_md5)
        report["status"] = "downloaded"
        _save_report(destination, report)
        print(f"Downloaded MD5: {report['actual_md5']}; SHA256: {report['sha256']}")
        if extract:
            count = safe_extract_zip(archive, destination / "extracted", workspace_root=destination)
            report.update(status="extracted", extracted_files=count, extraction_path=str(destination / "extracted"))
            _save_report(destination, report)
            print(f"Extracted {count} files below {destination / 'extracted'}")
        return True
    except (ValueError, OSError, zipfile.BadZipFile) as error:
        # SDKs can raise these types too, so exception strings stay private.
        print(f"PolypGen download stopped ({type(error).__name__}). Check release metadata, destination, and archive integrity.")
        return False
    except Exception as error:
        # Third-party exception strings can contain headers; do not print them.
        print(f"Synapse request failed ({type(error).__name__}). Check credentials, dataset access, and network connectivity.")
        return False


def main() -> int:
    parser = argparse.ArgumentParser(description="Download the pinned PolypGen v3 ZIP from Synapse")
    parser.add_argument("--dest-dir", type=Path, default=Path(__file__).parent / "data", help="Download workspace; extraction stays below this directory")
    parser.add_argument("--entity-id", default=DEFAULT_ENTITY, help=f"File entity (default: {DEFAULT_ENTITY})")
    parser.add_argument("--version", type=int, default=DEFAULT_VERSION, help="Pinned entity version (default: 1)")
    parser.add_argument("--metadata-only", action="store_true", help="Read public release metadata without data download")
    parser.add_argument("--extract", action="store_true", help="Validate and extract ZIP into DEST_DIR/extracted")
    parser.add_argument("--prompt-token", action="store_true", help="Enter a Synapse token at a hidden local prompt (not saved)")
    args = parser.parse_args()
    if args.prompt_token and args.metadata_only:
        parser.error("--prompt-token is unnecessary with public --metadata-only inspection")
    token = None
    if args.prompt_token:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("error", getpass.GetPassWarning)
                token = getpass.getpass("Synapse personal access token (input hidden): ").strip()
        except (EOFError, KeyboardInterrupt, getpass.GetPassWarning):
            print("Token entry canceled or a hidden terminal prompt is unavailable. No download was started.")
            return 1
        if not token:
            print("No token entered. No download was started.")
            return 1
    return 0 if download_polypgen_from_synapse(
        args.dest_dir, auth_token=token, entity_id=args.entity_id, version=args.version,
        metadata_only=args.metadata_only, extract=args.extract,
    ) else 1


if __name__ == "__main__":
    sys.exit(main())
