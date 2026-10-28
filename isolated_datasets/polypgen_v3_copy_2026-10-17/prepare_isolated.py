"""Extract and snapshot code inside this isolated workspace only."""
from __future__ import annotations

import hashlib
import json
import shutil
import stat
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parent
ARCHIVE = ROOT / "PolypGen2021_MultiCenterData_v3 (1).zip"
EXTRACTED = ROOT / "extracted"
SOURCE = ROOT.parents[1] / "work" / "family_baselines" / "polypgen"


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    marker = ROOT / "extraction_complete.json"
    if EXTRACTED.exists() or marker.exists():
        raise FileExistsError("Extraction directory already exists; refusing to overwrite it")
    with zipfile.ZipFile(ARCHIVE) as archive:
        members = archive.infolist()
        total = sum(member.file_size for member in members)
        if shutil.disk_usage(ROOT).free < total + 1024 ** 3:
            raise OSError("Insufficient free space for extraction and outputs")
        for member in members:
            target = (EXTRACTED / member.filename).resolve()
            if not target.is_relative_to(EXTRACTED.resolve()):
                raise ValueError(f"Unsafe archive path: {member.filename}")
            if stat.S_ISLNK(member.external_attr >> 16):
                raise ValueError(f"Archive contains a symbolic link: {member.filename}")
        EXTRACTED.mkdir()
        print(f"Extracting {len(members)} entries, {total / 1e9:.2f} GB uncompressed", flush=True)
        for index, member in enumerate(members, 1):
            archive.extract(member, EXTRACTED)
            if index % 2000 == 0 or index == len(members):
                print(f"Extracted {index}/{len(members)} entries (CRC checked during extraction)", flush=True)
    code = ROOT / "code"
    code.mkdir()
    hashes = {}
    for source in sorted(SOURCE.glob("*.py")):
        destination = code / source.name
        shutil.copy2(source, destination)
        hashes[source.name] = sha256(destination)
    for name in ("requirements.txt", "README_CBMP.md"):
        shutil.copy2(SOURCE / name, code / name)
    shutil.copytree(SOURCE / "tests", code / "tests", ignore=shutil.ignore_patterns("__pycache__", ".pytest_cache"))
    audit = ROOT / "audit"
    audit.mkdir()
    for name in ("polypgen_sequence_center_mapping_review_v2.json", "polypgen_sequence_class_audit.json"):
        shutil.copy2(ROOT.parents[1] / "work" / name, audit / name)
    result = {
        "archive": str(ARCHIVE), "archive_bytes": ARCHIVE.stat().st_size,
        "archive_sha256": sha256(ARCHIVE), "zip_entries": len(members),
        "uncompressed_bytes": total, "extracted": str(EXTRACTED),
        "source_code": str(SOURCE), "snapshot_sha256": hashes,
        "scope": "All new extraction, audit, features, and run outputs remain inside this isolated directory.",
        "nested_archives": "Retained as archive members; not recursively expanded or used as model inputs."
    }
    marker.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"Extraction and code snapshot complete: {ROOT}", flush=True)


if __name__ == "__main__":
    main()
