"""Audit the full local release and prepare a conservative sequence subset."""
from __future__ import annotations

import csv
import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "code"))
from polypgen_manifest import IMAGE_EXTENSIONS, _context, _is_original, _label, build_manifest
from cbmp_data import load_manifest, make_outer_splits

DATA = ROOT / "extracted" / "PolypGen2021_MultiCenterData_v3"
AUDIT = ROOT / "audit"
DOI = "https://doi.org/10.1038/s41597-023-01981-y"


def write_csv(path, rows, fields):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    if (AUDIT / "AUDIT_COMPLETE").exists():
        raise FileExistsError("Completed audit already exists; refusing to overwrite it")
    review = json.loads((AUDIT / "polypgen_sequence_center_mapping_review_v2.json").read_text())
    mapping, metadata_rows = {}, []
    for kind in ("positive", "negative"):
        for number, center in review[f"{kind}_mapping_verified"].items():
            seq = f"seq{number}" + ("_neg" if kind == "negative" else "")
            if seq == "seq1":
                continue  # separate unresolved annotation-quality concern, fixed before fitting
            direct = ((kind == "positive" and int(number) >= 16)
                      or (kind == "negative" and 11 <= int(number) <= 13))
            source = (f"Official v3 original filenames: every frame has explicit {center} token"
                      if direct else f"Ali et al. 2023 Figure 7, explicit label and center color; {DOI}")
            mapping[seq] = {"center": center, "source": source, "direct_filename": direct}
            metadata_rows.append({"sequence_id": seq, "center": center, "source": source})
    metadata_path = AUDIT / "reviewed_sequence_centers.csv"
    write_csv(metadata_path, metadata_rows, ["sequence_id", "center", "source"])
    inventory, folder_cache = [], {}
    candidates = sorted(p for p in DATA.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)
    for image in candidates:
        if not _is_original(image) or image.name.startswith("._") or "__MACOSX" in image.parts:
            continue
        center, seq, negative = _context(image, DATA)
        details = mapping.get(seq)
        status = "included_verified_sequence" if details else "excluded_no_sequence_group" if not seq else "excluded_unresolved_center"
        if seq in {"seq11", "seq7_neg"}:
            status = "excluded_conflicting_center_sources"
        elif seq == "seq1":
            status = "excluded_pending_annotation_review"
        elif details:
            center = details["center"]
            if details["direct_filename"]:
                tokens = set(re.findall(r"(?:^|[_-])(C[1-6])(?=[_-]|$)", image.stem, re.I))
                if {token.upper() for token in tokens} != {center}:
                    raise ValueError(f"Filename provenance check failed: {image}")
        label, label_source, error = "", "", ""
        try:
            observed = _label(image, negative, 127, 1, folder_cache)
            label, label_source = observed["label"], observed["label_source"]
        except (OSError, ValueError) as exc:
            error = str(exc)
            status = "excluded_annotation_error"
        inventory.append({"relative_path": image.relative_to(DATA).as_posix(), "sequence_id": seq,
                          "center": center or "", "label": label, "label_source": label_source,
                          "status": status, "annotation_error": error})
        if len(inventory) % 500 == 0:
            print(f"Audited annotations for {len(inventory)} original images", flush=True)
    write_csv(AUDIT / "full_release_inventory.csv", inventory, list(inventory[0]))
    sequence_rows = []
    for seq in sorted({r["sequence_id"] for r in inventory if r["sequence_id"]}):
        rows = [r for r in inventory if r["sequence_id"] == seq]
        sequence_rows.append({"sequence_id": seq, "center": rows[0]["center"], "status": rows[0]["status"],
                              "negative_frames": sum(r["label"] == 0 for r in rows),
                              "positive_frames": sum(r["label"] == 1 for r in rows),
                              "annotation_errors": sum(bool(r["annotation_error"]) for r in rows),
                              "source": mapping.get(seq, {}).get("source", "")})
    write_csv(AUDIT / "sequence_class_center_audit.csv", sequence_rows, list(sequence_rows[0]))
    print("Constructing strict manifest, image hashes, and decoded-pixel duplicate groups", flush=True)
    audit = build_manifest(DATA, AUDIT / "manifest.csv", sequence_metadata=metadata_path,
                           sequence_only=True, verified_sequences_only=True)
    if audit["blocking_rejections"]:
        raise ValueError("Blocking strict-audit rejections remain; inspect manifest.rejected.csv")
    rows = load_manifest(AUDIT / "manifest.csv")
    splits = make_outer_splits(rows, seed=17)
    center_rows = []
    for center in ("C1", "C2", "C3", "C4", "C5", "C6"):
        selected = [r for r in rows if r["center"] == center]
        eligible = next((s for s in splits if s["heldout_center"] == center), None)
        center_rows.append({"center": center, "negative_frames": sum(r["label"] == 0 for r in selected),
                            "positive_frames": sum(r["label"] == 1 for r in selected),
                            "sequences": len({r["sequence_id"] for r in selected}),
                            "negative_sequences": len({r["sequence_id"] for r in selected if r["label"] == 0}),
                            "positive_sequences": len({r["sequence_id"] for r in selected if r["label"] == 1}),
                            "fold_status": eligible["status"] if eligible else "absent_from_verified_subset"})
    write_csv(AUDIT / "center_class_eligibility.csv", center_rows, list(center_rows[0]))
    summary = {
        "scope": "Preliminary reviewed-sequence subset of PolypGen v3, not complete-release evaluation",
        "archive_sha256": json.loads((ROOT / "extraction_complete.json").read_text())["archive_sha256"],
        "mask_rule": "intensity >127, at least one foreground pixel; matching boxes must agree",
        "inventory_originals": len(inventory), "inventory_status_counts": dict(Counter(r["status"] for r in inventory)),
        "annotation_errors": [r for r in inventory if r["annotation_error"]],
        "verified_sequences": len(mapping), "included_frames": len(rows),
        "eligible_heldout_centers": [s["heldout_center"] for s in splits if s["status"] == "ready"],
        "centers": center_rows,
        "limitations": [
            "Patient identities are unavailable; video grouping does not establish patient independence.",
            "Still images have center IDs but no documented sequence/patient grouping and are excluded.",
            "Unresolved positive seq4,7,8,9,14,15 and disputed seq11,seq7_neg are excluded.",
            "seq1 is excluded pending annotation-quality review; all current masks and boxes label its36frames negative.",
            "C2 negatives occur in only one sequence; C5 positives occur in only one sequence.",
            "Center C1 is negative-only in this subset and is source-only; C3 is absent.",
            "Current draft experiments use cached ImageNet ResNet18, not DINOv2 or an adapted encoder."
        ],
        "metadata_sha256": hashlib.sha256(metadata_path.read_bytes()).hexdigest()
    }
    (AUDIT / "eligibility_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (AUDIT / "AUDIT_COMPLETE").write_text("Full inventory and verified subset eligibility audited.\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
