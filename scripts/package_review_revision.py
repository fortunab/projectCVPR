"""Build a narrow anonymous prediction-analysis package from completed evidence."""

from __future__ import annotations

import argparse
import json
import hashlib
from pathlib import Path
import shutil
import subprocess
import sys
import zipfile

from review_protocol import BASE, BUNDLE, CORE, EXT, ROOT, TASKS, read_csv, sha256, write_csv, write_json
from materialize_review_predictions import pack_file, restore_file


README = """# Controlled medical classifier study: analysis reproduction

This package contains the ten-seed predictions, source-selected settings, report
scripts, platform wheel locks, tests and verification records supporting the
paper. It is a **prediction-analysis package**. It omits raw medical images,
model checkpoints, fitted binary models and source-validation fit caches.
Original feature caches are needed for refitting, and are not supplied by this
archive. The source files for that route are included for inspection.

From this directory, with CPython 3.12:

Linux/WSL:
```
python3.12 -m venv .venv-review
source .venv-review/bin/activate
python -m pip install --require-hashes -r requirements/review-linux.lock
python -m pip check
mkdir -p tmp
python -m pytest tests/test_review_revision.py tests/test_external_review.py -q --basetemp tmp/analysis-tests-linux
python -B scripts/materialize_review_predictions.py --run-dir results
python -B scripts/report_review_revision.py --run-dir results --output-dir replay --attempts 2000 --seed 2027
python -B scripts/audit_aggregate_uncertainty.py --report-dir replay --output-dir replay-aggregates
```

PowerShell:
```
py -3.12 -m venv .venv-review
$ReviewPython = (Resolve-Path '.\\.venv-review\\Scripts\\python.exe').Path
& $ReviewPython -m pip install --require-hashes -r requirements/review-windows.lock
if ($LASTEXITCODE -ne 0) { throw 'Installation failed' }
& $ReviewPython -m pip check
if ($LASTEXITCODE -ne 0) { throw 'Dependency check failed' }
New-Item -ItemType Directory -Path tmp -Force | Out-Null
& $ReviewPython -m pytest tests/test_review_revision.py tests/test_external_review.py -q --basetemp tmp/analysis-tests-windows
if ($LASTEXITCODE -ne 0) { throw 'Protocol tests failed' }
& $ReviewPython -B scripts/materialize_review_predictions.py --run-dir results
if ($LASTEXITCODE -ne 0) { throw 'Prediction reconstruction failed' }
& $ReviewPython -B scripts/report_review_revision.py --run-dir results --output-dir replay --attempts 2000 --seed 2027
if ($LASTEXITCODE -ne 0) { throw 'Analysis replay failed' }
& $ReviewPython -B scripts/audit_aggregate_uncertainty.py --report-dir replay --output-dir replay-aggregates
if ($LASTEXITCODE -ne 0) { throw 'Aggregate uncertainty replay failed' }
```

Compare the numerical CSV files in `replay/` with `validation/report/`.
Repeated prediction blocks are stored as shared NPZ objects and JSON indexes.
Materialization restores the exact original CSV bytes and verifies their recorded
SHA256 hashes, without rounding any numeric field. It needs additional disk space
for the expanded CSVs but no medical image download.
`compute_usage.csv` is supplied as a recorded local fit-timing artifact and is not
reconstructed when binary fit caches are omitted. The replay's provenance also
records that omission. Prediction-derived metrics, rankings, selections, and
conditional intervals remain reconstructible. A fresh Windows numeric replay
was verified; Linux execution has not yet been tested. Data/model adapters have
fixture coverage, not completed official-WILDS or UNI/CONCH experiments.

Metrics are fractions; multiply by 100 for percentages or percentage points.
Seed SD concerns fitting variation on fixed cohorts. Group bootstrap intervals
condition on observed sequences/patients and fixed fitted heads, not a population
of hospitals. The study retains unresolved PolypGen patient linkage and a custom
histology subset. An additional seed is not an additional patient or hospital.

The frozen reference classifier files retain their original relative location
under `isolated_datasets/.../reproducibility/sources/baseline/code/`; this small
subtree contains only files required by the analysis imports and regression
checks. It is not a copy of the original working directories. `input_manifest.json`
lists the exact package files, byte sizes and SHA256 fingerprints.

See `docs/RUN_REVIEW_REVISION.md`, `EXTERNAL_EVALUATION.md`, and
`AUTHOR_REVIEW_RECORD.md` for the full local fitting commands, prerequisites and
scope of author versus automated assurance. Local fitting path examples in those
documents refer to the full research checkout, not to omitted archive inputs.
"""


def portable_caches(output):
    """Relocate paths, retaining labels/order and every numerical feature value."""
    import numpy as np
    records = []
    base = output / BASE.relative_to(ROOT)
    ext = output / EXT.relative_to(ROOT)
    manifest = BASE / "audit/manifest.csv"
    rows = read_csv(manifest)
    marker = "/PolypGen2021_MultiCenterData_v3/"
    for row in rows:
        row["image_path"] = row["relative_path"]
        for key in ("mask_path", "bbox_path"):
            value = row[key].replace("\\", "/")
            if value:
                if marker not in value:
                    raise ValueError("Unexpected annotation path during portable relocation")
                row[key] = value.split(marker, 1)[1]
    relocated_manifest = base / "audit/manifest.csv"
    write_csv(relocated_manifest, rows)
    audit = json.loads(manifest.with_suffix(".audit.json").read_text())
    original_audit_sha256 = sha256(manifest.with_suffix(".audit.json"))
    audit.update(data_root="RAW_POLYPGEN_ROOT", manifest="audit/manifest.csv",
                 sequence_metadata="audit/reviewed_sequence_centers.csv", manifest_sha256=sha256(relocated_manifest))
    write_json(relocated_manifest.with_suffix(".audit.json"), audit)
    shutil.copyfile(BASE / "audit/reviewed_sequence_centers.csv", base / "audit/reviewed_sequence_centers.csv")
    split = base / "results/seed17/outer_splits.json"
    split.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(BASE / "results/seed17/outer_splits.json", split)
    hist_manifest = EXT / "exp5_camelyon/audit/image_manifest.csv"
    hist_rows = read_csv(hist_manifest)
    for row in hist_rows:
        row["image_path"] = f"images/patient_{int(row['patient']):03d}/{row['sample_id']}"
    relocated_hist = ext / "exp5_camelyon/audit/image_manifest.csv"
    write_csv(relocated_hist, hist_rows)
    cohort = json.loads((EXT / "exp5_camelyon/cohort_plan.json").read_text())
    cohort["selected_manifest_sha256"] = sha256(relocated_hist)
    write_json(ext / "exp5_camelyon/cohort_plan.json", cohort)
    sources = {
        "polypgen_resnet18": BASE / "features/resnet18_imagenet1k_v1.npz",
        "polypgen_dinov2": EXT / "exp4_dinov2/features/dinov2_base_cls.npz",
        "camelyon_resnet18": EXT / "exp5_camelyon/features/resnet18_imagenet1k_v1.npz",
    }
    for task, source in sources.items():
        destination = output / source.relative_to(ROOT)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with np.load(source, allow_pickle=False) as original:
            features = original["features"]
            metadata = json.loads(str(original["metadata"].item()))
        selected_rows = hist_rows if task.startswith("camelyon") else rows
        if task.startswith("camelyon"):
            metadata["manifest_sha256"] = sha256(relocated_hist)
        original_matrix = hashlib.sha256(np.ascontiguousarray(features).tobytes()).hexdigest()
        np.savez_compressed(destination, features=features, metadata=json.dumps(metadata),
                            sample_ids=np.asarray([r["sample_id"] for r in selected_rows]))
        with np.load(destination, allow_pickle=False) as relocated:
            if not np.array_equal(features, relocated["features"]):
                raise ValueError("Portable feature values changed")
        records.append({"task": task, "original_archive_sha256": sha256(source),
                        "portable_archive_sha256": sha256(destination), "numerical_feature_sha256": original_matrix,
                        "shape": list(features.shape), "dtype": str(features.dtype),
                        "labels_centers_groups_content_hashes_order": "preserved",
                        "changes": "relative image/annotation paths; explicit sample IDs; histology manifest hash rebound"})
    probe = '''import hashlib,json,sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(sys.argv[1])/'scripts'))
from review_protocol import TASKS,load_cached_task
result={}
for task in TASKS:
 features,rows,splits,proof=load_cached_task(task)
 result[task]=hashlib.sha256(np.ascontiguousarray(features).tobytes()).hexdigest()
print(json.dumps(result))
'''
    check = subprocess.run([sys.executable, "-B", "-c", probe, str(output)], capture_output=True, text=True, check=True)
    values = json.loads(check.stdout)
    if any(values[r["task"]] != r["numerical_feature_sha256"] for r in records):
        raise ValueError("Portable protocol loader changed numerical feature values")
    write_json(output / "validation/portable_cache_verification.json", {
        "caches": records, "all_three_portable_protocol_loads": "passed",
        "original_polypgen_manifest_sha256": sha256(manifest),
        "portable_polypgen_manifest_sha256": sha256(relocated_manifest),
        "original_polypgen_audit_sha256": original_audit_sha256,
        "original_histology_manifest_sha256": sha256(hist_manifest),
        "portable_histology_manifest_sha256": sha256(relocated_hist),
        "scope": "cache relocation, exact array equality, original semantic manifest digest and protocol checks",
        "raw_images_reverified_during_packaging": False,
    })


def package(run_dir, report_dir, stability_dir, aggregate_dir, output, include_caches=False):
    plan = json.loads((run_dir / "run_manifest.json").read_text())
    done = json.loads((run_dir / "COMPLETE.json").read_text())
    report = json.loads((report_dir / "provenance.json").read_text())
    if (set(plan["tasks"]) != set(TASKS) or len(plan["seeds"]) != 10 or plan["seeds"] != done["seeds"]
            or sha256(run_dir / "run_manifest.json") != report["run_manifest_sha256"]):
        raise ValueError("Completed, verified ten-seed evidence is required")
    if output.exists():
        raise FileExistsError("Choose a new release directory")
    output.mkdir(parents=True)
    readme = README
    if include_caches:
        readme = readme.replace("**prediction-analysis package**", "**prediction-analysis and cached-refitting package**")
        readme = readme.replace("Original feature caches are needed for refitting, and are not supplied by this\narchive. The source files for that route are included for inspection.",
                                "Portable copies of the three audited feature matrices and their manifests\nare included for refitting. Their exact numerical identity is verified below.")
        readme += """
## Optional portable feature-cache bundle included

This variant additionally includes the three audited numerical feature matrices
and portable manifests needed by `run_review_revision.py`. Image/annotation paths
were made relative and the histology manifest hash rebound. Every feature value,
label, group, center, content hash and row order is preserved. The verification
record gives original/portable archive hashes and exact numerical-matrix hashes.
All three task loaders were checked in a fresh process against these relocated
files. No new raw-image audit or patient-linkage resolution is implied.

After the locked installation above, refit using a new directory:
```
python -B scripts/run_review_revision.py --run-dir refit --prepare-only
python -B scripts/run_review_revision.py --run-dir refit --resume
python -B scripts/report_review_revision.py --run-dir refit --output-dir refit-report
```
In PowerShell prefix these commands with `& $ReviewPython` in place of `python`.
Use `--tasks polypgen_resnet18 --seeds 17 --protocols historical --workers 1`
on both planning/execution commands for a small replay. The refit's manifest
records relocated archive hashes; compare numeric predictions and selected
settings, rather than timing-bearing binary file hashes.

Dataset attribution: PolypGen (Ali et al., Scientific Data 2023,
https://doi.org/10.1038/s41597-023-01981-y), CAMELYON17 (Bandi et al.,
https://doi.org/10.1109/TMI.2018.2867350), and WILDS
(https://wilds.stanford.edu/). Original data/model terms remain applicable.
Raw images and pretrained model weights are not included.
"""
    (output / "README.md").write_text(readme, encoding="utf-8")
    selected = []
    script_names = ["review_protocol.py", "run_review_revision.py", "report_review_revision.py",
                    "audit_review_stability.py", "export_review_tables.py", "export_review_supplement.py", "external_review_data.py",
                    "prepare_official_wilds.py", "extract_pathology_review_features.py",
                    "run_external_review.py", "evaluate_official_wilds.py", "run_reference_tests.py",
                    "materialize_review_predictions.py", "audit_aggregate_uncertainty.py"]
    for name in script_names:
        selected.append((ROOT / "scripts" / name, Path("scripts") / name))
    for path in (ROOT / "tests").glob("test_*review*.py"):
        selected.append((path, Path("tests") / path.name))
    for path in (ROOT / "requirements").glob("review-*"):
        selected.append((path, Path("requirements") / path.name))
    for name in ("RUN_REVIEW_REVISION.md", "EXTERNAL_EVALUATION.md", "REPRODUCIBILITY.md",
                 "REVIEW_RESPONSE.md", "AUTHOR_REVIEW_RECORD.md", "CONFIRMATION_PROTOCOL.md"):
        selected.append((ROOT / "docs" / name, Path("docs") / name))
    paper = ROOT / "outputs/revision_2026-11-04/overleaf"
    paper_names = ("main.tex", "introduction.tex", "references.bib", "preamble.tex", "cvpr.sty",
                   "ieeenat_fullname.bst", "supplement.tex", "full_ten_seed_tables.tex",
                   "review_factorial_intervals.tex", "review_ba_interactions.tex", "review_budget_intervals.tex",
                   "review_group_endpoints.tex", "review_aggregate_intervals.tex", "review_stability.tex")
    for name in paper_names:
        selected.append((paper / name, Path("paper") / name))
    for name in ("cbmp.py", "cbmp_data.py", "cbmp_evaluation.py", "polypgen_manifest.py"):
        selected.append((CORE / name, (CORE / name).relative_to(ROOT)))
    for name in ("sources_manifest.json", "cohort_order.json"):
        selected.append((BUNDLE / name, (BUNDLE / name).relative_to(ROOT)))
    for name in ("test_cbmp_weighted_heads.py", "test_cbmp_evaluation.py", "test_cbmp_manifest.py"):
        selected.append((BASE / "code/tests" / name, (BASE / "code/tests" / name).relative_to(ROOT)))
    for name in ("run_manifest.json", "COMPLETE.json", "pip_check.txt", "installed_packages.txt"):
        selected.append((run_dir / name, Path("results") / name))
    for task in plan["tasks"]:
        for seed in plan["seeds"]:
            folder = run_dir / task / f"seed{seed}"
            completed = json.loads((folder / "COMPLETE.json").read_text())
            if (sha256(folder / "predictions.csv") != completed["predictions_sha256"]
                    or sha256(folder / "source_heads_frozen.json") != completed["source_choices_sha256"]):
                raise ValueError("Result fingerprints changed before packaging")
            target = output / "results" / task / f"seed{seed}"
            target.mkdir(parents=True)
            pack_file(folder / "predictions.csv", target / "predictions.index.json", output / "results/prediction_objects")
            check_path = target / "predictions.verification.csv"
            restore_file(target / "predictions.index.json", output / "results/prediction_objects", check_path)
            check_path.unlink()
            for name in ("metrics.json", "source_heads_frozen.json", "COMPLETE.json", "run.log"):
                selected.append((folder / name, Path("results") / task / f"seed{seed}" / name))
    for source, target in ((report_dir, "report"), (stability_dir, "stability"), (aggregate_dir, "aggregates")):
        for path in source.iterdir():
            if path.suffix in (".json", ".csv"):
                selected.append((path, Path("validation") / target / path.name))
    evidence = ROOT / "outputs/revision_2026-11-07"
    for name in ("smoke_replay_verification.json", "portable_refit_verification.json", "test_verification.json", "core_regression_verification.json",
                 "core_regression_tests.txt", "hardware_observation.json", "linux_wheel_validation.json"):
        selected.append((evidence / name, Path("validation") / "checks" / name))
    for source, destination in selected:
        target = output / destination
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    if include_caches:
        portable_caches(output)
    private = []
    # Large prediction files contain only audited IDs and numeric output. Scan
    # all small code/docs/JSON plus the text headers of the CSV exports.
    for path in output.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix == ".csv":
            with path.open(encoding="utf-8-sig") as stream:
                content = "".join(next(stream, "") for _ in range(3))
        elif path.suffix in (".npz", ".npy"):
            continue
        else:
            content = path.read_text(encoding="utf-8-sig")
        if any(token in content.lower() for token in ("fortunab", "lenovo", "florin fortis")):
            private.append(path.relative_to(output).as_posix())
    if private:
        raise ValueError(f"Remove identifying material before publishing this package: {private}")
    files = [{"path": p.relative_to(output).as_posix(), "bytes": p.stat().st_size, "sha256": sha256(p)}
             for p in sorted(output.rglob("*")) if p.is_file()]
    write_json(output / "input_manifest.json", {"scope": "anonymous analysis and optional cached-refitting reproduction",
               "run_manifest_sha256": sha256(run_dir / "run_manifest.json"), "files": files,
               "omitted": ["raw images", "model checkpoints", "binary fitted/source fit caches"]
                          + ([] if include_caches else ["feature caches"]),
               "portable_feature_caches_included": include_caches,
               "identity_scan": "code/docs/JSON and CSV headers; no author identifiers matched"})
    archive = output.with_suffix(".zip")
    if archive.exists():
        raise FileExistsError("Release archive already exists")
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as bundle:
        for path in sorted(output.rglob("*")):
            if path.is_file():
                bundle.write(path, path.relative_to(output).as_posix())
    print(f"Packaged {len(files) + 1} files: {archive}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--report-dir", type=Path, required=True)
    parser.add_argument("--stability-dir", type=Path, required=True)
    parser.add_argument("--aggregate-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--include-caches", action="store_true")
    args = parser.parse_args()
    package(args.run_dir.resolve(), args.report_dir.resolve(), args.stability_dir.resolve(),
            args.aggregate_dir.resolve(), args.output_dir.resolve(), args.include_caches)
