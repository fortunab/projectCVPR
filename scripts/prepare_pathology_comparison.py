"""Audit the existing histology cohort and freeze executable UNI/CONCH commands."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path, PurePosixPath
import shlex

from review_protocol import EXT, ROOT, SEEDS, load_cached_task, read_csv, sha256, write_csv, write_json


IDENTITY = ("sample_id", "label", "center", "group_id", "sha256", "pixel_sha256")


def image_location(row, root):
    """Relocate an explicitly recorded path; a content hash must still match."""
    raw = row["image_path"].replace("\\", "/")
    path = Path(raw)
    if not path.is_file():
        marker = "isolated_datasets/polypgen_extensions_2026-10-20/exp5_camelyon/images/"
        if raw.count(marker) != 1:
            raise ValueError(f"Cannot relocate the recorded image path for {row['sample_id']}")
        tail = PurePosixPath(raw.split(marker, 1)[1])
        if tail.is_absolute() or ".." in tail.parts:
            raise ValueError("Unsafe recorded image suffix")
        path = root / marker / Path(*tail.parts)
    resolved = path.resolve()
    if not resolved.is_relative_to(root.resolve()):
        raise ValueError("Expected the audited images within the project tree")
    if not resolved.is_file() or sha256(resolved) != row["sha256"]:
        raise ValueError(f"Missing or altered source image: {row['sample_id']}")
    return resolved


def runtime_verification_metadata(path):
    """Validate an explicitly supplied smoke check; never discover old local proofs."""
    if path is None:
        return {}
    path = Path(path).resolve()
    runtime = json.loads(path.read_text(encoding='utf-8'))
    if runtime['extractor_sha256'] != sha256(ROOT / 'scripts/extract_pathology_review_features.py'):
        raise ValueError('Provider runtime check belongs to another extractor revision')
    return {'untrained_runtime_verification_sha256': sha256(path)}


def prepare(output, registry, checkpoint_root, runtime_verification=None):
    output, registry, checkpoint_root = map(lambda p: Path(p).resolve(), (output, registry, checkpoint_root))
    if output.exists():
        raise FileExistsError("Choose a new plan directory; existing results are preserved")
    runtime_metadata = runtime_verification_metadata(runtime_verification)
    manifest = EXT / "exp5_camelyon/audit/image_manifest.csv"
    features, cached_rows, splits, reference = load_cached_task("camelyon_resnet18")
    del features
    original = read_csv(manifest)
    identity = lambda records: [[str(row[field]) for field in IDENTITY] for row in records]
    if identity(original) != identity(cached_rows):
        raise ValueError("Raw manifest and reference feature cohort differ")
    portable = []
    for i, row in enumerate(original):
        image = image_location(row, ROOT)
        portable.append(dict(row, image_path=image.relative_to(ROOT).as_posix()))
        if i % 2000 == 0:
            print(f"Verified original image bytes: {i}/{len(original)}", flush=True)
    assert identity(portable) == identity(original)
    models = json.loads(registry.read_text(encoding="utf-8"))["models"]
    if set(models) != {"uni", "conch"}:
        raise ValueError("The paired comparison requires both original encoders")
    output.mkdir(parents=True)
    portable_manifest = output / "image_manifest.csv"
    write_csv(portable_manifest, portable)
    clinical_hash = hashlib.sha256(json.dumps(identity(original), separators=(",", ":")).encode()).hexdigest()
    commands = {"fetch": ["scripts/fetch_pathology_checkpoints.py", "--registry", str(registry),
                           "--dest", str(checkpoint_root)], "extract": {}}
    for encoder, model in models.items():
        commands["extract"][encoder] = [
            "scripts/extract_pathology_review_features.py", "--manifest", str(portable_manifest),
            "--image-root", str(ROOT), "--encoder", encoder,
            "--checkpoint", str(checkpoint_root / encoder / model["filename"]),
            "--expected-checkpoint-sha256", model["sha256"],
            "--model-revision", model["revision"], "--task", f"camelyon_custom_{encoder}",
            "--protocol", "custom_source_only", "--output-dir", str(output / f"features_{encoder}"),
            "--device", "cpu", "--batch-size", "4", "--threads", "4",
        ]
    commands["fit"] = ["scripts/run_external_review.py"]
    for encoder in models:
        commands["fit"].extend(["--task-spec", str(output / f"features_{encoder}/task.json")])
    commands["fit"].extend(["--run-dir", str(output / "heads"), "--protocols", "historical", "matched4", "factorial",
                             "--seeds", *map(str, SEEDS), "--threads", "4", "--workers", "1"])
    commands["report"] = ["scripts/report_review_revision.py", "--run-dir", str(output / "heads"),
                           "--output-dir", str(output / "report"), "--attempts", "2000", "--seed", "2027"]
    commands["aggregate"] = ["scripts/audit_aggregate_uncertainty.py", "--report-dir", str(output / "report"),
                              "--output-dir", str(output / "aggregate_uncertainty")]
    commands["compare"] = ["scripts/compare_pathology_encoders.py", "--plan-dir", str(output),
                            "--output-dir", str(output / "comparison")]
    code = ("prepare_pathology_comparison.py", "fetch_pathology_checkpoints.py", "extract_pathology_review_features.py",
            "run_external_review.py", "external_review_data.py", "run_review_revision.py", "review_protocol.py",
            "compare_pathology_encoders.py", "verify_pathology_runtime.py")
    plan = {"scope": "same custom CAMELYON cohort as ResNet18; not the complete official benchmark",
            "status": "prepared; weights/extraction/fitting not executed",
            "rows": len(original), "recorded_patients": len({r['group_id'] for r in original}),
            "rows_per_center": dict(Counter(r['center'] for r in original)),
            "patients_per_center": {c: len({r['group_id'] for r in original if r['center'] == c})
                                    for c in sorted({r['center'] for r in original})},
            "original_manifest_sha256": sha256(manifest), "portable_manifest_sha256": sha256(portable_manifest),
            "ordered_clinical_identity_sha256": clinical_hash,
            "path_relocation_only": True, "all_raw_image_bytes_verified": True,
            "reference": reference, "outer_splits": splits, "models": models,
            "registry_sha256": sha256(registry), "seeds": list(SEEDS),
            "protocols": ["historical", "matched4", "factorial"], "expected_task_seed_jobs": 20,
            "expected_target_head_seed_evaluations": 880,
            "code_sha256": {name: sha256(ROOT / 'scripts' / name) for name in code},
            "test_data_previously_inspected": True, "new_independent_hospitals": False,
            "provider_pretraining_overlap_independently_verified": False}
    plan.update(runtime_metadata)
    plan['provider_requirements_sha256'] = sha256(ROOT / 'requirements/pathology-cpu.txt')
    write_json(output / "comparison_plan.json", plan)
    write_json(output / "commands.json", commands)
    ps = ["# Generated local commands. Obtain model access and log in locally first.",
          "$ErrorActionPreference = 'Stop'", f"Set-Location -LiteralPath '{str(ROOT).replace(chr(39), chr(39)*2)}'",
          "$ProviderPython = (Resolve-Path '.venv-pathology-cpu/Scripts/python.exe').Path",
          "$ReviewPython = (Resolve-Path '.venv-cbmp-review/Scripts/python.exe').Path",
          f"New-Item -ItemType Directory -Path '{str(output / 'logs').replace(chr(39), chr(39)*2)}' -Force | Out-Null"]
    bash = ["#!/usr/bin/env bash", "# Generated for this local tree; prepare again after relocating the project.",
            "set -euo pipefail", "cd " + shlex.quote(str(ROOT)),
            'ProviderPython=".venv-pathology-cpu/bin/python"', 'ReviewPython=".venv-cbmp-review/bin/python"',
            'mkdir -p ' + shlex.quote(str(output / 'logs'))]
    steps = [("ProviderPython", commands['fetch'], 'checkpoint_download')]
    steps += [("ProviderPython", args, 'extract_' + encoder) for encoder, args in commands['extract'].items()]
    steps += [("ReviewPython", commands['fit'] + ['--prepare-only'], 'fitting_plan'),
              ("ReviewPython", commands['fit'] + ['--resume'], 'fitting'),
              ("ReviewPython", commands['report'], 'report'),
              ("ReviewPython", commands['aggregate'], 'aggregate'), ("ReviewPython", commands['compare'], 'comparison')]
    for python, args, stage in steps:
        log = output / 'logs' / (stage + '.log')
        ps.append("& $" + python + " -B " + " ".join("'" + str(x).replace("'", "''") + "'" for x in args)
                  + " 2>&1 | Tee-Object -FilePath '" + str(log).replace("'", "''") + "'")
        ps.append("if ($LASTEXITCODE -ne 0) { throw 'Comparison stage failed; inspect its output before retrying' }")
        bash.append('"$' + python + '" -B ' + " ".join(shlex.quote(str(x)) for x in args)
                    + ' 2>&1 | tee ' + shlex.quote(str(log)))
    (output / 'run_comparison.ps1').write_text('\n'.join(ps) + '\n', encoding='utf-8')
    (output / 'run_comparison.sh').write_text('\n'.join(bash) + '\n', encoding='utf-8')
    print(f"Prepared {len(original)} images, {plan['recorded_patients']} patients; 20 fitting jobs await pretrained weights")
    return plan


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--registry', type=Path, default=ROOT / 'configs/pathology_models.json')
    parser.add_argument('--checkpoint-root', type=Path, default=ROOT / 'checkpoints/pathology')
    parser.add_argument('--runtime-verification', type=Path,
                        help='Optional explicit untrained smoke-check JSON for this extractor revision')
    args = parser.parse_args()
    prepare(args.output_dir, args.registry, args.checkpoint_root, args.runtime_verification)
