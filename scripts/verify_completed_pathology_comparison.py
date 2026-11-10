"""Read-only integrity audit of completed pathology runs; never fits a model.

Writes a separate verification record and a snapshot of the final analysis sources.
Historical data, predictions, plans and the executed-source snapshot are preserved.
"""
from __future__ import annotations

import argparse
from collections import Counter
import datetime
import json
from pathlib import Path
import shutil
import xml.etree.ElementTree as ET

from compare_pathology_encoders import validate_comparison
from external_review_data import load_external_task
from fetch_pathology_checkpoints import validate_checkpoint
from review_protocol import ROOT, SEEDS, read_csv, sha256, write_json


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def csv_data_rows(path):
    # Prediction fields contain no embedded newlines; hashes below validate the bytes.
    with path.open('rb') as stream:
        return sum(block.count(b'\n') for block in iter(lambda: stream.read(4*1024*1024), b''))-1


def verify(plan_dir, output, tests):
    plan_dir, output, tests = (Path(p).resolve() for p in (plan_dir, output, tests))
    snapshot = plan_dir / 'analysis_source_snapshot'
    require(not output.exists() and not snapshot.exists(), 'Choose new verification/snapshot outputs')
    plan = read_json(plan_dir / 'comparison_plan.json')
    executed = read_json(plan_dir / 'executed_commands.json')
    execution_complete = read_json(plan_dir / 'EXECUTION_COMPLETE.json')
    require(execution_complete['executed_commands_sha256'] == sha256(plan_dir / 'executed_commands.json'),
            'Execution receipt differs from the recorded commands')
    require(executed['original_comparison_plan_sha256'] == sha256(plan_dir / 'comparison_plan.json'),
            'Original plan changed')
    require(not (plan_dir / 'EXECUTION_RUNNING').exists(), 'Executor is still active')
    original_snapshot = plan_dir / 'executed_source_snapshot'
    original_sources = read_json(original_snapshot / 'source_manifest.json')
    for record in original_sources['files']:
        require(sha256(original_snapshot / record['path']) == record['sha256'],
                f'Historical source snapshot changed: {record["path"]}')
    for name, digest in executed['code_sha256'].items():
        require(sha256(original_snapshot / 'scripts' / name) == digest,
                f'Executed source missing from preserved snapshot: {name}')

    reference_run = ROOT / 'reproduction_runs/review_2026-11-07_full'
    reference_report = ROOT / 'outputs/revision_2026-11-07/review_report'
    validate_comparison(plan, reference_run, reference_report, plan_dir / 'heads', plan_dir / 'report')
    report = read_json(plan_dir / 'report/provenance.json')
    require(report['worker_metrics_recomputed'] and report['target_support_verified_across_all_heads_protocols_seeds'],
            'Prediction-level report verification is incomplete')
    require(report['reporter_sha256'] == sha256(ROOT / 'scripts/report_review_revision.py'), 'Reporter source changed')
    aggregate = read_json(plan_dir / 'aggregate_uncertainty/provenance.json')
    require(aggregate['point_estimates_reconstruct_report'], 'Aggregate points were not reconstructed')
    require(aggregate['auditor_sha256'] == sha256(ROOT / 'scripts/audit_aggregate_uncertainty.py'),
            'Aggregate auditor source changed')
    require(aggregate['report_provenance_sha256'] == sha256(plan_dir / 'report/provenance.json'),
            'Aggregate intervals use another report')
    for name, digest in aggregate['input_hashes'].items():
        require(sha256(plan_dir / 'report' / name) == digest, f'Aggregate input changed: {name}')

    extraction, jobs = {}, []
    prediction_hashes = {(r['task'], int(r['seed'])): r['sha256'] for r in report['predictions']}
    for encoder in ('uni', 'conch'):
        validate_checkpoint(ROOT / 'checkpoints/pathology' / encoder / plan['models'][encoder]['filename'],
                            plan['models'][encoder])
        features, rows, splits, metadata = load_external_task(plan_dir / f'features_{encoder}/task.json')
        require(features.shape == (17200, plan['models'][encoder]['embedding_dimension']), 'Unexpected feature shape')
        cache = metadata['feature_metadata']
        require(cache['checkpoint_sha256'] == plan['models'][encoder]['sha256'], 'Wrong pretrained checkpoint')
        require(cache['initial_batch_bitwise_repeat_check'] and cache['raw_image_hashes_reverified'],
                'Feature extraction validation is incomplete')
        require(cache['extraction_script_sha256'] == executed['code_sha256']['extract_pathology_review_features.py'],
                'Executed extraction source differs from cached provenance')
        require(cache['checkpoint_validation']['missing_required_keys'] == 0, 'Required model weights are missing')
        extraction[encoder] = {'shape': list(features.shape), 'features_sha256': metadata['feature_sha256'],
                               'extraction_sha256': sha256(plan_dir / f'features_{encoder}/extraction.json'),
                               'checkpoint_sha256': cache['checkpoint_sha256'], 'model_revision': cache['model_revision']}
        del features, rows, splits
        task = f'camelyon_custom_{encoder}'
        for seed in SEEDS:
            folder = plan_dir / 'heads' / task / f'seed{seed}'
            completion = read_json(folder / 'COMPLETE.json')
            metrics = read_json(folder / 'metrics.json')
            digest = sha256(folder / 'predictions.csv')
            require(digest == completion['predictions_sha256'] == prediction_hashes[task, seed],
                    f'Predictions changed: {task}/seed{seed}')
            require(sha256(folder / 'source_heads_frozen.json') == completion['source_choices_sha256'],
                    f'Frozen source choices changed: {task}/seed{seed}')
            count = csv_data_rows(folder / 'predictions.csv')
            require(len(metrics) == completion['models'] == 44 and count == 149600, 'Incomplete job outputs')
            jobs.append({'task': task, 'seed': seed, 'evaluations': len(metrics), 'prediction_rows': count,
                         'predictions_sha256': digest, 'source_choices_sha256': completion['source_choices_sha256']})

    counts = {}
    for path, expected in (('report/image_metrics_by_seed.csv', 880),
                           ('comparison/image_metrics_seed_means.csv', 132),
                           ('comparison/patient_macro_seed_means.csv', 132),
                           ('aggregate_uncertainty/paired_aggregate_intervals.csv', 192)):
        count = len(read_csv(plan_dir / path))
        require(count == expected, f'Incomplete report: {path}')
        counts[path] = count
    stability = plan_dir / 'prototype_stability'
    pairs = read_csv(stability / 'prototype_stability_by_seed_pair.csv')
    identity = read_csv(stability / 'single_prototype_identity_checks.csv')
    representatives = read_csv(stability / 'source_prototype_representatives.csv')
    require(identity and all(r['status'] == 'verified' for r in identity), 'Centroid identity check failed')
    require(all(r['source_center'] in ('0', '3', '4') for r in representatives), 'A target entered source inspection')
    require(read_json(stability / 'provenance.json')['clinical_semantic_inspection_performed'] is False,
            'Do not infer a clinical inspection attestation')
    source_validation = read_json(plan_dir / 'source_gpu_validation/resource_selection.json')
    require(source_validation['source_images_only'], 'Resource selection was not source-only')
    require(source_validation['comparison_plan_sha256'] == sha256(plan_dir / 'comparison_plan.json'),
            'Resource selection uses another plan')
    valid_settings = [r for r in source_validation['records'] if r['status'] == 'valid']
    require(all(r['repeat_bitwise_equal'] and r['maximum_cpu_gpu_feature_difference'] <= 2e-5 for r in valid_settings),
            'Source numerical validation failed')
    suites = ET.parse(tests).getroot().findall('testsuite')
    require(suites and all(int(s.get('failures', 0)) == int(s.get('errors', 0)) == 0 for s in suites),
            'Targeted tests did not pass')

    # Historical manuscript and release archives are checked, not rewritten.
    historical = read_json(ROOT / 'outputs/revision_2026-11-07/verification.json')
    require(sha256(ROOT / 'outputs/introduction.tex') == historical['manuscript_checks']['main_manuscript_sha256']
            if 'manuscript_checks' in historical else
            sha256(ROOT / 'outputs/introduction.tex') == '1b469753ca601e168ed396365b39e925a9b08ebb99dce91be4a67a144be29f46',
            'Manuscript was changed after the recorded revision')
    for name, record in historical['archive_checksums'].items():
        require(sha256(ROOT / 'outputs/revision_2026-11-07' / name) == record['sha256'], 'Historical archive changed')
    assets = read_json(plan_dir / 'paper_assets/summary.json')
    for key, name in (('patch_summary_sha256', 'image_metrics_seed_means.csv'),
                      ('patient_macro_summary_sha256', 'patient_macro_seed_means.csv')):
        require(assets[key] == sha256(plan_dir / 'comparison' / name), 'Export uses another summary')
    require(assets['exporter_sha256'] == sha256(ROOT / 'scripts/export_pathology_comparison_report.py'),
            'Results exporter source changed')

    source_files = [ROOT / 'README.md', *(ROOT / 'docs' / name for name in
        ('UNI_CONCH_COMPARISON.md', 'EXTERNAL_EVALUATION.md', 'REVIEW_RESPONSE.md', 'REPRODUCIBILITY.md')),
        *(ROOT / 'scripts' / name for name in ('audit_aggregate_uncertainty.py',
        'export_pathology_comparison_report.py', 'prepare_pathology_comparison.py',
        'benchmark_pathology_extraction.py', 'verify_completed_pathology_comparison.py')),
        ROOT / 'tests/test_pathology_setup.py', ROOT / 'tests/test_aggregate_report_scope.py']
    snapshot.mkdir()
    source_records = []
    for source in source_files:
        relative = source.relative_to(ROOT)
        target = snapshot / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        source_records.append({'path': relative.as_posix(), 'sha256': sha256(target)})
    write_json(snapshot / 'source_manifest.json', {'files': source_records,
               'scope': 'final corrected analysis, setup and documentation; historical execution snapshot retained'})
    manifest_rows = read_csv(plan_dir / 'image_manifest.csv')
    record = {'verified_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
              'scope': 'custom same-cohort pretrained comparison; not full official WILDS or independent confirmation',
              'completed_task_seed_jobs': len(jobs), 'target_head_seed_evaluations': sum(r['evaluations'] for r in jobs),
              'prediction_rows': sum(r['prediction_rows'] for r in jobs), 'jobs': jobs, 'extraction': extraction,
              'report_counts': counts, 'recorded_patients': len({r['group_id'] for r in manifest_rows}),
              'patients_per_center': {c: len({r['group_id'] for r in manifest_rows if r['center'] == c})
                                     for c in sorted({r['center'] for r in manifest_rows})},
              'prototype_seed_pair_comparisons': len(pairs),
              'prototype_pair_status_counts': dict(Counter(r['status'] for r in pairs)),
              'single_prototype_identity_checks': len(identity),
              'maximum_single_prototype_coordinate_error': max(float(r['maximum_absolute_difference']) for r in identity),
              'nearest_source_records': len(representatives), 'clinical_semantic_review_performed': False,
              'targeted_tests_passed': sum(int(s.get('tests', 0)) for s in suites), 'tests_sha256': sha256(tests),
              'feature_and_prediction_hashes_verified': True, 'historical_source_snapshot_verified': True,
              'historical_manuscript_and_archives_unchanged': True,
              'analysis_source_manifest_sha256': sha256(snapshot / 'source_manifest.json'),
              'verification_script_sha256': sha256(Path(__file__)),
              'unexecuted': ['full official WILDS', 'independent patient-linked confirmation',
                             'modern DG backbone training', 'Linux execution', 'new manuscript figure compilation']}
    write_json(output, record)
    print(json.dumps({k: record[k] for k in ('completed_task_seed_jobs', 'target_head_seed_evaluations',
           'prediction_rows', 'prototype_seed_pair_comparisons', 'single_prototype_identity_checks',
           'maximum_single_prototype_coordinate_error', 'nearest_source_records', 'targeted_tests_passed')}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--tests', type=Path, required=True)
    args = parser.parse_args()
    verify(args.plan_dir, args.output, args.tests)
