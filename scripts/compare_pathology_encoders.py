"""Compare completed UNI/CONCH runs with the unchanged ten-seed ResNet18 run.

No placeholders are emitted: the same target identities, seeds, grids and
registered pretrained checkpoint fingerprints must be verified first.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from review_protocol import ROOT, read_csv, sha256, write_csv, write_json


def validate_comparison(plan, reference_run, reference_report, foundation_run, foundation_report):
    def read(path):
        return json.loads(Path(path).read_text(encoding='utf-8'))
    reference_manifest = read(reference_run / 'run_manifest.json')
    foundation_manifest = read(foundation_run / 'run_manifest.json')
    expected = {'camelyon_custom_uni', 'camelyon_custom_conch'}
    if set(foundation_manifest['tasks']) != expected:
        raise ValueError('Both registered foundation-encoder tasks must be completed')
    for folder, manifest in ((reference_run, reference_manifest), (foundation_run, foundation_manifest)):
        complete = read(folder / 'COMPLETE.json')
        if set(complete['tasks']) != set(manifest['tasks']) or complete['seeds'] != manifest['seeds']:
            raise ValueError('Incomplete fitting jobs')
        if manifest['seeds'] != plan['seeds'] or manifest['protocols'] != plan['protocols']:
            raise ValueError('Fitting seeds/protocols differ between encoders')
    if reference_manifest['candidate_grids'] != foundation_manifest['candidate_grids']:
        raise ValueError('Encoder tuning grids differ')
    reference_input = reference_manifest['inputs']['camelyon_resnet18']
    if reference_input['manifest_sha256'] != plan['original_manifest_sha256']:
        raise ValueError('Reference cohort differs from the audited comparison')
    if reference_input['feature_sha256'] != plan['reference']['feature_sha256']:
        raise ValueError('Reference feature matrix changed')
    for encoder in ('uni', 'conch'):
        source = foundation_manifest['inputs'][f'camelyon_custom_{encoder}']
        meta = source['feature_metadata']
        if (source['manifest_sha256'] != plan['portable_manifest_sha256']
                or meta['checkpoint_sha256'] != plan['models'][encoder]['sha256']
                or meta['model_revision'] != plan['models'][encoder]['revision']):
            raise ValueError('Foundation cohort or pinned pretrained weights differ')
    provenance = []
    for report, run in ((reference_report, reference_run), (foundation_report, foundation_run)):
        record = read(report / 'provenance.json')
        if record['run_manifest_sha256'] != sha256(run / 'run_manifest.json'):
            raise ValueError('Report does not belong to the completed run')
        provenance.append(record)
    for encoder in ('uni', 'conch'):
        for center in ('1', '2'):
            reference_support = provenance[0]['target_support_sha256'][f'camelyon_resnet18/{center}']
            new_support = provenance[1]['target_support_sha256'][f'camelyon_custom_{encoder}/{center}']
            if reference_support != new_support:
                raise ValueError('Target sample identities, labels or groups differ')


def compare(plan_dir, reference_run, reference_report, output):
    plan_dir, reference_run, reference_report, output = map(
        lambda p: Path(p).resolve(), (plan_dir, reference_run, reference_report, output))
    if output.exists():
        raise FileExistsError('Choose a new comparison-output directory')
    plan = json.loads((plan_dir / 'comparison_plan.json').read_text(encoding='utf-8'))
    run, report = plan_dir / 'heads', plan_dir / 'report'
    validate_comparison(plan, reference_run, reference_report, run, report)
    combined = {}
    task_encoders = {'camelyon_resnet18': 'ResNet18', 'camelyon_custom_uni': 'UNI', 'camelyon_custom_conch': 'CONCH'}
    for name in ('image_metrics_seed_means.csv', 'patient_macro_seed_means.csv'):
        reference = [r for r in read_csv(reference_report / name) if r['task'] == 'camelyon_resnet18']
        updated = read_csv(report / name)
        records = [dict(encoder=task_encoders[r['task']], **r) for r in reference + updated]
        if not records or any(int(r['fitting_seeds']) != 10 for r in records):
            raise ValueError('Every reported comparison must include ten fitting seeds')
        combined[name] = records
    output.mkdir(parents=True)
    for name, records in combined.items():
        write_csv(output / name, records)
    image = {(r['encoder'], r['protocol'], r['center'], r['method']): r
             for r in combined['image_metrics_seed_means.csv']}
    for protocol in plan['protocols']:
        methods = sorted({key[3] for key in image if key[1] == protocol})
        for metric, label in (('balanced_accuracy', 'balanced accuracy'), ('auroc', 'AUROC')):
            tex = [r'\begin{table*}[t]', r'\centering\small', r'\setlength{\tabcolsep}{3pt}',
                   r'\begin{tabular}{lrrrrrr}', r'\hline',
                   r'Head & ResNet18 H2 & UNI H2 & CONCH H2 & ResNet18 H1 & UNI H1 & CONCH H1 \\', r'\hline']
            for method in methods:
                values = []
                for center in ('2', '1'):
                    for encoder in ('ResNet18', 'UNI', 'CONCH'):
                        row = image[encoder, protocol, center, method]
                        values.append(f"${100*float(row[metric]):.2f}\\pm{100*float(row[metric+'_seed_sd']):.2f}$")
                tex.append(method.replace('_', r'\_') + ' & ' + ' & '.join(values) + r' \\')
            tex.extend([r'\hline', r'\end{tabular}',
                        r'\caption{' + f"Patch {label} (percent, mean and fitting-seed SD) on identical sampled patients; "
                        + f"{protocol} grids. H2 is primary and H1 secondary. Model-specific preprocessing is preserved. "
                        + "This custom subset comparison is descriptive and is not an official WILDS leaderboard result.}",
                        r'\end{table*}'])
            (output / f'{protocol}_{metric}.tex').write_text('\n'.join(tex) + '\n', encoding='utf-8')
    write_json(output / 'provenance.json', {'comparison_plan_sha256': sha256(plan_dir / 'comparison_plan.json'),
               'reference_report_sha256': sha256(reference_report / 'provenance.json'),
               'foundation_report_sha256': sha256(report / 'provenance.json'),
               'same_target_support_seeds_grids_verified': True,
               'scope': 'descriptive encoder-configuration comparison; no cross-encoder population inference'})
    print(f'Compared three encoders: {len(combined["image_metrics_seed_means.csv"])} patch-score rows')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan-dir', type=Path, required=True)
    parser.add_argument('--reference-run', type=Path, default=ROOT / 'reproduction_runs/review_2026-11-07_full')
    parser.add_argument('--reference-report', type=Path, default=ROOT / 'outputs/revision_2026-11-07/review_report')
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    compare(args.plan_dir, args.reference_run, args.reference_report, args.output_dir)
