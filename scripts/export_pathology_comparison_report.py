"""Export descriptive same-cohort encoder comparisons and native TikZ figures."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from review_protocol import ROOT, read_csv, sha256, write_json


ENCODERS = ('ResNet18', 'UNI', 'CONCH')
HEADS = (('cbmp', 'CBMP'), ('linear_logistic', 'Linear'), ('trainable_cosine', 'Cosine'))
COLORS = ('pathologyResnetBlue', 'pathologyUniOrange', 'pathologyConchGreen')


def figure(records, metric, label, endpoint, figure_label):
    cells = {(r['encoder'], r['protocol'], r['center'], r['method']): r for r in records}
    lines = [r'\begin{figure*}[t]', r'\centering', r'\begin{tikzpicture}[font=\scriptsize]',
             r'\definecolor{pathologyResnetBlue}{HTML}{0072B2}',
             r'\definecolor{pathologyUniOrange}{HTML}{E69F00}',
             r'\definecolor{pathologyConchGreen}{HTML}{009E73}']
    for index, (encoder, color) in enumerate(zip(ENCODERS, COLORS)):
        x = 4.1 + index*2.2
        lines.extend([f'\\fill[{color}] ({x:.2f},5.05) rectangle ({x+.22:.2f},5.23);',
                      f'\\node[anchor=west] at ({x+.30:.2f},5.14) {{{encoder}}};'])
    for panel, (center, title) in enumerate((('2', 'H2 (primary)'), ('1', 'H1 (secondary)'))):
        lines.append(f'\\begin{{scope}}[xshift={panel*8.1:.2f}cm]')
        lines.append(f'\\node[font=\\small\\bfseries] at (3.9,4.58) {{{title}}};')
        for tick in range(0, 101, 20):
            y = tick*.04
            lines.extend([f'\\draw[black!15] (.9,{y:.3f}) -- (7.0,{y:.3f});',
                          f'\\node[anchor=east] at (.8,{y:.3f}) {{{tick}}};'])
        lines.append(f'\\node[rotate=90] at (.12,2) {{{label} (\\%)}};')
        for h, (method, head_label) in enumerate(HEADS):
            middle = 1.95 + h*2.0
            for e, (encoder, color) in enumerate(zip(ENCODERS, COLORS)):
                row = cells[encoder, 'matched4', center, method]
                mean, sd = 100*float(row[metric]), 100*float(row[metric+'_seed_sd'])
                x = middle + (e-1)*.48
                y, low, high = .04*mean, .04*(mean-sd), .04*(mean+sd)
                lines.append(f'\\fill[{color}] ({x-.20:.3f},0) rectangle ({x+.20:.3f},{y:.5f});')
                lines.append(f'\\draw[black!80] ({x:.3f},{low:.5f}) -- ({x:.3f},{high:.5f});')
                lines.append(f'\\draw[black!80] ({x-.09:.3f},{low:.5f}) -- ({x+.09:.3f},{low:.5f});')
                lines.append(f'\\draw[black!80] ({x-.09:.3f},{high:.5f}) -- ({x+.09:.3f},{high:.5f});')
            lines.append(f'\\node at ({middle:.3f},-.30) {{{head_label}}};')
        lines.extend([r'\draw (.9,0) -- (7,0);', r'\draw (.9,0) -- (.9,4);', r'\end{scope}'])
    lines.extend([r'\end{tikzpicture}', r'\caption{' + endpoint +
                  ' on the identical custom CAMELYON subset under four-candidate source-only selection. '
                  'Bars show ten-seed means; whiskers show fitting-seed standard deviation, not patient- or hospital-population uncertainty. '
                  'CBMP, linear logistic and trained cosine are prespecified views; all nine heads appear in the accompanying tables.}',
                  r'\label{' + figure_label + '}', r'\end{figure*}'])
    return '\n'.join(lines)+'\n'


def export(plan_dir, output, reference_report=None):
    plan_dir, output = Path(plan_dir).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError('Choose a new results-export directory')
    provenance = json.loads((plan_dir / 'comparison/provenance.json').read_text())
    if not provenance['same_target_support_seeds_grids_verified']:
        raise ValueError('The paired encoder comparison must be verified first')
    reference_report = Path(reference_report or ROOT / 'outputs/revision_2026-11-07/review_report').resolve()
    if sha256(reference_report / 'provenance.json') != provenance['reference_report_sha256']:
        raise ValueError('Reference intervals must belong to the verified encoder comparison')
    if sha256(plan_dir / 'report/provenance.json') != provenance['foundation_report_sha256']:
        raise ValueError('Foundation intervals must belong to the verified encoder comparison')
    interval_paths = (reference_report / 'paired_ba_intervals.csv', plan_dir / 'report/paired_ba_intervals.csv')
    intervals = {(r['task'], r['protocol'], r['center'], r['comparison']): r
                 for path in interval_paths for r in read_csv(path)}
    patch = read_csv(plan_dir / 'comparison/image_metrics_seed_means.csv')
    patient = read_csv(plan_dir / 'comparison/patient_macro_seed_means.csv')
    if any(int(r['fitting_seeds']) != 10 for r in patch+patient):
        raise ValueError('All displayed cells require ten fitted seeds')
    output.mkdir(parents=True)
    (output / 'patch_balanced_accuracy.tex').write_text(figure(patch, 'balanced_accuracy', 'Patch balanced accuracy',
        'Patch-level balanced accuracy', 'fig:pathology_encoders_patch'), encoding='utf-8')
    (output / 'patient_macro_balanced_accuracy.tex').write_text(figure(patient, 'balanced_accuracy', 'Patient-macro patch BA',
        'Equal-patient averages of patch balanced accuracy; these are not patient-diagnosis scores',
        'fig:pathology_encoders_patient_macro'), encoding='utf-8')
    cells = {(r['encoder'], r['protocol'], r['center'], r['method']): r for r in patch}
    paragraphs = ['# Pretrained pathology encoders: same-cohort comparison', '',
        'All numbers below use the four-candidate source-only selection protocol and ten fitting seeds. '
        'All three encoders use identical target identities, labels and patient groups. Model-specific preprocessing is preserved.', '',
        'This is the 17,200-patch custom CAMELYON17-WILDS subset, with 26 source patients, 9 H2 target patients and 8 H1 target patients. '
        'It is not a complete official benchmark or an independent cohort.', '']
    summary = []
    for center, title in (('2', 'H2: primary target'), ('1', 'H1: secondary target')):
        paragraphs.extend([f'## {title}', '', '| Encoder | Head | Patch BA (%) | AUROC (%) |', '|---|---|---:|---:|'])
        for encoder in ENCODERS:
            for method, head in HEADS:
                row = cells[encoder, 'matched4', center, method]
                ba, auc = 100*float(row['balanced_accuracy']), 100*float(row['auroc'])
                paragraphs.append(f'| {encoder} | {head} | {ba:.2f} | {auc:.2f} |')
            cbmp, linear = (cells[encoder, 'matched4', center, method] for method in ('cbmp', 'linear_logistic'))
            summary.append({'encoder': encoder, 'center': center,
                            'cbmp_minus_linear_ba_points': 100*(float(cbmp['balanced_accuracy'])-float(linear['balanced_accuracy']))})
        paragraphs.append('')
    tasks = {'ResNet18': 'camelyon_resnet18', 'UNI': 'camelyon_custom_uni', 'CONCH': 'camelyon_custom_conch'}
    paragraphs.extend(['## Paired CBMP-minus-linear differences', '',
        'Patch balanced-accuracy differences in percentage points; 95% percentile patient-bootstrap intervals '
        'use 2,000 paired draws with fitted heads held fixed. These exploratory comparisons are not corrected for multiplicity.', '',
        '| Target | Encoder | Difference (points) | 95% interval |', '|---|---|---:|---:|'])
    paired = []
    for center in ('2', '1'):
        for encoder in ENCODERS:
            row = intervals[tasks[encoder], 'matched4', center, 'cbmp_minus_linear_logistic']
            difference, low, high = (100*float(row[k]) for k in ('difference', 'ci_low', 'ci_high'))
            expected = 100*(float(cells[encoder, 'matched4', center, 'cbmp']['balanced_accuracy'])
                            - float(cells[encoder, 'matched4', center, 'linear_logistic']['balanced_accuracy']))
            if abs(difference-expected) > 1e-9:
                raise ValueError('Paired contrast and displayed point estimates differ')
            paragraphs.append(f'| H{center} | {encoder} | {difference:+.2f} | [{low:+.2f}, {high:+.2f}] |')
            paired.append({'encoder': encoder, 'center': center, 'difference_points': difference,
                           'ci_low_points': low, 'ci_high_points': high, 'valid_draws': int(row['valid_draws'])})
    paragraphs.extend(['', '## Reading the comparison', '',
        'The encoders are compared on the same patients to reduce cohort-composition differences. '
        'The four-candidate protocol controls candidate counts for tunable families; centroid controls remain fixed. '
        'It does not equalize coverage, optimization cost or representation capacity. '
        'The historical-grid and fixed K/beta tables retain the complementary controls.', ''])
    for encoder in ('UNI', 'CONCH'):
        gains = [100*(float(cells[encoder, 'matched4', '2', method]['balanced_accuracy'])
                      - float(cells['ResNet18', 'matched4', '2', method]['balanced_accuracy']))
                 for method, _ in HEADS]
        paragraphs.append(f'At primary H2, {encoder} changes patch BA relative to ResNet18 by '
                          f'{gains[0]:+.2f} points for CBMP, {gains[1]:+.2f} for linear logistic and '
                          f'{gains[2]:+.2f} for trained cosine. These are descriptive encoder-configuration differences.')
        paragraphs.append('')
    paragraphs.extend(['## Interpretation limits', '',
        'These are descriptive point estimates. Seed SD measures fitting variation on the same patients. '
        'Conditional paired patient-bootstrap intervals are in `report/paired_ba_intervals.csv`; '
        'patient-macro patch intervals are in `aggregate_uncertainty/paired_aggregate_intervals.csv`. '
        'Neither interval estimates variation across a population of hospitals.', '',
        'The native TikZ figures show the prespecified CBMP/linear/cosine views. Full CSV and LaTeX tables include all nine heads, '
        'both original and four-candidate grids, and fixed prototype/balancing controls. '
        'Encoder capacity, pretraining and preprocessing change together, so differences do not isolate an architecture effect.', '',
        'Patient-macro patch metrics average within-patient patch classification. No clinical patient/slide ground truth has been inferred. '
        'The experiment does not resolve PolypGen patient linkage or replace external confirmation. '
        'Absence of overlap between provider pretraining data and this cohort has not been independently established.'])
    (output / 'RESULTS.md').write_text('\n'.join(paragraphs)+'\n', encoding='utf-8')
    write_json(output / 'summary.json', {'protocol': 'matched4', 'descriptive_cbmp_minus_linear': summary,
               'comparison_provenance_sha256': sha256(plan_dir / 'comparison/provenance.json'),
               'patch_summary_sha256': sha256(plan_dir / 'comparison/image_metrics_seed_means.csv'),
               'patient_macro_summary_sha256': sha256(plan_dir / 'comparison/patient_macro_seed_means.csv'),
               'paired_patch_ba_intervals': paired,
               'paired_interval_inputs_sha256': {str(path.relative_to(ROOT)): sha256(path) for path in interval_paths},
               'exporter_sha256': sha256(Path(__file__))})
    print('Exported descriptive result tables and two native TikZ figures')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--reference-report', type=Path)
    args = parser.parse_args()
    export(args.plan_dir, args.output_dir, args.reference_report)
