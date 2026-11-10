"""Endpoint-presence regression checks for histology-only analysis."""
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from audit_aggregate_uncertainty import audit
from review_protocol import grids, read_csv, write_csv, write_json


def test_histology_only_report_needs_no_sequence_files(tmp_path):
    report = tmp_path / 'report'
    records, summaries = [], []
    for method in grids('matched4'):
        baseline = .8 if method == 'cbmp' else .7
        summaries.append({'task': 'camelyon_custom_uni', 'protocol': 'matched4', 'center': '2',
                          'method': method, 'balanced_accuracy': baseline, 'auroc': baseline})
        for seed in range(10):
            for group in ('patient:a', 'patient:b'):
                records.append({'task': 'camelyon_custom_uni', 'protocol': 'matched4', 'center': '2',
                                'method': method, 'seed': seed, 'group_id': group,
                                'balanced_accuracy': baseline, 'auroc': baseline})
    write_csv(report / 'group_metrics_by_seed.csv', records)
    write_csv(report / 'patient_macro_seed_means.csv', summaries)
    write_json(report / 'provenance.json', {'scope': 'synthetic histology-only regression fixture'})
    audit(report, tmp_path / 'audit', attempts=50)
    results = read_csv(tmp_path / 'audit/paired_aggregate_intervals.csv')
    assert len(results) == 32
    for row in results:
        assert float(row['difference']) == pytest.approx(.1)
        assert float(row['ci_low']) == pytest.approx(.1)
        assert float(row['ci_high']) == pytest.approx(.1)
        assert int(row['valid_draws']) == 50


def test_missing_sequence_files_are_rejected_when_polypgen_is_present(tmp_path):
    write_csv(tmp_path / 'report/group_metrics_by_seed.csv', [{'task': 'polypgen_resnet18'}])
    with pytest.raises(ValueError, match='PolypGen report requires'):
        audit(tmp_path / 'report', tmp_path / 'audit')


def test_missing_patient_summary_is_rejected_for_histology(tmp_path):
    write_csv(tmp_path / 'report/group_metrics_by_seed.csv', [{'task': 'camelyon_custom_conch'}])
    with pytest.raises(ValueError, match='CAMELYON report requires'):
        audit(tmp_path / 'report', tmp_path / 'audit')
