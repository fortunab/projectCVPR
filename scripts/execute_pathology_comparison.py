"""Execute the prepared paired study with audited source-only runtime settings.

Original preparation files are preserved. Exact executed arguments, source
fingerprints, environments, stage logs and receipts are saved separately.
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
from pathlib import Path
import subprocess
import sys

from review_protocol import ROOT, sha256, write_json
from fetch_pathology_checkpoints import validate_checkpoint
from external_review_data import load_external_task


def replace_option(arguments, option, value):
    result = list(arguments)
    result[result.index(option) + 1] = str(value)
    return result


def execute(plan_dir, resources, provider, reviewer):
    plan_dir, resources, provider, reviewer = map(lambda p: Path(p).resolve(), (plan_dir, resources, provider, reviewer))
    plan_path = plan_dir / 'comparison_plan.json'
    plan = json.loads(plan_path.read_text())
    commands = json.loads((plan_dir / 'commands.json').read_text())
    selection = json.loads(resources.read_text())
    if selection['comparison_plan_sha256'] != sha256(plan_path) or set(selection['selected']) != {'uni', 'conch'}:
        raise ValueError('Resource selection belongs to another cohort/plan')
    if selection.get('extractor_sha256') and selection['extractor_sha256'] != sha256(ROOT / 'scripts/extract_pathology_review_features.py'):
        raise ValueError('Source numerical validation and extractor revisions differ')
    changed = [name for name, digest in plan['code_sha256'].items() if sha256(ROOT / 'scripts' / name) != digest]
    if set(changed) - {'extract_pathology_review_features.py'}:
        raise ValueError(f'Unexpected changes to the frozen preparation sources: {changed}')
    for encoder in ('uni', 'conch'):
        validate_checkpoint(ROOT / 'checkpoints/pathology' / encoder / plan['models'][encoder]['filename'], plan['models'][encoder])
        for option in ('device', 'threads', 'batch_size'):
            commands['extract'][encoder] = replace_option(commands['extract'][encoder], '--'+option.replace('_', '-'), selection['selected'][encoder][option])
    code_names = list(plan['code_sha256']) + ['execute_pathology_comparison.py']
    executed = {'original_comparison_plan_sha256': sha256(plan_path), 'resource_selection_sha256': sha256(resources),
                'provider_python': str(provider), 'review_python': str(reviewer), 'commands': commands,
                'code_sha256': {name: sha256(ROOT / 'scripts' / name) for name in code_names},
                'amendments_before_target_extraction': [
                    'CONCH permits absent unused text_decoder keys; all required visual and supplied shapes remain validated',
                    'Compute device/batch/thread settings selected using source-only throughput and numerical agreement'],
                'seeds_grids_cohort_unchanged': True}
    executed_path = plan_dir / 'executed_commands.json'
    if executed_path.exists():
        if json.loads(executed_path.read_text()) != executed:
            raise ValueError('Resume requires identical executed sources, inputs and resources')
    else:
        write_json(executed_path, executed)
    logs = plan_dir / 'logs'
    logs.mkdir(exist_ok=True)
    for label, interpreter in (('provider', provider), ('review', reviewer)):
        check = subprocess.run([str(interpreter), '-m', 'pip', 'check'], capture_output=True, text=True)
        (logs / (label+'_pip_check.txt')).write_text(check.stdout+check.stderr, encoding='utf-8')
        if check.returncode:
            raise RuntimeError(f'{label} environment dependency check failed')
        freeze = subprocess.run([str(interpreter), '-m', 'pip', 'freeze', '--all'], capture_output=True, text=True, check=True)
        (logs / (label+'_installed_packages.txt')).write_text(freeze.stdout, encoding='utf-8')
    running = plan_dir / 'EXECUTION_RUNNING'
    with running.open('x', encoding='utf-8') as stream:
        stream.write(str(os.getpid()))
    def stage(label, interpreter, arguments):
        if sha256(plan_dir / 'image_manifest.csv') != plan['portable_manifest_sha256']:
            raise ValueError('The frozen cohort manifest changed during execution')
        for name, digest in executed['code_sha256'].items():
            if sha256(ROOT / 'scripts' / name) != digest:
                raise ValueError(f'Source changed during execution: {name}')
        started = datetime.datetime.now(datetime.timezone.utc).isoformat()
        print(f'START {label}', flush=True)
        with (logs / (label+'.log')).open('a', encoding='utf-8') as logfile:
            process = subprocess.Popen([str(interpreter), '-u', '-B', *arguments], cwd=ROOT,
                                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                       text=True, encoding='utf-8', errors='replace')
            for line in process.stdout:
                logfile.write(line)
                logfile.flush()
                print(f'[{label}] {line}', end='', flush=True)
            code = process.wait()
        if code:
            raise RuntimeError(f'{label} failed with exit {code}; inspect its stage log')
        write_json(logs / (label+'_complete.json'), {'started_utc': started,
                   'finished_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
                   'executed_commands_sha256': sha256(executed_path), 'exit_code': code})
    try:
        for encoder in ('uni', 'conch'):
            task = plan_dir / ('features_'+encoder) / 'task.json'
            if not task.exists():
                stage('extract_'+encoder, provider, commands['extract'][encoder])
            features, rows, splits, metadata = load_external_task(task)
            cache = metadata['feature_metadata']
            if (metadata['task'] != 'camelyon_custom_'+encoder
                    or metadata['manifest_sha256'] != plan['portable_manifest_sha256']
                    or cache['checkpoint_sha256'] != plan['models'][encoder]['sha256']
                    or cache['model_revision'] != plan['models'][encoder]['revision']
                    or features.shape != (plan['rows'], plan['models'][encoder]['embedding_dimension'])):
                raise ValueError('Extracted cache differs from the registered encoder/cohort')
            for option in ('device', 'threads', 'batch_size'):
                if cache[option] != selection['selected'][encoder][option]:
                    raise ValueError('Cache uses different runtime settings')
            del features, rows, splits
        stage('fitting_plan', reviewer, commands['fit']+['--prepare-only', '--resume'])
        stage('fitting', reviewer, commands['fit']+['--resume'])
        for label, completion in (('report', plan_dir / 'report/provenance.json'),
                                  ('aggregate', plan_dir / 'aggregate_uncertainty/provenance.json'),
                                  ('compare', plan_dir / 'comparison/provenance.json')):
            if not completion.exists():
                stage(label, reviewer, commands[label])
        write_json(plan_dir / 'EXECUTION_COMPLETE.json', {
            'executed_commands_sha256': sha256(executed_path), 'completed_task_seed_jobs': 20,
            'expected_target_head_seed_evaluations': 880, 'scope': 'custom same-cohort comparison; not official WILDS'})
        print('Paired pretrained feature extraction, fitting and comparison complete', flush=True)
    finally:
        running.unlink(missing_ok=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan-dir', type=Path, required=True)
    parser.add_argument('--resources', type=Path, required=True)
    parser.add_argument('--provider-python', type=Path, required=True)
    parser.add_argument('--review-python', type=Path, default=Path(sys.executable))
    args = parser.parse_args()
    execute(args.plan_dir, args.resources, args.provider_python, args.review_python)
