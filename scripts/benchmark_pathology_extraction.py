"""Select CPU throughput settings using only prespecified source images.

This measures frozen-encoder runtime and numerical consistency, never classifier
accuracy. It does not access target images or choose model hyperparameters.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path
import random
import time

import numpy as np
from PIL import Image

from review_protocol import ROOT, read_csv, sha256, write_json
from fetch_pathology_checkpoints import validate_checkpoint


def benchmark(plan_dir, output, encoders):
    plan_dir, output = Path(plan_dir).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError('Choose a new source-benchmark output directory')
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ.setdefault('HF_HUB_CACHE', str(ROOT / 'tmp/pathology_hf_cache/hub'))
    os.environ.setdefault('TRANSFORMERS_CACHE', str(ROOT / 'tmp/pathology_hf_cache/transformers'))
    import torch
    from extract_pathology_review_features import build_encoder
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(17)
    np.random.seed(17)
    random.seed(17)
    plan = json.loads((plan_dir / 'comparison_plan.json').read_text())
    manifest = plan_dir / 'image_manifest.csv'
    assert sha256(manifest) == plan['portable_manifest_sha256']
    source_rows = [r for r in read_csv(manifest) if r['partition'] == 'source_train'][:32]
    assert len(source_rows) == 32 and all(r['center'] in ('0', '3', '4') for r in source_rows)
    for row in source_rows:
        if sha256(ROOT / row['image_path']) != row['sha256']:
            raise ValueError('Source benchmark image bytes changed')
    output.mkdir(parents=True)
    records, selected = [], {}
    for encoder in encoders:
        checkpoint = validate_checkpoint(ROOT / 'checkpoints/pathology' / encoder / 'pytorch_model.bin', plan['models'][encoder])
        torch.set_num_threads(4)
        model, transform, encode, definition, implementation = build_encoder(encoder, checkpoint)
        images = []
        for row in source_rows:
            with Image.open(ROOT / row['image_path']) as image:
                images.append(transform(image.convert('RGB')))
        images = torch.stack(images)
        baseline = None
        for threads, batch_size in ((4, 4), (4, 16), (8, 4), (8, 16), (8, 32)):
            torch.set_num_threads(threads)
            with torch.inference_mode():
                encode(images[:batch_size])
                started = time.monotonic()
                values = []
                for start in range(0, len(images), batch_size):
                    values.append(torch.nn.functional.normalize(encode(images[start:start+batch_size]).float(), dim=1).cpu())
                elapsed = time.monotonic() - started
                result = torch.cat(values).numpy()
            if baseline is None:
                baseline = result.copy()
            difference = float(np.max(np.abs(result - baseline)))
            if not np.isfinite(result).all() or difference > 2e-5:
                raise ValueError('Resource settings changed source embeddings beyond the recorded numerical tolerance')
            record = {'encoder': encoder, 'threads': threads, 'batch_size': batch_size,
                      'source_images': len(images), 'elapsed_seconds': elapsed,
                      'images_per_second': len(images)/elapsed, 'maximum_feature_difference': difference}
            records.append(record)
            write_json(output / 'progress.json', {'records': records, 'scope': 'source-only throughput; no target metrics'})
            print(json.dumps(record), flush=True)
        winner = max((r for r in records if r['encoder'] == encoder), key=lambda r: r['images_per_second'])
        selected[encoder] = {key: winner[key] for key in ('threads', 'batch_size', 'images_per_second')}
        selected[encoder]['device'] = 'cpu'
        del model, encode, transform, images, values, result, baseline
        gc.collect()
    record = {'scope': 'source-only throughput benchmark before target extraction; not predictive evaluation',
              'source_ids': [r['sample_id'] for r in source_rows], 'records': records, 'selected': selected,
              'comparison_plan_sha256': sha256(plan_dir / 'comparison_plan.json'),
              'benchmark_script_sha256': sha256(Path(__file__)), 'pretrained_checkpoints_strictly_loaded': True,
              'extractor_sha256': sha256(ROOT / 'scripts/extract_pathology_review_features.py'),
              'selection_rule': 'highest source-image throughput within numerical tolerance 2e-5',
              'torch': torch.__version__, 'device': 'cpu'}
    write_json(output / 'resource_selection.json', record)
    print('Source-only resource selection complete', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--encoders', nargs='+', choices=('uni', 'conch'), default=['uni', 'conch'])
    args = parser.parse_args()
    benchmark(args.plan_dir, args.output_dir, args.encoders)
