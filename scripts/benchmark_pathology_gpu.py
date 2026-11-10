"""Check pretrained CUDA/CPU agreement and throughput on source images only."""
from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path
import subprocess
import time

import numpy as np
from PIL import Image

from review_protocol import ROOT, read_csv, sha256, write_json
from fetch_pathology_checkpoints import validate_checkpoint


def benchmark(plan_dir, output):
    plan_dir, output = Path(plan_dir).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError('Choose a new GPU-check directory')
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    os.environ.setdefault('HF_HUB_CACHE', str(ROOT / 'tmp/pathology_hf_cache/hub'))
    os.environ.setdefault('TRANSFORMERS_CACHE', str(ROOT / 'tmp/pathology_hf_cache/transformers'))
    import torch
    from extract_pathology_review_features import build_encoder
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is unavailable; retain the independently validated CPU environment')
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(17)
    np.random.seed(17)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
    plan = json.loads((plan_dir / 'comparison_plan.json').read_text())
    rows = [r for r in read_csv(plan_dir / 'image_manifest.csv') if r['partition'] == 'source_train'][:32]
    assert len(rows) == 32 and all(r['center'] in ('0', '3', '4') for r in rows)
    for row in rows:
        if sha256(ROOT / row['image_path']) != row['sha256']:
            raise ValueError('Source benchmark image changed')
    output.mkdir(parents=True)
    records, selected, audits = [], {}, {}
    for encoder in ('uni', 'conch'):
        checkpoint = validate_checkpoint(ROOT / 'checkpoints/pathology' / encoder / 'pytorch_model.bin', plan['models'][encoder])
        model, transform, encode, definition, implementation = build_encoder(encoder, checkpoint)
        audits[encoder] = model._comparison_checkpoint_audit
        images = []
        for row in rows:
            with Image.open(ROOT / row['image_path']) as image:
                images.append(transform(image.convert('RGB')))
        images = torch.stack(images)
        torch.set_num_threads(8)
        with torch.inference_mode():
            reference = torch.cat([torch.nn.functional.normalize(encode(images[i:i+4]).float(), dim=1)
                                   for i in range(0, len(images), 4)]).numpy()
        torch.set_num_threads(4)
        model.to('cuda')
        for batch_size in (1, 4, 8, 16):
            try:
                torch.cuda.reset_peak_memory_stats()
                with torch.inference_mode():
                    first = encode(images[:batch_size].to('cuda')).float()
                    second = encode(images[:batch_size].to('cuda')).float()
                    repeated = torch.equal(first, second)
                    if not repeated:
                        raise RuntimeError('Repeated GPU embeddings were not bitwise identical')
                    del first, second
                    torch.cuda.synchronize()
                    started = time.monotonic()
                    values = []
                    for i in range(0, len(images), batch_size):
                        values.append(torch.nn.functional.normalize(encode(images[i:i+batch_size].to('cuda')).float(), dim=1).cpu())
                    torch.cuda.synchronize()
                    elapsed = time.monotonic() - started
                    array = torch.cat(values).numpy()
                difference = float(np.max(np.abs(array - reference)))
                if not np.isfinite(array).all() or difference > 2e-5:
                    raise RuntimeError('GPU/CPU source embedding agreement exceeded tolerance 2e-5')
                record = {'encoder': encoder, 'device': 'cuda', 'threads': 4, 'batch_size': batch_size,
                          'images_per_second': len(images)/elapsed, 'elapsed_seconds': elapsed,
                          'source_images': len(images), 'maximum_cpu_gpu_feature_difference': difference,
                          'repeat_bitwise_equal': repeated, 'peak_cuda_bytes': torch.cuda.max_memory_allocated(),
                          'status': 'valid'}
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                record = {'encoder': encoder, 'batch_size': batch_size, 'status': 'out_of_memory'}
            records.append(record)
            write_json(output / 'progress.json', {'records': records, 'scope': 'source only; no target metrics'})
            print(json.dumps(record), flush=True)
        winner = max((r for r in records if r['encoder'] == encoder and r['status'] == 'valid'),
                     key=lambda r: r['images_per_second'])
        selected[encoder] = {key: winner[key] for key in ('device', 'threads', 'batch_size', 'images_per_second')}
        del model, encode, transform, images, reference, values, array
        gc.collect()
        torch.cuda.empty_cache()
    driver = subprocess.run(['nvidia-smi', '--query-gpu=name,driver_version,memory.total', '--format=csv,noheader'],
                            capture_output=True, text=True, check=True).stdout.strip()
    record = {'scope': 'pretrained source-only numerical/runtime validation before target extraction',
              'selected': selected, 'records': records, 'checkpoint_validation': audits,
              'source_ids': [r['sample_id'] for r in rows], 'source_images_only': True,
              'selection_rule': 'highest source throughput among deterministic settings with max CPU/GPU difference <= 2e-5',
              'torch': torch.__version__, 'cuda': torch.version.cuda, 'gpu': torch.cuda.get_device_name(),
              'driver_observation': driver, 'precision': 'float32; TF32 and autocast disabled; SDPA math backend',
              'comparison_plan_sha256': sha256(plan_dir / 'comparison_plan.json'),
              'benchmark_script_sha256': sha256(Path(__file__)),
              'extractor_sha256': sha256(ROOT / 'scripts/extract_pathology_review_features.py')}
    write_json(output / 'resource_selection.json', record)
    print('Pretrained CUDA/CPU source agreement and GPU throughput verified', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    benchmark(args.plan_dir, args.output_dir)
