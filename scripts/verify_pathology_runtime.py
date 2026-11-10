"""Check optional provider imports/architectures without pretrained weights.

This is an untrained runtime smoke check. It produces no classification metrics
and must never be presented as a UNI/CONCH experiment.
"""
from __future__ import annotations

import argparse
import gc
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import random
import subprocess
import sys
import tempfile
import time

import numpy as np

from extract_pathology_review_features import load_complete_state
from review_protocol import ROOT, sha256, write_json


def verify(output):
    os.environ.setdefault("HF_HUB_CACHE", str(ROOT / "tmp/pathology_hf_cache/hub"))
    os.environ.setdefault("TRANSFORMERS_CACHE", str(ROOT / "tmp/pathology_hf_cache/transformers"))
    Path(os.environ["HF_HUB_CACHE"]).mkdir(parents=True, exist_ok=True)
    Path(os.environ["TRANSFORMERS_CACHE"]).mkdir(parents=True, exist_ok=True)
    import torch
    import timm
    import torchvision
    from conch.open_clip_custom.coca_model import CoCa
    from conch.open_clip_custom.factory import CFG_DIR
    from huggingface_hub import get_token

    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError("Choose a new runtime-check directory")
    output.mkdir(parents=True)
    check = subprocess.run([sys.executable, "-m", "pip", "check"], capture_output=True, text=True)
    (output / 'pip_check.txt').write_text(check.stdout + check.stderr, encoding='utf-8')
    if check.returncode:
        raise RuntimeError("Provider dependency check failed")
    freeze = subprocess.run([sys.executable, "-m", "pip", "freeze", "--all"], capture_output=True, text=True, check=True)
    (output / 'installed_packages.txt').write_text(freeze.stdout, encoding='utf-8')
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.manual_seed(17)
    random.seed(17)
    np.random.seed(17)
    torch.use_deterministic_algorithms(True)
    strict = {}
    with tempfile.TemporaryDirectory(dir=output) as temporary:
        checkpoint = Path(temporary) / 'explicit_tiny_untrained_fixture.pt'
        model = torch.nn.Linear(3, 2)
        torch.save({'state_dict': {'module.' + k: v for k, v in model.state_dict().items()}}, checkpoint)
        load_complete_state(model, checkpoint)
        strict['nested_prefixed_state_loads'] = True
        torch.save({'weight': model.weight}, checkpoint)
        try:
            load_complete_state(model, checkpoint)
        except RuntimeError:
            strict['missing_keys_rejected'] = True
        else:
            raise AssertionError('Missing checkpoint parameters were accepted')
        torch.save({'weight': torch.ones((4, 3)), 'bias': torch.zeros(2)}, checkpoint)
        try:
            load_complete_state(model, checkpoint)
        except RuntimeError:
            strict['incorrect_shapes_rejected'] = True
        else:
            raise AssertionError('Incorrect checkpoint shapes were accepted')
    del model
    architecture = {}
    for encoder in ('uni', 'conch'):
        started = time.monotonic()
        if encoder == 'uni':
            model = timm.create_model('vit_large_patch16_224', pretrained=False, img_size=224,
                                      patch_size=16, init_values=1e-5, num_classes=0, dynamic_img_size=True)
            size, dimension = 224, 1024
            encode = lambda image: model(image)
        else:
            config = json.loads((CFG_DIR / 'conch_ViT-B-16.json').read_text())
            config.pop('custom_text', None)
            model = CoCa(**config)
            size, dimension = 448, 512
            encode = lambda image: model.encode_image(image, proj_contrast=False, normalize=False)
        model.eval().requires_grad_(False)
        parameter_count = sum(p.numel() for p in model.parameters())
        with torch.inference_mode():
            image = torch.zeros((1, 3, size, size))
            first = encode(image).float()
            second = encode(image).float()
        assert first.shape == (1, dimension) and torch.isfinite(first).all()
        assert torch.equal(first, second)
        architecture[encoder] = {'parameters': parameter_count, 'input_size': size,
                                  'output_dimension': dimension, 'repeat_bitwise_equal': True,
                                  'pretrained_weights_loaded': False, 'initialization_and_two_forwards_seconds': time.monotonic()-started}
        print(f"Untrained architecture smoke check passed: {encoder}, dimension {dimension}", flush=True)
        del encode, model, image, first, second
        gc.collect()
    conch_distribution = importlib.metadata.distribution('conch')
    direct_url = json.loads(conch_distribution.read_text('direct_url.json') or '{}')
    record = {'scope': 'untrained architectures/imports only; no pretrained feature or task evaluation',
              'python': platform.python_version(), 'platform': platform.platform(),
              'torch': torch.__version__, 'torchvision': torchvision.__version__,
              'timm': timm.__version__, 'conch_source': direct_url,
              'pip_check_passed': True, 'cuda_used': False, 'threads': 4,
              'local_huggingface_token_present': bool(get_token()),
              'all_model_weights_downloaded': False, 'pretrained_experiments_executed': False,
              'strict_loader_checks': strict, 'architecture_smoke_checks': architecture,
              'model_registry_sha256': sha256(ROOT / 'configs/pathology_models.json'),
              'runtime_checker_sha256': sha256(Path(__file__)),
              'extractor_sha256': sha256(ROOT / 'scripts/extract_pathology_review_features.py')}
    write_json(output / 'runtime_verification.json', record)
    print('Verified provider runtime; actual pretrained experiments still require approved checkpoints', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    verify(args.output_dir)
