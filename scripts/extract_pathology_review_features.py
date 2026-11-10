"""Extract frozen original-UNI or CONCH visual features from a verified manifest.

Install the selected provider's official runtime separately. Supply weights
obtained under its access terms; this command never authenticates or downloads.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import inspect
import json
import os
from pathlib import Path
import random
import re
import time

import numpy as np
from PIL import Image

from review_protocol import ROOT, read_csv, sha256, write_json


def load_complete_state(model, checkpoint, resize=None, allowed_missing_prefixes=()):
    """Reject partially loaded weights instead of silently accepting random keys."""
    import torch
    if checkpoint.suffix == ".safetensors":
        from safetensors.torch import load_file
        state = load_file(str(checkpoint), device="cpu")
    else:
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    if state and all(key.startswith("module.") for key in state):
        state = {key[7:]: value for key, value in state.items()}
    if resize is not None:
        resize(state, model)
    if allowed_missing_prefixes:
        result = model.load_state_dict(state, strict=False)
        missing_required = [key for key in result.missing_keys
                            if not any(key.startswith(prefix) for prefix in allowed_missing_prefixes)]
        if missing_required or result.unexpected_keys:
            raise RuntimeError(f"Incomplete required checkpoint state: missing={missing_required}, unexpected={result.unexpected_keys}")
        audit = {"allowed_unused_missing_prefixes": list(allowed_missing_prefixes),
                 "unused_missing_keys": len(result.missing_keys), "missing_required_keys": 0,
                 "unexpected_keys": 0, "all_loaded_shapes_validated": True}
    else:
        model.load_state_dict(state, strict=True)
        audit = {"allowed_unused_missing_prefixes": [], "unused_missing_keys": 0,
                 "missing_required_keys": 0, "unexpected_keys": 0, "all_loaded_shapes_validated": True}
    model._comparison_checkpoint_audit = audit


def build_encoder(encoder, checkpoint):
    import torch
    if encoder == "uni":
        import timm
        from torchvision import transforms
        model = timm.create_model("vit_large_patch16_224", pretrained=False, img_size=224,
                                  patch_size=16, init_values=1e-5, num_classes=0, dynamic_img_size=True)
        load_complete_state(model, checkpoint)
        preprocessing = transforms.Compose([
            transforms.Resize(224), transforms.CenterCrop(224), transforms.ToTensor(),
            transforms.Normalize(mean=(.485, .456, .406), std=(.229, .224, .225)),
        ])
        definition = "Original UNI (2024) ViT-L/16 CLS feature, before final cache L2 normalization"
        encode = lambda batch: model(batch)
        implementation = Path(inspect.getfile(type(model)))
    else:
        from conch.open_clip_custom import create_model_from_pretrained
        from conch.open_clip_custom.coca_model import resize_pos_embed
        model, preprocessing = create_model_from_pretrained("conch_ViT-B-16", checkpoint_path=str(checkpoint))
        # The public release omits the unused caption-generation decoder. Every
        # visual/text-encoder key and supplied shape must nevertheless match.
        load_complete_state(model, checkpoint, resize=resize_pos_embed,
                            allowed_missing_prefixes=("text_decoder.",))
        definition = "CONCH encode_image(proj_contrast=False, normalize=False), then cache L2 normalization"
        encode = lambda batch: model.encode_image(batch, proj_contrast=False, normalize=False)
        implementation = Path(inspect.getfile(type(model)))
    model.eval().requires_grad_(False)
    return model, preprocessing, encode, definition, implementation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--encoder", choices=("uni", "conch"), required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--expected-checkpoint-sha256", help="Optional independently pinned checkpoint fingerprint")
    parser.add_argument("--model-revision", required=True, help="Exact provider revision/commit, not main/latest")
    parser.add_argument("--task", required=True, help="Unique identifier starting with camelyon_")
    parser.add_argument("--protocol", choices=("custom_source_only", "official_ood_validation"), required=True)
    parser.add_argument("--official-audit", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    if args.output_dir.exists() or not args.checkpoint.is_file():
        parser.error("Choose a new output directory and supply an existing local checkpoint")
    if args.batch_size < 1 or args.threads < 1 or not re.fullmatch(r"camelyon_[A-Za-z0-9_-]+", args.task):
        parser.error("Positive resources and a camelyon_ task identifier are required")
    if args.model_revision.lower() in ("main", "latest", "master"):
        parser.error("Pin the provider's exact model revision")
    if args.expected_checkpoint_sha256 and (
            not re.fullmatch(r"[0-9a-f]{64}", args.expected_checkpoint_sha256)
            or sha256(args.checkpoint) != args.expected_checkpoint_sha256):
        parser.error("Checkpoint bytes do not match the independently pinned SHA256")
    if args.protocol == "official_ood_validation" and not args.official_audit:
        parser.error("The complete official manifest audit is required")
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ.setdefault("HF_HUB_CACHE", str(ROOT / "tmp/pathology_hf_cache/hub"))
    os.environ.setdefault("TRANSFORMERS_CACHE", str(ROOT / "tmp/pathology_hf_cache/transformers"))
    Path(os.environ["HF_HUB_CACHE"]).mkdir(parents=True, exist_ok=True)
    Path(os.environ["TRANSFORMERS_CACHE"]).mkdir(parents=True, exist_ok=True)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    import torch
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    torch.manual_seed(17)
    random.seed(17)
    np.random.seed(17)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    if args.device == "cuda":
        # Match the deterministic source-only CUDA/CPU numerical validation.
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is unavailable")
    rows = read_csv(args.manifest)
    if not rows or len({r["sample_id"] for r in rows}) != len(rows):
        raise ValueError("A nonempty manifest with unique sample IDs is required")
    model, preprocessing, encode, definition, implementation = build_encoder(args.encoder, args.checkpoint)
    model.to(args.device)
    args.output_dir.mkdir(parents=True)
    cache_path = args.output_dir / "features.npz"
    temporary = args.output_dir / "features.partial.npy"
    feature_array = None
    started = time.monotonic()
    with torch.inference_mode():
        for start in range(0, len(rows), args.batch_size):
            batch = []
            for row in rows[start:start + args.batch_size]:
                path = Path(row.get("path") or row["image_path"])
                path = path if path.is_absolute() else args.image_root / path
                if sha256(path) != row["sha256"]:
                    raise ValueError(f"Image content changed: {row['sample_id']}")
                with Image.open(path) as image:
                    batch.append(preprocessing(image.convert("RGB")))
            tensor = torch.stack(batch).to(args.device)
            values = encode(tensor).float()
            if start == 0 and not torch.equal(values, encode(tensor).float()):
                raise ValueError("Repeated initial-batch features were not bitwise identical")
            if values.ndim != 2 or not torch.isfinite(values).all() or torch.any(values.norm(dim=1) <= 1e-12):
                raise ValueError("Invalid encoder output")
            if values.shape[1] != {"uni": 1024, "conch": 512}[args.encoder]:
                raise ValueError("Output dimension differs from the original encoder's prescribed visual representation")
            values = torch.nn.functional.normalize(values, dim=1).cpu().numpy()
            if feature_array is None:
                feature_array = np.lib.format.open_memmap(temporary, mode="w+", dtype="float32",
                                                          shape=(len(rows), values.shape[1]))
            feature_array[start:start + len(values)] = values
            if start % (args.batch_size * 100) == 0:
                completed = start + len(values)
                elapsed = time.monotonic() - started
                eta = elapsed / completed * (len(rows) - completed)
                print(f"Encoded {completed}/{len(rows)} images; elapsed {elapsed:.1f}s; estimated remaining {eta:.1f}s", flush=True)
    feature_array.flush()
    packages = {}
    for name in ("torch", "torchvision", "timm", "numpy", "Pillow", "conch", "safetensors"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            pass
    metadata = {
        "encoder": args.encoder, "checkpoint_sha256": sha256(args.checkpoint),
        "model_revision": args.model_revision, "manifest_sha256": sha256(args.manifest),
        "preprocessing": str(preprocessing), "embedding_definition": definition,
        "encoder_implementation_sha256": sha256(implementation), "packages": packages,
        "extraction_script_sha256": sha256(Path(__file__)),
        "all_checkpoint_keys_validated": True,
        "checkpoint_validation": model._comparison_checkpoint_audit,
        "expected_checkpoint_sha256": args.expected_checkpoint_sha256,
        "device": args.device, "threads": args.threads, "batch_size": args.batch_size,
        "device_name": torch.cuda.get_device_name() if args.device == "cuda" else "CPU",
        "attention_backend": "SDPA math" if args.device == "cuda" else "provider CPU default",
        "autocast": False, "tf32": False,
        "torch_build": torch.__config__.show(), "cuda_version": torch.version.cuda,
        "deterministic_algorithms": True, "cudnn_benchmark": False,
        "initial_batch_bitwise_repeat_check": True, "stochastic_augmentation": False,
        "raw_image_hashes_reverified": True, "normalized": True,
        "pretraining_overlap": "not independently established; must be reviewed before scientific claims",
    }
    np.savez_compressed(cache_path, features=feature_array,
                        sample_ids=np.asarray([r["sample_id"] for r in rows]), metadata=json.dumps(metadata))
    del feature_array
    temporary.unlink()
    spec = {"task": args.task, "protocol": args.protocol, "manifest": str(args.manifest.resolve()),
            "manifest_sha256": sha256(args.manifest), "features": "features.npz",
            "features_sha256": sha256(cache_path)}
    if args.official_audit:
        spec.update(official_audit=str(args.official_audit.resolve()),
                    official_audit_sha256=sha256(args.official_audit))
    write_json(args.output_dir / "task.json", spec)
    write_json(args.output_dir / "extraction.json", metadata)
    print(f"Prepared {args.output_dir / 'task.json'}", flush=True)


if __name__ == "__main__":
    main()
