"""Fetch pinned, gated original UNI/CONCH weights after local user login.

No credentials are accepted as command arguments or written to the record.
Provider access/terms must be completed by the user on Hugging Face first.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from review_protocol import ROOT, sha256, write_json


def validate_checkpoint(path, model):
    path = Path(path)
    if not path.is_file() or path.stat().st_size != model["bytes"] or sha256(path) != model["sha256"]:
        raise ValueError(f"Checkpoint size/SHA256 does not match pinned {model['repository']} weights")
    return path


def fetch(registry, destination, encoders, downloader=None, token_reader=None):
    registry = Path(registry).resolve()
    models = json.loads(registry.read_text(encoding="utf-8"))["models"]
    if not encoders or len(encoders) != len(set(encoders)) or not set(encoders) <= models.keys():
        raise ValueError("Select distinct registered encoders")
    if downloader is None or token_reader is None:
        from huggingface_hub import get_token, hf_hub_download
        downloader = downloader or hf_hub_download
        token_reader = token_reader or get_token
    destination = Path(destination).resolve()
    verified = {}
    for name in encoders:
        model = models[name]
        folder = destination / name
        checkpoint = folder / model["filename"]
        if checkpoint.exists():
            # Never overwrite an unverified or differently versioned local file.
            validate_checkpoint(checkpoint, model)
        else:
            token = token_reader()
            if not token:
                raise RuntimeError(
                    f"Local Hugging Face login is missing. Request access at "
                    f"https://huggingface.co/{model['repository']} and log in locally; "
                    "do not paste tokens in the chat or pass them as command arguments."
                )
            folder.mkdir(parents=True, exist_ok=True)
            try:
                actual = downloader(repo_id=model["repository"], filename=model["filename"],
                                    revision=model["revision"], local_dir=str(folder), token=token)
            except Exception as error:
                # Suppress request internals; report the type/status without credentials.
                status = getattr(getattr(error, "response", None), "status_code", None)
                raise RuntimeError(
                    f"Cannot download {model['repository']}: {type(error).__name__}, HTTP {status}. "
                    "Confirm local login and provider access before retrying."
                ) from None
            if Path(actual).resolve() != checkpoint:
                raise ValueError("Provider returned a checkpoint outside the requested local location")
            validate_checkpoint(checkpoint, model)
        verified[name] = {"path": str(checkpoint), "repository": model["repository"],
                          "revision": model["revision"], "bytes": model["bytes"], "sha256": model["sha256"]}
        print(f"Verified {name}: {model['revision']}", flush=True)
    record = {"registry_sha256": sha256(registry), "checkpoints": verified,
              "credentials_recorded": False, "weights_redistributable_in_project": False}
    write_json(destination / "verified_checkpoints.json", record)
    return record


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, default=ROOT / "configs/pathology_models.json")
    parser.add_argument("--dest", type=Path, default=ROOT / "checkpoints/pathology")
    parser.add_argument("--encoders", nargs="+", choices=("uni", "conch"), default=["uni", "conch"])
    args = parser.parse_args()
    fetch(args.registry, args.dest, args.encoders)
