"""Protocol checks that need neither WILDS images nor gated model weights."""

import copy
import json
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from external_review_data import load_external_task, official_splits
from review_protocol import sha256, write_csv, write_json
from run_review_revision import fit_worker


def official_fixture(tmp_path):
    rows = []
    for center in ("0", "3", "4", "1", "2"):
        for j in range(4):
            identity = f"{center}:{j}"
            rows.append({"sample_id": identity, "dataset_index": len(rows), "center": center,
                         "group_id": f"patient:{center}:{j // 2}", "sha256": identity,
                         "label": j % 2, "partition": "val" if center == "1" else "test" if center == "2" else "train"})
    manifest = tmp_path / "manifest.csv"
    write_csv(manifest, rows)
    values = np.random.default_rng(2).normal(size=(len(rows), 6))
    values /= np.linalg.norm(values, axis=1, keepdims=True)
    metadata = {"encoder": "fixture", "checkpoint_sha256": "fixture-weight-id", "model_revision": "fixture-rev",
                "manifest_sha256": sha256(manifest), "preprocessing": "fixture", "embedding_definition": "fixture"}
    cache = tmp_path / "features.npz"
    np.savez(cache, features=values, sample_ids=[r["sample_id"] for r in rows], metadata=json.dumps(metadata))
    audit = tmp_path / "official_audit.json"
    write_json(audit, {"dataset": "camelyon17", "version": "1.0", "split_scheme": "official",
                       "manifest_sha256": sha256(manifest), "official_indices": {
                           p: [r["dataset_index"] for r in rows if r["partition"] == p] for p in ("train", "val", "test")}})
    spec = tmp_path / "task.json"
    write_json(spec, {"task": "camelyon_fixture", "protocol": "official_ood_validation",
                      "manifest": manifest.name, "manifest_sha256": sha256(manifest),
                      "features": cache.name, "features_sha256": sha256(cache),
                      "official_audit": audit.name, "official_audit_sha256": sha256(audit)})
    return rows, spec


def test_official_partition_keeps_validation_out_of_final_training(tmp_path):
    rows, spec = official_fixture(tmp_path)
    _, loaded, splits, proof = load_external_task(spec)
    outer = splits[0]
    assert {loaded[i]["center"] for i in outer["outer_train"]} == {"0", "3", "4"}
    assert {loaded[i]["center"] for i in outer["outer_validation"]} == {"1"}
    assert {loaded[i]["center"] for i in outer["outer_test"]} == {"2"}
    assert set(outer["outer_train"]).isdisjoint(outer["outer_validation"])
    assert outer["inner_folds"][0]["validation"] == outer["outer_validation"]
    assert proof["protocol"] == "official_ood_validation"
    leaked = copy.deepcopy(rows)
    leaked[-1]["group_id"] = leaked[0]["group_id"]
    with pytest.raises(ValueError, match="leakage"):
        official_splits(leaked)


def test_external_cache_identity_and_complete_official_support_are_required(tmp_path):
    rows, spec = official_fixture(tmp_path)
    definition = json.loads(spec.read_text())
    definition["features_sha256"] = "wrong"
    write_json(spec, definition)
    with pytest.raises(ValueError, match="feature cache identity"):
        load_external_task(spec)
    definition["task"] = "camelyon_../../outside"
    write_json(spec, definition)
    with pytest.raises(ValueError, match="task identifier"):
        load_external_task(spec)
    official_fixture(tmp_path)
    manifest = tmp_path / "manifest.csv"
    write_csv(manifest, rows[:-1])
    definition = json.loads(spec.read_text())
    definition["manifest_sha256"] = sha256(manifest)
    audit_path = tmp_path / "official_audit.json"
    audit = json.loads(audit_path.read_text())
    audit["manifest_sha256"] = sha256(manifest)
    write_json(audit_path, audit)
    definition["official_audit_sha256"] = sha256(audit_path)
    write_json(spec, definition)
    with pytest.raises(ValueError, match="capped, removed, or reordered"):
        load_external_task(spec)


def test_official_worker_refits_only_training_hospitals(tmp_path, monkeypatch):
    import review_protocol
    from cbmp_evaluation import fit_head as original_fit
    _, spec = official_fixture(tmp_path)
    _, _, _, proof = load_external_task(spec)
    monkeypatch.setattr(review_protocol, "grids", lambda _: {
        "linear_logistic": [{"kind": "linear", "C": .1}, {"kind": "linear", "C": 1.0}]})
    fits = []

    def checked_fit(configuration, features, labels, centers, groups, seed):
        fits.append(set(centers))
        assert set(centers) == {"0", "3", "4"}
        return original_fit(configuration, features, labels, centers, groups, seed)

    monkeypatch.setattr("cbmp_evaluation.fit_head", checked_fit)
    output = tmp_path / "run"
    output.mkdir()
    write_json(output / "run_manifest.json", {"inputs": {"camelyon_fixture": proof}})
    fit_worker(output, "camelyon_fixture", 17, ["matched4"], 1, str(spec))
    assert len(fits) == 3
    done = output / "camelyon_fixture/seed17/COMPLETE.json"
    assert json.loads(done.read_text())["models"] == 1
