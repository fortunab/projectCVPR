"""Regression checks for selection boundaries and review-protocol endpoints."""

import copy
import json
from pathlib import Path
import sys

import joblib
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import review_protocol
from run_review_revision import fit_worker, source_sequence_threshold


def fixture_data():
    rows = []
    for center in ("A", "B", "T"):
        for index in range(4):
            identity = f"{center}:{index}"
            rows.append({"sample_id": identity, "center": center, "label": index % 2,
                         "group_id": f"{center}:group{index // 2}", "sha256": identity})
    values = np.random.default_rng(6).normal(size=(12, 6))
    values /= np.linalg.norm(values, axis=1, keepdims=True)
    splits = [{"heldout_center": "T", "outer_train": list(range(8)), "outer_test": list(range(8, 12)),
               "inner_folds": [{"train": list(range(4)), "validation": list(range(4, 8))},
                               {"train": list(range(4, 8)), "validation": list(range(4))}]}]
    return values, rows, splits


def test_candidate_budget_is_four_distinct_settings_per_tunable_family():
    for name, candidates in review_protocol.grids("matched4").items():
        expected = 1 if name.startswith("single_centroid") else 4
        assert len(candidates) == expected
        assert len({json.dumps(c, sort_keys=True) for c in candidates}) == expected
    assert len(review_protocol.SEEDS) == len(set(review_protocol.SEEDS)) == 10
    assert {17, 29, 43} <= set(review_protocol.SEEDS)


def test_split_validator_rejects_shared_groups_and_duplicate_content():
    _, rows, splits = fixture_data()
    review_protocol.validate_splits(rows, splits)
    for field in ("group_id", "sha256"):
        leaked = copy.deepcopy(rows)
        leaked[8][field] = leaked[0][field]
        with pytest.raises(ValueError, match="leakage"):
            review_protocol.validate_splits(leaked, splits)
    changed = copy.deepcopy(splits)
    changed[0]["inner_folds"][0]["validation"].append(8)
    with pytest.raises(ValueError, match="source-validation"):
        review_protocol.validate_splits(rows, changed)


def test_sequence_threshold_requires_two_source_sequence_classes():
    scores = np.array([[0, .9], [0, .8], [0, 100]])
    result = source_sequence_threshold(scores, [1, 0, 1], ["p", "n", "only_positive"], ["S", "S", "U"])
    assert result["eligible_source_centers"] == ["S"]
    assert .8 < result["threshold"] < .9
    assert result["source_balanced_accuracy"] == 1
    assert source_sequence_threshold(scores[:2], [1, 1], ["a", "b"], ["S", "S"]) is None


def test_changing_target_labels_cannot_change_source_selection(monkeypatch, tmp_path):
    from cbmp_evaluation import fit_head as original_fit

    features, rows, splits = fixture_data()
    proof = {"fixture": "same frozen source identities"}
    small_grid = {
        "linear_logistic": [{"kind": "linear", "C": c} for c in (.1, 1.0)],
        "single_centroid_group_weighted": [{"kind": "prototype", "k": 1, "beta": 0.0}],
    }
    monkeypatch.setattr(review_protocol, "grids", lambda _: small_grid)
    observed_centers = []

    def checked_fit(spec, values, labels, centers, groups, seed):
        observed_centers.extend(centers.tolist())
        assert set(centers) <= {"A", "B"}
        return original_fit(spec, values, labels, centers, groups, seed)

    monkeypatch.setattr("cbmp_evaluation.fit_head", checked_fit)
    choices = []
    source_scores = []
    for run_index in range(2):
        local = copy.deepcopy(rows)
        if run_index:
            for row in local[8:]:
                row["label"] = 1 - row["label"]
        monkeypatch.setattr(review_protocol, "load_cached_task", lambda _: (features, local, splits, proof))
        root = tmp_path / str(run_index)
        root.mkdir()
        (root / "run_manifest.json").write_text(json.dumps({"inputs": {"polypgen_fixture": proof}}))
        fit_worker(root, "polypgen_fixture", 17, ["matched4"], 1)
        folder = root / "polypgen_fixture/seed17"
        selected = json.loads((folder / "source_heads_frozen.json").read_text())["choices"]
        choices.append([(c["method"], c["configuration"], c["temperature"], c["sequence_rule"]) for c in selected])
        models = [joblib.load(folder / c["model"])["model"] for c in selected]
        from cbmp_evaluation import head_scores
        source_scores.append([head_scores(model, features[:8]) for model in models])
        assert json.loads((folder / "COMPLETE.json").read_text())["models"] == 2
    assert observed_centers and "T" not in observed_centers
    assert choices[0] == choices[1]
    for before, after in zip(*source_scores):
        np.testing.assert_array_equal(before, after)


def test_sequence_endpoint_preserves_single_class_limitation():
    from report_review_revision import metrics, sequence_rows

    rows = [{"group_id": "a", "label": "1", "margin": ".5", "source_sequence_threshold": "0"},
            {"group_id": "a", "label": "0", "margin": "-1", "source_sequence_threshold": "0"},
            {"group_id": "b", "label": "1", "margin": ".2", "source_sequence_threshold": "0"}]
    entities = sequence_rows(rows)
    result = metrics([r["label"] for r in entities], [r["prediction"] for r in entities],
                     [r["score"] for r in entities])
    assert result["n_observations"] == 2
    assert result["n_positive"] == 2
    assert result["auroc"] is None and result["balanced_accuracy"] is None


def test_group_bootstrap_matches_explicit_observation_repetition():
    from report_review_revision import pair_counts, ranking_samples, group_confusions, confusion_ba
    from sklearn.metrics import roc_auc_score, balanced_accuracy_score

    rows = [{"group_id": g, "label": y, "prediction": p, "probability_class_1": score}
            for g, y, p, score in [("a", 1, 1, .7), ("a", 0, 1, .5), ("b", 1, 0, .5), ("b", 0, 0, .3)]]
    groups, pairs, wins = pair_counts(rows)
    multiplicities = np.array([[2, 1], [0, 1]])
    ranks = ranking_samples(pairs, wins[None], multiplicities)
    _, confusion = group_confusions(rows)
    bas = confusion_ba(confusion, multiplicities)
    for draw, counts in enumerate(multiplicities):
        repeated = [r for group, count in zip(groups, counts) for _ in range(count) for r in rows if r["group_id"] == group]
        labels = [r["label"] for r in repeated]
        scores = [r["probability_class_1"] for r in repeated]
        np.testing.assert_allclose(ranks[draw, 0, 0], roc_auc_score(labels, scores))
        np.testing.assert_allclose(bas[draw], balanced_accuracy_score(labels, [r["prediction"] for r in repeated]))
    assert np.isnan(ranks[1, 0, 2])


def test_prototype_matching_is_invariant_to_cluster_indices():
    from audit_review_stability import matched_cosines
    vectors = np.eye(3)
    np.testing.assert_array_equal(matched_cosines(vectors, vectors[[2, 0, 1]]), np.ones(3))
    assert matched_cosines(vectors, vectors[:2]) is None


def test_prediction_archive_is_exact_detects_corruption_and_preserves_existing_partial(tmp_path):
    from materialize_review_predictions import pack_file, restore_file
    from review_protocol import write_csv

    rows = [{"task": "polypgen_fixture", "seed": "17", "protocol": "historical", "center": "C5",
             "sample_id": f"sample:{i}", "group_id": "séquence", "label": str(i), "dataset_index": "",
             "method": "cbmp", "prediction": str(i), "probability_class_1": "0.12345678901234567",
             "margin": "-0.00000000000000003", "source_sequence_threshold": "0.0"} for i in (0, 1)]
    source, index, objects = tmp_path / "source.csv", tmp_path / "index.json", tmp_path / "objects"
    write_csv(source, rows)
    pack_file(source, index, objects)
    restored = tmp_path / "restored.csv"
    assert restore_file(index, objects, restored) == 2
    assert restored.read_bytes() == source.read_bytes()
    record = json.loads(index.read_text())
    corrupted = objects / (record["blocks"][0]["prediction_object"] + ".npz")
    corrupted.write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="bytes changed"):
        restore_file(index, objects, tmp_path / "corrupt.csv")
    assert not (tmp_path / "corrupt.csv.partial").exists()
    partial = tmp_path / "protected.csv.partial"
    partial.write_bytes(b"existing output must survive")
    with pytest.raises(FileExistsError):
        restore_file(index, objects, tmp_path / "protected.csv")
    assert partial.read_bytes() == b"existing output must survive"
