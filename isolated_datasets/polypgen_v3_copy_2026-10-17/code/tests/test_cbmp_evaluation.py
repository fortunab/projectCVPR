"""Protocol checks on tiny generated feature fixtures, never experimental data.

Run from polypgen/: python -m unittest discover -s tests -p test_cbmp_evaluation.py -v
"""
from __future__ import annotations

import copy
import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

import joblib
import numpy as np

import cbmp_evaluation as evaluation
from cbmp import center_group_weights


class _SelectionFixtureHead:
    """Return known correct/incorrect scores to isolate candidate selection."""

    def __init__(self, correct):
        self.correct = correct

    def scores(self, features):
        predicted = features[:, 0].astype(int)
        if not self.correct:
            predicted = 1 - predicted
        return np.column_stack([predicted == 0, predicted == 1]).astype(float)


class EvaluationProtocolTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.rows, features = [], []
        for center, shift in (("A", 5), ("B", -5), ("C", 10)):
            angles = np.deg2rad(np.array([-40, -10, 10, 40, 140, 170, 190, 220]) + shift)
            for index, angle in enumerate(angles):
                sample_id = f"{center}_{index}"
                features.append([np.cos(angle), np.sin(angle)])
                self.rows.append({"sample_id": sample_id, "label": int(index >= 4),
                                  "center": center, "group_id": sample_id,
                                  "sha256": f"fixture_hash_{sample_id}"})
        self.features = np.array(features)
        self.splits = [{
            "heldout_center": "C", "status": "ready", "outer_train": list(range(16)),
            "outer_test": list(range(16, 24)), "selection_protocol": "leave_one_source_center_out",
            "inner_folds": [
                {"train": list(range(8, 16)), "validation": list(range(8))},
                {"train": list(range(8)), "validation": list(range(8, 16))},
            ],
        }]
        self.cheap_methods = {
            "cbmp": [{"kind": "prototype", "k": 1, "beta": 0.0},
                     {"kind": "prototype", "k": 2, "beta": 1.0}],
            "multi_prototype_unbalanced": [{"kind": "prototype", "k": 1, "beta": 0.0},
                                            {"kind": "prototype", "k": 2, "beta": 0.0}],
        }

    def run_evaluation(self, path, *, features=None, rows=None, splits=None):
        with redirect_stdout(io.StringIO()):
            return evaluation.evaluate_loco(
                self.features if features is None else features,
                self.rows if rows is None else rows,
                self.splits if splits is None else splits,
                path, seed=17, methods=self.cheap_methods,
            )

    def test_outer_target_changes_cannot_change_selection_temperature_or_fitted_model(self):
        first_path, changed_path = self.root / "original", self.root / "changed_target"
        first = self.run_evaluation(first_path)
        changed_features = self.features.copy()
        changed_features[16:] = np.roll(changed_features[16:], 1, axis=1)
        changed_rows = copy.deepcopy(self.rows)
        for row in changed_rows[16:]:
            row["label"] = 1 - row["label"]
        changed = self.run_evaluation(changed_path, features=changed_features, rows=changed_rows)
        for original, modified in zip(first, changed):
            self.assertEqual(original["configuration"], modified["configuration"])
            self.assertEqual(original["temperature"], modified["temperature"])
            method = original["method"]
            original_artifact = joblib.load(first_path / "models" / "C" / f"{method}.joblib")
            modified_artifact = joblib.load(changed_path / "models" / "C" / f"{method}.joblib")
            self.assertEqual(original_artifact["configuration"], modified_artifact["configuration"])
            self.assertEqual(original_artifact["temperature"], modified_artifact["temperature"])
            np.testing.assert_array_equal(original_artifact["model"].prototypes,
                                          modified_artifact["model"].prototypes)
            np.testing.assert_array_equal(original_artifact["model"].valid,
                                          modified_artifact["model"].valid)
        self.assertEqual(json.loads((first_path / "selection_search.json").read_text()),
                         json.loads((changed_path / "selection_search.json").read_text()))

    def test_unbalanced_multiple_prototypes_have_their_own_full_k_grid(self):
        methods = evaluation.candidate_methods()
        self.assertEqual([spec["k"] for spec in methods["multi_prototype_unbalanced"]], [1, 2, 4, 8])
        self.assertEqual({spec["beta"] for spec in methods["multi_prototype_unbalanced"]}, {0.0})
        self.assertEqual({(spec["k"], spec["beta"]) for spec in methods["cbmp"]},
                         {(k, beta) for k in (1, 2, 4, 8) for beta in (0.0, 0.5, 1.0)})
        labels = np.array([0, 1, 0, 1, 0, 1, 0, 1])
        features = np.column_stack([labels, np.ones(len(labels))])
        centers = np.array(["A"] * 4 + ["B"] * 4)
        groups = np.arange(8).astype(str)
        folds = [(np.arange(4), np.arange(4, 8)), (np.arange(4, 8), np.arange(4))]

        def fit_fixture(spec, *_args):
            # The best balanced setting uses K=1; the unbalanced optimum uses K=2.
            return _SelectionFixtureHead((spec["k"], spec["beta"]) in {(1, 0.5), (2, 0.0)})

        with patch.object(evaluation, "fit_head", side_effect=fit_fixture):
            cbmp_spec, _, _ = evaluation.select_head(methods["cbmp"], features, labels,
                                                     centers, groups, folds, seed=17)
            unbalanced_spec, _, _ = evaluation.select_head(methods["multi_prototype_unbalanced"],
                                                           features, labels, centers, groups, folds, seed=17)
        self.assertEqual((cbmp_spec["k"], cbmp_spec["beta"]), (1, 0.5))
        self.assertEqual((unbalanced_spec["k"], unbalanced_spec["beta"]), (2, 0.0))

    def test_weighted_linear_passes_explicit_weights_without_second_class_balance(self):
        labels = np.array([0, 0, 0, 1, 1, 1])
        features = self.features[:6]
        centers = np.array(["A", "A", "B", "A", "A", "B"])
        groups = np.array(["a0", "a0", "b0", "a1", "a1", "b1"])
        expected = center_group_weights(labels, centers, groups, beta=1.0)
        expected *= len(expected) / expected.sum()
        with patch.object(evaluation, "LogisticRegression") as constructor:
            model = constructor.return_value
            model.n_iter_, model.max_iter = np.array([2]), 3000
            result = evaluation.fit_head({"kind": "linear", "C": 0.1, "beta": 1.0},
                                         features, labels, centers, groups, seed=17)
        self.assertIs(result, model)
        self.assertIsNone(constructor.call_args.kwargs["class_weight"])
        np.testing.assert_allclose(model.fit.call_args.kwargs["sample_weight"], expected)

    def test_unweighted_linear_retains_class_balance(self):
        labels = np.array([row["label"] for row in self.rows[:8]])
        with patch.object(evaluation, "LogisticRegression") as constructor:
            model = constructor.return_value
            model.n_iter_, model.max_iter = np.array([2]), 3000
            evaluation.fit_head({"kind": "linear", "C": 0.1}, self.features[:8], labels,
                                np.array(["A"] * 8), np.arange(8).astype(str), seed=17)
        self.assertEqual(constructor.call_args.kwargs["class_weight"], "balanced")
        self.assertIsNone(model.fit.call_args.kwargs["sample_weight"])

    def test_positive_temperature_preserves_argmax_and_normalized_probabilities(self):
        scores = np.array([[0.2, -0.1], [-9., 4.], [1000., 1000.], [-1000., -1001.]])
        for tau in (0.02, 0.5, 1., 30.):
            with self.subTest(temperature=tau):
                probabilities = evaluation.probabilities(scores, tau)
                np.testing.assert_array_equal(probabilities.argmax(axis=1), scores.argmax(axis=1))
                np.testing.assert_allclose(probabilities.sum(axis=1), 1.)
                self.assertTrue(np.all(np.isfinite(probabilities)))

    def test_partition_guard_rejects_index_leakage_before_fitting(self):
        cases = {}
        outer_overlap = copy.deepcopy(self.splits)
        outer_overlap[0]["outer_test"].append(0)
        cases["outer_train_test_overlap"] = outer_overlap
        inner_train_target = copy.deepcopy(self.splits)
        inner_train_target[0]["inner_folds"][0]["train"].append(16)
        cases["target_in_inner_training"] = inner_train_target
        inner_validation_target = copy.deepcopy(self.splits)
        inner_validation_target[0]["inner_folds"][0]["validation"].append(16)
        cases["target_in_inner_validation"] = inner_validation_target
        inner_overlap = copy.deepcopy(self.splits)
        inner_overlap[0]["inner_folds"][0]["validation"].append(8)
        cases["inner_train_validation_overlap"] = inner_overlap
        for name, splits in cases.items():
            with self.subTest(boundary=name), patch.object(evaluation, "fit_head") as fit:
                with self.assertRaisesRegex(ValueError, "partition boundaries"):
                    self.run_evaluation(self.root / name, splits=splits)
                fit.assert_not_called()


if __name__ == "__main__":
    unittest.main()
