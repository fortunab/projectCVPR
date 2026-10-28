"""Checks of weighting semantics for the learned cosine baseline.

The tiny contradictory vectors below are optimization fixtures, not research
data. Run from polypgen/: python -m unittest discover -s tests -p test_cbmp_weighted_heads.py -v
"""
from __future__ import annotations

import unittest

import numpy as np

from cbmp import fit_cosine_head


class WeightedCosineHeadTests(unittest.TestCase):
    def setUp(self):
        self.features = np.array([[1., 0.], [-1., 0.], [1., 0.], [-1., 0.]])
        self.labels = np.array([0, 0, 1, 1])

    def fit(self, sample_weights=None, **kwargs):
        return fit_cosine_head(self.features, self.labels, 2, seed=7,
                               sample_weights=sample_weights, **kwargs)

    def test_weights_change_fit_towards_high_weight_observations(self):
        first_weights = np.array([10., 1., 1., 10.])
        # Reverse the preferred labels at each identical feature vector.
        second_weights = np.array([1., 10., 10., 1.])
        first = self.fit(first_weights)
        second = self.fit(second_weights)
        np.testing.assert_array_equal(first.predict(self.features), [0, 1, 0, 1])
        np.testing.assert_array_equal(second.predict(self.features), [1, 0, 1, 0])
        first_loss = -np.log(first.probabilities(self.features)[np.arange(4), self.labels])
        second_loss = -np.log(second.probabilities(self.features)[np.arange(4), self.labels])
        self.assertLess(np.average(first_loss, weights=first_weights),
                        np.average(second_loss, weights=first_weights))
        self.assertLess(np.average(second_loss, weights=second_weights),
                        np.average(first_loss, weights=second_weights))

    def test_rescaling_weights_does_not_change_model(self):
        weights = np.array([4., 0., 2., 1.])
        baseline = self.fit(weights)
        for scale in (1e-200, 7., 1e200):
            with self.subTest(scale=scale):
                np.testing.assert_allclose(self.fit(weights * scale).weights,
                                           baseline.weights, rtol=1e-12, atol=1e-12)

    def test_default_matches_previous_inverse_class_frequency_fit(self):
        features = np.array([[1., 0.], [0., 1.], [-1., 0.], [0., -1.], [1., 1.]])
        labels = np.array([0, 0, 0, 1, 1])
        inverse_counts = 1. / np.bincount(labels)[labels]
        implicit = fit_cosine_head(features, labels, 2, seed=19)
        explicit = fit_cosine_head(features, labels, 2, seed=19,
                                   sample_weights=inverse_counts)
        np.testing.assert_allclose(implicit.weights, explicit.weights, rtol=1e-12, atol=1e-12)

    def test_provided_weights_do_not_apply_class_balancing_again(self):
        # Duplicating a sample and dividing its weight preserves the objective,
        # even though class counts change. Extra class balancing would break it.
        features = np.array([[1., 0.], [0., 1.], [-1., 0.], [0., -1.]])
        labels = np.array([0, 0, 1, 1])
        weights = np.array([2., 1., 3., 1.])
        reference = fit_cosine_head(features, labels, 2, seed=31, sample_weights=weights)
        duplicated = fit_cosine_head(np.vstack([features, features[0]]),
                                     np.r_[labels, labels[0]], 2, seed=31,
                                     sample_weights=np.array([1., 1., 3., 1., 1.]))
        np.testing.assert_allclose(reference.weights, duplicated.weights, rtol=1e-11, atol=1e-11)

    def test_invalid_weights_are_rejected(self):
        cases = (
            np.ones(3), np.ones((4, 1)), np.array([1., -1., 1., 1.]),
            np.array([1., np.nan, 1., 1.]), np.array([1., np.inf, 1., 1.]),
            np.zeros(4), np.array([1., 1., 0., 0.]),
        )
        for weights in cases:
            with self.subTest(weights=weights), self.assertRaises(ValueError):
                self.fit(weights)

    def test_zero_weight_is_allowed_if_each_class_has_positive_weight(self):
        model = self.fit(np.array([1., 0., 0., 1.]))
        self.assertTrue(np.all(np.isfinite(model.weights)))

    def test_invalid_shapes_and_class_coverage_are_rejected(self):
        cases = (
            (np.ones(4), self.labels, 2),
            (np.ones((0, 2)), np.array([], dtype=int), 2),
            (np.ones((4, 0)), self.labels, 2),
            (self.features, self.labels[:, None], 2),
            (self.features, self.labels[:-1], 2),
            (self.features, self.labels.astype(float), 2),
            (self.features, np.array([0, 0, 0, 0]), 2),
            (self.features, np.array([0, 0, 1, 2]), 2),
        )
        for features, labels, n_classes in cases:
            with self.subTest(shape=np.shape(features), labels=labels), self.assertRaises(ValueError):
                fit_cosine_head(features, labels, n_classes)


if __name__ == "__main__":
    unittest.main()
