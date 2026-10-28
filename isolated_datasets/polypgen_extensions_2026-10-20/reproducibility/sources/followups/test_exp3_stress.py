"""Guards for whole-group sampling and source-only nested partitions."""
import unittest

import numpy as np

from common import load_data
from exp3_stress import (build_plan, center_inventory, covering_sample, inner_partitions,
                         minimum_cover, proportional_quotas)


class SamplingGuards(unittest.TestCase):
    def test_exact_feasible_quotas(self):
        inventory = {"A": {"a": {0}, "b": {1}, "c": {0}},
                     "B": {"d": {0, 1}, "e": {1}, "f": {0}, "g": {1}}}
        quotas = proportional_quotas(inventory, 4)
        self.assertEqual(sum(quotas.values()), 4)
        for center, quota in quotas.items():
            self.assertGreaterEqual(quota, minimum_cover(inventory[center]))
            self.assertLessEqual(quota, len(inventory[center]))
        self.assertEqual(quotas, proportional_quotas(inventory, 4))

    def test_class_coverage_and_no_replacement(self):
        groups = {"a": {0}, "b": {0}, "c": {1}, "d": {1}}
        selected, detail = covering_sample(groups, 2, np.random.default_rng(17))
        self.assertEqual(len(selected), len(set(selected)))
        self.assertEqual(set().union(*(groups[g] for g in selected)), {0, 1})
        self.assertTrue(detail["feasible"])

    def test_exact_fallback(self):
        groups = {"a": {0}, "b": {0}, "c": {1}, "d": {1}}
        selected, detail = covering_sample(groups, 2, np.random.default_rng(17), rejection_limit=0)
        self.assertEqual(set().union(*(groups[g] for g in selected)), {0, 1})
        self.assertEqual(detail["sampler"], "uniform_exact_combinations")
        self.assertEqual(detail["feasible_combinations"], 4)

    def test_infeasibility_is_explicit(self):
        groups = {"a": {0}, "b": {1}}
        with self.assertRaises(ValueError):
            covering_sample(groups, 1, np.random.default_rng(17))
        with self.assertRaises(ValueError):
            proportional_quotas({"A": groups, "B": groups}, 3)

    def test_binary_mixed_group_can_cover_both_classes(self):
        self.assertEqual(minimum_cover({"a": {0, 1}, "b": {0}}), 1)


class ActualProtocolGuards(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = load_data()
        cls.plan = build_plan(cls.data)

    def test_all_predeclared_conditions_feasible(self):
        self.assertEqual(len(self.plan), 48)
        self.assertTrue(all(c["status"] == "ready" for c in self.plan))

    def test_source_groups_whole_and_original_labels_preserved(self):
        d = self.data
        for condition in self.plan:
            chosen = set(condition["outer_train"])
            outer = next(s for s in d["splits"] if s["heldout_center"] == condition["heldout_center"])
            inventory = center_inventory(outer["outer_train"], d["y"], d["centers"], d["groups"])
            for center, group_map in inventory.items():
                retained = [i for i in chosen if d["centers"][i] == center]
                self.assertEqual(set(d["y"][retained]), set().union(*group_map.values()))
                for group in condition["selected_groups"][center]:
                    all_original = set(np.flatnonzero(d["groups"] == group))
                    self.assertTrue(all_original <= chosen)

    def test_target_never_trains_or_selects_and_inner_centers_match(self):
        for condition in self.plan:
            target = set(condition["outer_test"])
            self.assertFalse(set(condition["outer_train"]) & target)
            outer = next(s for s in self.data["splits"] if s["heldout_center"] == condition["heldout_center"])
            self.assertEqual([f["validation_center"] for f in condition["inner_folds"]],
                             [f["validation_center"] for f in outer["inner_folds"]])
            for fold in condition["inner_folds"]:
                self.assertFalse((set(fold["train"]) | set(fold["validation"])) & target)
                self.assertFalse(set(fold["train"]) & set(fold["validation"]))

    def test_paired_budget_exact_and_all_source_centers_retained(self):
        for condition in self.plan:
            mate = next(c for c in self.plan if c["heldout_center"] == condition["heldout_center"]
                        and c["requested_dominant_fraction"] == condition["requested_dominant_fraction"]
                        and c["subset_seed"] == condition["subset_seed"] and c["arm"] != condition["arm"])
            self.assertEqual(condition["composition"]["total_groups"], mate["composition"]["total_groups"])
            self.assertEqual(set(condition["composition"]["centers"]), set(condition["baseline_composition"]["centers"]))

    def test_invalid_target_and_inner_class_coverage_rejected(self):
        outer = self.data["splits"][0]
        with self.assertRaises(ValueError):
            inner_partitions(outer["outer_train"] + [outer["outer_test"][0]], outer, self.data)
        bad = [i for i in outer["outer_train"] if self.data["centers"][i] == "C1"]
        with self.assertRaises(ValueError):
            inner_partitions(bad, outer, self.data)


if __name__ == "__main__":
    unittest.main(verbosity=2)
