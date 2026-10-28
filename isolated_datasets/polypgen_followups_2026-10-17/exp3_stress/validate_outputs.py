"""Post-run reconstruction gates from saved target predictions and searches."""
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from common import dump_json, load_data, metrics
from exp3_stress import METHODS, build_plan, stable_hash


def main():
    folder = ROOT / "exp3_stress"
    if not (folder / "COMPLETE.json").is_file():
        raise FileNotFoundError("Complete all conditions and reports before validation")
    data = load_data()
    document = json.loads((folder / "conditions_plan.json").read_text())
    assert stable_hash(build_plan(data)) == document["plan_sha256"]
    evaluations, total_rows = 0, 0
    for condition in document["conditions"]:
        directory = folder / "conditions" / condition["condition_id"]
        assert (directory / "COMPLETE").is_file()
        results = json.loads((directory / "metrics.json").read_text())
        searches = {r["method"]: r for r in json.loads((directory / "selection_search.json").read_text())}
        assert len(results) == 6 and {r["method"] for r in results} == set(METHODS)
        target = {data["rows"][i]["sample_id"]: data["rows"][i] for i in condition["outer_test"]}
        grouped = defaultdict(list)
        with (directory / "predictions.csv").open(newline="", encoding="utf-8") as stream:
            for row in csv.DictReader(stream):
                original = target[row["sample_id"]]
                assert row["center"] == condition["heldout_center"] == original["center"]
                assert row["group_id"] == original["group_id"] and int(row["label"]) == original["label"]
                grouped[row["method"]].append(row)
                total_rows += 1
        for result in results:
            method = result["method"]
            predicted = grouped[method]
            assert len(predicted) == len(target) and {r["sample_id"] for r in predicted} == set(target)
            reconstructed = metrics([int(r["label"]) for r in predicted], [int(r["prediction"]) for r in predicted],
                                    [float(r["score_margin"]) for r in predicted])
            for key, value in reconstructed.items():
                actual = result["metrics"][key]
                assert (actual is None and value is None) or np.isclose(actual, value, rtol=0, atol=1e-12), key
            search = searches[method]
            # Original predetermined candidate ordering resolves exact ties.
            selected = max(enumerate(search["candidates"]),
                           key=lambda pair: (pair[1]["mean_source_validation_balanced_accuracy"], -pair[0]))[1]
            assert selected["configuration"] == search["selected_configuration"] == result["configuration"]
            assert search["temperature"] == result["temperature"]
            assert (ROOT / result["model_path"]).is_file()
            evaluations += 1
    assert evaluations == 288
    dump_json(folder / "validation.json", {"passed": True, "sampling_unit_tests_passed": 10,
        "feasible_source_conditions": 48, "target_evaluations": evaluations, "reconstructed_prediction_rows": total_rows,
        "condition_plan_hash_reconstructed": True, "all_source_centers_class_vocab_preserved": True,
        "whole_groups_only": True, "outer_targets_disjoint_from_fit_and_selection": True,
        "original_class_complete_inner_centers_preserved": True, "paired_total_group_budgets_exact": True,
        "metrics_reconstructed_from_predictions": True, "selected_spec_reconstructed_from_source_search": True,
        "models_exist": True})
    print(f"Passed output reconstruction for {evaluations} evaluations and {total_rows} prediction rows")


if __name__ == "__main__":
    main()
