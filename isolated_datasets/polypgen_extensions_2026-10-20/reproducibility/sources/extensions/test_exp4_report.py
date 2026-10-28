"""Check paired group resampling and prevention of mixed-cohort comparisons."""
import csv

import numpy as np
import pytest

from exp4_dinov2 import CENTERS, SEEDS
from exp4_report import load_predictions, paired_bootstrap


def synthetic_rows():
    return [{"sample_id": f"{center}_{label}", "center": center,
             "group_id": f"{center}_group{label}", "label": label}
            for center in CENTERS for label in (0, 1)]


def test_exp4_bootstrap_uses_identical_whole_groups_across_encoders():
    rows = synthetic_rows()
    labels = np.array([r["label"] for r in rows])
    # Encoder 1 is perfect; encoder 2 predicts only negatives. This contrast
    # must remain -0.5 in every paired class-complete group resample.
    predictions = np.stack([np.stack([labels, np.zeros_like(labels)])] * len(SEEDS))
    mean, worst, metadata = paired_bootstrap(rows, list(range(len(rows))), predictions)
    assert 2 <= metadata["accepted"] < metadata["attempted"]
    assert np.all(mean[:, 1] - mean[:, 0] == -.5)
    assert np.all(worst[:, 1] - worst[:, 0] == -.5)


@pytest.mark.parametrize("corruption", ("label", "group_id", "duplicate"))
def test_exp4_paired_predictions_reject_different_cohorts(tmp_path, corruption):
    rows = synthetic_rows()
    dirs = {}
    for seed in SEEDS:
        dirs[seed] = tmp_path / str(seed)
        dirs[seed].mkdir()
        records = [{**row, "method": "control", "prediction": row["label"]} for row in rows]
        if seed == SEEDS[0]:
            if corruption == "duplicate":
                records.append(records[0].copy())
            elif corruption == "label":
                records[0]["label"] = 1
            else:
                records[0]["group_id"] = "different_patient"
        with (dirs[seed] / "predictions.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)
    with pytest.raises(ValueError):
        load_predictions(rows, dirs, ("control",), "method")
