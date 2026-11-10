"""Audit fixed prototype choices using saved PolypGen scores and group resampling."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from audit_group_ranking import concordant_count, summarize


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "isolated_datasets/polypgen_v3_copy_2026-10-17"
EXT = ROOT / "isolated_datasets/polypgen_extensions_2026-10-20"
INPUTS = {
    "ResNet18": ROOT / "isolated_datasets/polypgen_followups_2026-10-17/exp2_factorial/predictions.csv",
    "DINOv2": EXT / "exp4_dinov2/factorial/predictions.csv",
}
SELECTIONS = {"ResNet18": BASE / "results", "DINOv2": EXT / "exp4_dinov2/main"}
SEEDS = (17, 29, 43)
CENTERS = ("C2", "C4", "C5", "C6")
CELLS = ("A_K1_beta0", "B_K1_beta1", "C_K4_beta0", "D_K4_beta1")
METRICS = ("pooled_auc", "within_auc", "between_auc")
CONTRASTS = {
    "balancing_at_K1": (-1, 1, 0, 0),
    "balancing_at_K4": (0, 0, -1, 1),
    "multiplicity_at_beta0": (-1, 0, 1, 0),
    "multiplicity_at_beta1": (0, -1, 0, 1),
    "interaction": (1, -1, -1, 1),
}


def pair_matrices(frame, groups):
    """Return pair counts and concordances indexed by the original group IDs."""
    positives, negatives = [], []
    for group in groups:
        local = frame.loc[frame.group_id == group]
        positives.append(local.loc[local.label == 1, "probability_class_1"].to_numpy())
        negatives.append(np.sort(local.loc[local.label == 0, "probability_class_1"].to_numpy()))
    pairs = np.outer([len(p) for p in positives], [len(n) for n in negatives])
    wins = np.array([[concordant_count(p, n) for n in negatives] for p in positives])
    return pairs, wins


def resampled_metrics(pair_counts, concordances, multiplicities):
    """Count repeated groups jointly for every saved head, without refitting."""
    total_pairs = np.einsum("bi,ij,bj->b", multiplicities, pair_counts, multiplicities)
    same_pairs = np.einsum("bi,i,bi->b", multiplicities, np.diag(pair_counts), multiplicities)
    total_wins = np.einsum("bi,cij,bj->bc", multiplicities, concordances, multiplicities)
    same_wins = np.einsum(
        "bi,ci,bi->bc", multiplicities, np.diagonal(concordances, axis1=1, axis2=2), multiplicities
    )
    counts = (total_pairs, same_pairs, total_pairs - same_pairs)
    wins = (total_wins, same_wins, total_wins - same_wins)
    result = np.full((len(multiplicities), len(concordances), len(METRICS)), np.nan)
    for index, (numerator, denominator) in enumerate(zip(wins, counts)):
        np.divide(numerator, denominator[:, None], out=result[:, :, index],
                  where=denominator[:, None] > 0)
    return result


def contrast_rows(encoder, center, point, bootstrap):
    records = []
    for name, coefficients in CONTRASTS.items():
        active = np.flatnonzero(coefficients)
        weights = np.array(coefficients)[active]
        for index, metric in enumerate(METRICS):
            delta = bootstrap[:, active, index] @ weights
            valid = delta[np.isfinite(delta)]
            interval = np.quantile(valid, [.025, .975]) if len(valid) else (np.nan, np.nan)
            records.append({
                "encoder": encoder, "center": center, "contrast": name, "metric": metric,
                "difference": point[active, index] @ weights,
                "ci_low": interval[0], "ci_high": interval[1],
                "valid_draws": len(valid), "undefined_draws": len(delta) - len(valid),
            })
    return records


def audit(output, attempts, seed):
    frames, fingerprints = {}, []
    reference = None
    for encoder, path in INPUTS.items():
        frame = pd.read_csv(path, float_precision="round_trip")
        if (set(frame.seed) != set(SEEDS) or set(frame.center) != set(CENTERS)
                or set(frame.cell) != set(CELLS) or frame.group_id.isna().any()
                or frame.sample_id.isna().any() or not set(frame.label) <= {0, 1}
                or not frame.probability_class_1.between(0, 1).all()):
            raise ValueError(f"Unexpected factorial export: {encoder}")
        for (fit_seed, cell), local in frame.groupby(["seed", "cell"]):
            support = local.set_index("sample_id")[["center", "group_id", "label"]].sort_index()
            if not support.index.is_unique:
                raise ValueError(f"Duplicate sample: {encoder}, {fit_seed}, {cell}")
            if reference is None:
                reference = support
            else:
                pd.testing.assert_frame_equal(support, reference)
            expected_k, expected_beta = ((1, 0), (1, 1), (4, 0), (4, 1))[CELLS.index(cell)]
            if not (local.K.eq(expected_k).all() and local.beta.eq(expected_beta).all()):
                raise ValueError(f"Incorrect cell metadata: {cell}")
        frames[encoder] = frame
        fingerprints.append({"input": path.relative_to(ROOT).as_posix(),
                             "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "rows": len(frame)})

    generator = np.random.default_rng(seed)
    resamples = {}
    for center in CENTERS:
        groups = sorted(reference.loc[reference.center == center, "group_id"].unique())
        resamples[center] = (groups, generator.multinomial(
            len(groups), np.full(len(groups), 1 / len(groups)), size=attempts))

    by_seed, contrasts, settings, source_checks = [], [], [], []
    for encoder, frame in frames.items():
        center_points, center_bootstrap = [], []
        for center in CENTERS:
            groups, counts = resamples[center]
            cell_concordances = []
            for cell in CELLS:
                seed_concordances = []
                for fit_seed in SEEDS:
                    local = frame.loc[(frame.center == center) & (frame.cell == cell) & (frame.seed == fit_seed)]
                    pairs, wins = pair_matrices(local, groups)
                    summary = summarize(local)
                    check = resampled_metrics(pairs, wins[None], np.ones((1, len(groups))))[0, 0]
                    np.testing.assert_allclose(check, [summary[k] for k in METRICS], rtol=0, atol=1e-12)
                    by_seed.append({"encoder": encoder, "center": center, "cell": cell, "seed": fit_seed,
                                    "K": int(local.K.iloc[0]), "beta": float(local.beta.iloc[0]), **summary})
                    seed_concordances.append(wins)
                cell_concordances.append(np.mean(seed_concordances, axis=0))
            wins = np.stack(cell_concordances)
            point = resampled_metrics(pairs, wins, np.ones((1, len(groups))))[0]
            bootstrap = resampled_metrics(pairs, wins, counts)
            center_points.append(point)
            center_bootstrap.append(bootstrap)
            contrasts.extend(contrast_rows(encoder, center, point, bootstrap))
        macro_point = np.mean(center_points, axis=0)
        macro_bootstrap = np.mean(center_bootstrap, axis=0)
        contrasts.extend(contrast_rows(encoder, "equal_center_mean", macro_point, macro_bootstrap))

        for fit_seed in SEEDS:
            selected = SELECTIONS[encoder] / f"seed{fit_seed}/metrics_by_center.json"
            search = SELECTIONS[encoder] / f"seed{fit_seed}/selection_search.json"
            records = json.loads(selected.read_text(encoding="utf-8-sig"))
            candidates = json.loads(search.read_text(encoding="utf-8-sig"))
            for record in records:
                if record["method"] != "cbmp":
                    continue
                center = record["heldout_center"]
                settings.append({"encoder": encoder, "center": center, "seed": fit_seed,
                                 **record["configuration"], "temperature": record["temperature"]})
                entry = next(r for r in candidates if r["heldout_center"] == center and r["method"] == "cbmp")
                for candidate in entry["candidates"]:
                    config = candidate["configuration"]
                    source_checks.append({"encoder": encoder, "center": center, "seed": fit_seed,
                                          **config, "source_validation_ba":
                                          candidate["mean_source_validation_balanced_accuracy"],
                                          "selected": config == record["configuration"]})
            for path in (selected, search):
                fingerprints.append({"input": path.relative_to(ROOT).as_posix(),
                                     "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})

    paired_path = EXT / "exp4_dinov2/report/paired_sequence_bootstrap.npz"
    with np.load(paired_path, allow_pickle=False) as saved:
        methods = list(saved["methods"])
        offset = len(methods)
        draws = saved["mean"][:, offset + methods.index("cbmp")] - saved["mean"][
            :, offset + methods.index("trainable_cosine")]
    paired_values = []
    for fit_seed in SEEDS:
        metrics = json.loads((SELECTIONS["DINOv2"] / f"seed{fit_seed}/metrics_by_center.json").read_text())
        for center in CENTERS:
            scores = {r["method"]: r["metrics"]["balanced_accuracy"]
                      for r in metrics if r["heldout_center"] == center}
            paired_values.append(scores["cbmp"] - scores["trainable_cosine"])
    fingerprints.append({"input": paired_path.relative_to(ROOT).as_posix(),
                         "sha256": hashlib.sha256(paired_path.read_bytes()).hexdigest()})
    seed_frame = pd.DataFrame(by_seed)
    means = seed_frame.drop(columns="seed").groupby(["encoder", "center", "cell"], as_index=False).mean()
    output.mkdir(parents=True, exist_ok=False)
    seed_frame.to_csv(output / "factorial_ranking_by_seed.csv", index=False)
    means.to_csv(output / "factorial_ranking_means.csv", index=False)
    pd.DataFrame(contrasts).to_csv(output / "factorial_ranking_contrasts.csv", index=False)
    pd.DataFrame(settings).to_csv(output / "cbmp_selected_settings.csv", index=False)
    pd.DataFrame(source_checks).to_csv(output / "cbmp_source_validation.csv", index=False)
    pd.DataFrame([{
        "encoder": "DINOv2", "metric": "equal_center_mean_balanced_accuracy",
        "comparison": "cbmp_minus_trainable_cosine", "difference": np.mean(paired_values),
        "ci_low": np.quantile(draws, .025), "ci_high": np.quantile(draws, .975),
        "valid_draws": len(draws), "scope": "existing paired sequence-bootstrap artifact; fixed fitted heads",
    }]).to_csv(output / "paired_main_contrast.csv", index=False)
    (output / "provenance.json").write_text(json.dumps({
        "scope": "post-hoc analysis of saved fixed-factorial predictions and source selections",
        "attempts": attempts, "seed": seed, "seeds": SEEDS, "cells": CELLS,
        "units": "recorded sequences sampled with replacement within each fixed target center",
        "pair_weight": "product of original-group bootstrap multiplicities; ties receive half credit",
        "coupling": "identical draws for both encoders, all fixed settings, and fitting seeds",
        "aggregation": "seed means, then equal center means for draws defined at every center",
        "intervals": "percentile intervals conditional on defined components and fixed fitted heads",
        "input_files": fingerprints,
    }, indent=2) + "\n", encoding="utf-8")
    print(f"Saved {len(seed_frame)} seed cells, {len(means)} mean cells, {len(contrasts)} paired contrasts")
    print(pd.DataFrame(contrasts).query("contrast == 'balancing_at_K4' and metric == 'between_auc'").to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--attempts", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=2027)
    args = parser.parse_args()
    if args.attempts < 1:
        parser.error("--attempts must be positive")
    audit(args.output_dir, args.attempts, args.seed)
