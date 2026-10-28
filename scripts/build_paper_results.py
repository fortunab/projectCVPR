"""Verify saved predictions and generate manuscript tables without fitting models."""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.metrics import balanced_accuracy_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
EXT = ROOT / "isolated_datasets/polypgen_extensions_2026-10-20"
BASE = ROOT / "isolated_datasets/polypgen_v3_copy_2026-10-17"
METHODS = {
    "cbmp": "CBMP",
    "multi_prototype_unbalanced": "Unbalanced multi-prototype",
    "linear_logistic": "Linear logistic",
    "trainable_cosine": "Trained cosine",
    "linear_group_center_weighted": "Weighted linear",
    "cosine_group_center_weighted": "Weighted cosine",
    "single_centroid_group_weighted": "Group centroid",
    "single_centroid_center_balanced": "Center centroid",
    "single_centroid_image_weighted": "Image centroid",
}


def rows(path):
    with path.open(newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_predictions(paths, centers, n_samples):
    cells, identity, seen = defaultdict(list), {}, set()
    for seed_hint, path in paths:
        for row in rows(path):
            seed = int(row.get("seed") or seed_hint)
            method = row["method"]
            center = str(row.get("center", row.get("hospital")))
            sample, group = row["sample_id"], row["group_id"]
            label, pred, prob = int(row["label"]), int(row["prediction"]), float(row["probability_class_1"])
            key = (seed, method, center, sample)
            if key in seen or label not in (0, 1) or pred not in (0, 1) or not np.isfinite(prob) or not 0 <= prob <= 1:
                raise ValueError(f"Invalid/duplicated prediction: {key}")
            seen.add(key)
            support = (center, group, label)
            if sample in identity and identity[sample] != support:
                raise ValueError("Inconsistent sample/label/group support")
            identity[sample] = support
            cells[seed, method, center].append((sample, label, pred, prob))
    expected = {(s, m, c) for s in (17, 29, 43) for m in METHODS for c in centers}
    if set(cells) != expected or len(identity) != n_samples:
        raise ValueError("Incomplete prediction grid or wrong cohort support")
    per_center_support = {c: {k for k, v in identity.items() if v[0] == c} for c in centers}
    metrics = {}
    for key, data in cells.items():
        if {r[0] for r in data} != per_center_support[key[2]]:
            raise ValueError("Methods/seeds do not share identical target samples")
        y, pred, prob = [np.array([r[i] for r in data]) for i in (1, 2, 3)]
        metrics[key] = (float(balanced_accuracy_score(y, pred)), float(roc_auc_score(y, prob)))
    return metrics, identity, len(seen)


def assert_close(actual, expected, description):
    if not np.isclose(actual, float(expected), atol=1e-12, rtol=0):
        raise ValueError(f"Stored summary differs from predictions: {description}: {actual} vs {expected}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    out = args.output_dir.resolve()
    if out.exists():
        raise FileExistsError("Choose a new output directory; earlier outputs are preserved")
    paths_r = [(s, BASE / f"results/seed{s}/predictions.csv") for s in (17, 29, 43)]
    paths_d = [(s, EXT / f"exp4_dinov2/main/seed{s}/predictions.csv") for s in (17, 29, 43)]
    paths_c = [(None, EXT / "exp5_camelyon/evaluation/predictions.csv")]
    centers = ("C2", "C4", "C5", "C6")
    r, support_r, count_r = verify_predictions(paths_r, centers, 5221)
    d, support_d, count_d = verify_predictions(paths_d, centers, 5221)
    c, _, count_c = verify_predictions(paths_c, ("2", "1"), 6800)
    if support_r != support_d:
        raise ValueError("ResNet18/DINOv2 comparison is not paired")
    summary_path = EXT / "exp4_dinov2/report/summary_metrics.csv"
    cam_path = EXT / "exp5_camelyon/report/summary_by_hospital.csv"
    summaries = {(x["encoder"], x["method"]): x for x in rows(summary_path)}
    cam = {(x["hospital"], x["method"]): x for x in rows(cam_path)}
    for enc, data in (("ResNet18", r), ("DINOv2 base", d)):
        for method in METHODS:
            center_mean = [np.mean([data[s, method, ce][0] for s in (17, 29, 43)]) for ce in centers]
            stored = summaries[enc, method]
            assert_close(np.mean(center_mean), stored["mean_center_balanced_accuracy"], (enc, method, "BA"))
            assert_close(np.min(center_mean), stored["worst_center_balanced_accuracy"], (enc, method, "worst"))
            assert_close(np.mean([v[1] for k, v in data.items() if k[1] == method]), stored["mean_center_auroc"], (enc, method, "AUC"))
    for (hospital, method), stored in cam.items():
        for metric, index in (("balanced_accuracy", 0), ("auroc", 1)):
            assert_close(np.mean([c[s, method, hospital][index] for s in (17, 29, 43)]), stored[metric + "_mean"], (hospital, method, metric))
    out.mkdir(parents=True)
    tex = [r"% Generated from prediction-verified CSVs by scripts/build_paper_results.py.",
           r"\begin{table*}[t]\centering\small",
           r"\setlength{\tabcolsep}{5pt}", r"\begin{tabular}{lrrrrrr}\toprule",
           r" & \multicolumn{3}{c}{ResNet18} & \multicolumn{3}{c}{DINOv2 ViT-B/14}\\",
           r"Classifier & Mean BA & Worst BA & AUROC & Mean BA & Worst BA & AUROC\\\midrule"]
    for method, name in METHODS.items():
        values = [100 * float(summaries[enc, method][field]) for enc in ("ResNet18", "DINOv2 base")
                  for field in ("mean_center_balanced_accuracy", "worst_center_balanced_accuracy", "mean_center_auroc")]
        tex.append(name + " & " + " & ".join(f"{v:.2f}" for v in values) + r"\\")
    tex += [r"\bottomrule\end{tabular}",
            r"\caption{PolypGen reviewed-sequence subset. All values are percentages, averaged over three fitting seeds and equally over C2, C4, C5 and C6. Worst BA is the minimum center-specific seed mean. AUROC is the equal-center mean. All heads use the same features and source-only folds within an encoder; tuning grids differ as disclosed in the text.}",
            r"\label{tab:polypgen_main}\end{table*}"]
    (out / "table_polypgen.tex").write_text("\n".join(tex) + "\n", encoding="utf-8")
    tex = [r"\begin{table}[t]\centering\footnotesize\setlength{\tabcolsep}{3pt}",
           r"\begin{tabular}{lrrrr}\toprule", r" & \multicolumn{2}{c}{Primary H2} & \multicolumn{2}{c}{Secondary H1}\\",
           r"Classifier & BA & AUROC & BA & AUROC\\\midrule"]
    for method, name in METHODS.items():
        values = [100 * float(cam[h, method][field]) for h in ("2", "1") for field in ("balanced_accuracy_mean", "auroc_mean")]
        tex.append(name + " & " + " & ".join(f"{v:.2f}" for v in values) + r"\\")
    tex += [r"\bottomrule\end{tabular}", r"\caption{Patient-capped CAMELYON17-WILDS subset with frozen ResNet18. Percentages average three fitting seeds. Hospital 2 and hospital 1 are separate prespecified evaluations; neither selects settings. This is a custom subset evaluation.}", r"\label{tab:camelyon_main}\end{table}"]
    (out / "table_camelyon.tex").write_text("\n".join(tex) + "\n", encoding="utf-8")
    factorial_path = EXT / "exp4_dinov2/report/factorial_encoder_summary.csv"
    fac = {(x["encoder"], x["method"]): x for x in rows(factorial_path)}
    tex = [r"\begin{table}[t]\centering\small\begin{tabular}{rrrr}\toprule",
           r"$K$ & $\beta$ & ResNet18 BA & DINOv2 BA\\\midrule"]
    for key, k, beta in (("A_K1_beta0", 1, 0), ("B_K1_beta1", 1, 1), ("C_K4_beta0", 4, 0), ("D_K4_beta1", 4, 1)):
        vals = [100 * float(fac[enc, key]["mean_center_balanced_accuracy"]) for enc in ("ResNet18", "DINOv2 base")]
        tex.append(f"{k} & {beta} & {vals[0]:.2f} & {vals[1]:.2f}" + r"\\")
    tex += [r"\bottomrule\end{tabular}", r"\caption{Fixed prototype-count and balancing ablation on PolypGen. Group weighting is retained in every cell; settings are fixed rather than chosen from target performance. BA is expressed as a percentage.}", r"\label{tab:factorial}\end{table}"]
    (out / "table_factorial.tex").write_text("\n".join(tex) + "\n", encoding="utf-8")
    inputs = [p for _, p in paths_r + paths_d + paths_c] + [summary_path, cam_path, factorial_path]
    evidence = {"prediction_records_verified": {"polypgen_resnet18": count_r, "polypgen_dinov2": count_d, "camelyon": count_c},
                "main_evaluations": {"polypgen_resnet18": len(r), "polypgen_dinov2": len(d), "camelyon": len(c)},
                "point_metrics_recomputed_from_predictions": True, "summary_tolerance": 1e-12,
                "new_training_performed": False, "new_bootstrap_performed": False,
                "uncertainty_source": "Existing scoped reports; this command verifies point metrics and paired support.",
                "inputs": [{"path": p.relative_to(ROOT).as_posix(), "sha256": sha(p)} for p in inputs]}
    (out / "verification.json").write_text(json.dumps(evidence, indent=2), encoding="utf-8")
    print(json.dumps({k: evidence[k] for k in ("prediction_records_verified", "main_evaluations", "point_metrics_recomputed_from_predictions")}, indent=2))


if __name__ == "__main__":
    main()
