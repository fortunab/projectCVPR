"""Assemble verified exploratory experiment reports without changing the manuscript."""
from __future__ import annotations

import csv
import importlib.metadata
import json
import platform
from pathlib import Path

from common import ROOT, dump_json, sha256


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def pct(value):
    return f"{100 * float(value):.1f}"


def main():
    required = [ROOT / "exp1_failure" / "COMPLETE.json", ROOT / "exp2_factorial" / "COMPLETE",
                ROOT / "exp3_stress" / "COMPLETE.json", ROOT / "preservation_check.json"]
    if any(not path.is_file() for path in required):
        raise ValueError("All three experiments and final preservation check must finish first")
    setup = json.loads((ROOT / "setup_complete.json").read_text())
    if sha256(ROOT / "PROTOCOL.md") != setup["protocol_sha256"]:
        raise ValueError("Frozen protocol changed")
    preservation = json.loads((ROOT / "preservation_check.json").read_text())
    if not preservation["unchanged"]:
        raise ValueError("Original inputs or outputs changed; inspect preservation check")
    exp1 = json.loads(required[0].read_text())
    exp2 = json.loads((ROOT / "exp2_factorial" / "completion.json").read_text())
    exp3 = json.loads(required[2].read_text())
    if exp1["n_threshold_controls"] != 108 or exp2["final_evaluations"] != 48 or exp3["new_target_evaluations"] != 288:
        raise ValueError("Incomplete evaluation counts")
    threshold = read_csv(ROOT / "exp1_failure" / "summary_metrics.csv")
    factorial = read_csv(ROOT / "exp2_factorial" / "summary_metrics.csv")
    effects = read_csv(ROOT / "exp2_factorial" / "paired_effects.csv")
    stress = read_csv(ROOT / "exp3_stress" / "summary.csv")
    sequences = read_csv(ROOT / "exp1_failure" / "sequence_metrics_seed_means.csv")
    cbmp_c5_seq5 = next(r for r in sequences if r["center"] == "C5" and r["method"] == "cbmp"
                        and r["group_id"] == "sequence:seq5" and r["decision_rule"] == "zero")
    cbmp_original = next(r for r in threshold if r["method"] == "cbmp" and r["decision_rule"] == "zero")
    cbmp_threshold = next(r for r in threshold if r["method"] == "cbmp" and r["decision_rule"] == "source_selected")
    text = ["# Experiments 1–3: exploratory follow-up results", "",
            "All three experiments completed on the same audited 5,444-frame/37-sequence subset, frozen ImageNet ResNet18 features, and four held-out centers C2,C4,C5,C6. The original test results had already been inspected when this follow-up protocol was designed, so independent confirmation remains necessary.", "",
            "## Principal findings", "",
            f"1. Source-only threshold control changes CBMP mean balanced accuracy from {pct(cbmp_original['mean_center_balanced_accuracy'])}% to {pct(cbmp_threshold['mean_center_balanced_accuracy'])}%, with worst-center performance remaining {pct(cbmp_threshold['worst_center_balanced_accuracy'])}%. It does not resolve the observed failure.",
            f"2. Within C5's mixed seq5, CBMP AUROC is {pct(cbmp_c5_seq5['auroc'])}%, while pooled C5 AUROC is43.9%. Between-sequence score differences are an important descriptive lead. This does not establish a causal imaging artifact or patient-level effect.",
            "3. Fixed multiplicity and center balancing do not consistently improve transfer. At K1, equal center weighting lowers the mean; K4 harms the mean relative to the unbalanced K1 control. Component interactions differ by center and must not be interpreted as universal invariance.",
            "4. Source-composition stress gives no consistent CBMP advantage over matched total-group scarcity. Linear logistic retains the strongest mean performance in the evaluated stress conditions.", "",
            "## Experiment 1: source-selected thresholds and failure analysis", "",
            "Nine original classifiers x four targets x three fitting seeds =108 threshold controls. The existing final heads are reused; their selected settings stay fixed. Each decision threshold uses only refitted inner source-fold margins. Score probabilities and AUROC do not change. All140,967 original zero-threshold predictions were reproduced exactly.", "",
            "| Classifier | Original mean BA (%) | Source-threshold mean BA (%) | Original worst BA (%) | Adjusted worst BA (%) |",
            "|---|---:|---:|---:|---:|"]
    names = {"cbmp": "CBMP", "multi_prototype_unbalanced": "Unbalanced multiple prototypes", "linear_logistic": "Linear logistic",
             "trainable_cosine": "Trained cosine", "linear_group_center_weighted": "Group/center weighted linear",
             "cosine_group_center_weighted": "Group/center weighted cosine", "single_centroid_group_weighted": "Group weighted centroid",
             "single_centroid_center_balanced": "Center balanced centroid", "single_centroid_image_weighted": "Image weighted centroid"}
    for method in dict.fromkeys(r["method"] for r in threshold):
        zero = next(r for r in threshold if r["method"] == method and r["decision_rule"] == "zero")
        changed = next(r for r in threshold if r["method"] == method and r["decision_rule"] == "source_selected")
        text.append(f"| {names[method]} | {pct(zero['mean_center_balanced_accuracy'])} | {pct(changed['mean_center_balanced_accuracy'])} | {pct(zero['worst_center_balanced_accuracy'])} | {pct(changed['worst_center_balanced_accuracy'])} |")
    text += ["", "C2 has strong ranking but a poorly transferring operating point; its source-selected threshold reduces CBMP balanced accuracy from60.3% to57.7%. C5 has poor pooled ranking. Its single mixed sequence has high within-sequence ranking, while the three negative-only sequences have much lower specificity. Prototype contribution/effective-group tables, class-score summaries, and retrieval figures describe these associations without establishing clinical explanations.",
             "", "[Experiment 1 report](exp1_failure/RESULTS.md) | [Threshold table](exp1_failure/table_threshold_control.tex) | [Sequence metrics](exp1_failure/sequence_metrics_seed_means.csv)",
             "", "![Score and ROC diagnostics](exp1_failure/failure_score_diagnostics.png)",
             "", "## Experiment 2: fixed K x beta factorial", "",
             "Four fixed cells x four targets x three fitting seeds =48 final evaluations. Group weighting is constant. Every realized class-prototype count matched its requested K. All24 K1 comparisons reproduced the corresponding original centroid predictions exactly. Temperatures use source-only validation NLL; no target score chooses a cell.", "",
             "| K | beta | Mean BA (%) | Worst-center BA (%) | Mean AUROC (%) |", "|---:|---:|---:|---:|---:|"]
    for row in factorial:
        text.append(f"| {row['K']} | {float(row['beta']):.0f} | {pct(row['mean_center_balanced_accuracy'])} | {pct(row['worst_center_balanced_accuracy'])} | {pct(row['mean_center_auroc'])} |")
    text += ["", "| Paired contrast | Mean BA change (pp) | Conditional95% interval (pp) |", "|---|---:|---:|"]
    for row in effects:
        if row["scope"] == "equal_center_mean":
            text.append(f"| {row['contrast']} | {100*float(row['effect_balanced_accuracy']):+.2f} | [{100*float(row['conditional_ci_low']):+.2f}, {100*float(row['conditional_ci_high']):+.2f}] |")
    text += ["", "The2,000 whole-sequence bootstrap attempts accepted1,026 class-complete draws. Intervals pair identical resamples across cells and fitting seeds, are conditional on fixed trained heads and four observed centers, and do not establish independent patients or population-level hospital uncertainty. These exploratory component intervals are descriptive.",
             "", "[Experiment 2 report](exp2_factorial/RESULTS.md) | [Factorial table](exp2_factorial/table_factorial.tex) | [Per-center effects](exp2_factorial/paired_effects.csv)",
             "", "![Fixed factorial](exp2_factorial/factorial_mean_worst.png)",
             "", "## Experiment 3: source composition versus matched group scarcity", "",
             "48 unique retained-source conditions x six classifier variants =288 new target evaluations. Fitting seed17 stays fixed; source-subset seeds101,211,307 vary whole retained sequences. Each source center keeps its original observed class vocabulary, and every original class-complete source-validation center remains available. All288 selected configurations and metrics were reconstructed from saved searches and375,912 per-frame predictions.",
             "", "The composition arm reduces the source center with the largest number of groups: C6 for targets C2,C4,C5, and C4 for target C6. The paired scarcity control retains the same total number of groups with proportional class-covering source quotas. Frame counts and label prevalence are not matched and are disclosed. Reducing a dominant center may reduce source imbalance; retention is not a monotonic increasing-domain-shift scale.",
             "", "| Classifier | Dominant-center retention | Composition mean / worst BA (%) | Matched mean / worst BA (%) | Composition minus matched (pp) |",
             "|---|---:|---:|---:|---:|"]
    for method in dict.fromkeys(r["method"] for r in stress):
        for fraction in (.5, .25):
            a = next(r for r in stress if r["method"] == method and float(r["fraction"]) == fraction and r["arm"] == "composition")
            b = next(r for r in stress if r["method"] == method and float(r["fraction"]) == fraction and r["arm"] == "matched_scarcity")
            text.append(f"| {names[method]} | {100*fraction:.0f}% | {pct(a['mean_center_balanced_accuracy'])} / {pct(a['worst_center_balanced_accuracy'])} | {pct(b['mean_center_balanced_accuracy'])} / {pct(b['worst_center_balanced_accuracy'])} | {100*float(a['composition_minus_matched_mean_bAcc']):+.1f} |")
    text += ["", "CBMP's composition-minus-control effect changes sign between retention levels. The three subset-seed effects vary substantially; no naive pooled p-value or confidence interval over hospitals is claimed. Each center is weighted equally after averaging its subset metrics. The full-source anchors use original seed17 only, separately from the three-fitting-seed averages used in experiments1–2.",
             "", "[Experiment 3 report](exp3_stress/RESULTS.md) | [Sampling audit](exp3_stress/SAMPLING.md) | [Stress table](exp3_stress/table_stress.tex)",
             "", "![Composition and scarcity](exp3_stress/figures/performance_retention.png)",
             "", "## Relation to the manuscript", "",
             "The current results support a conditional, diagnostic account of prototype transfer. They do not support an empirical advantage for the current CBMP mechanism. The most informative observation is the mismatch between within-sequence and pooled-center ranking, alongside component sensitivity and sparse source support. Any new mechanism suggested by these findings needs a separately frozen independent evaluation.",
             "", "The manuscript still needs reconciliation with actual execution before experiments are inserted: this study uses frozen ResNet18 and a partial reviewed sequence subset; temperature selection is the implementation's bounded continuous source-NLL optimization, whereas the draft lists a discrete grid. Existing manuscript files were not edited.",
             "", "## Verification and preservation", "",
             f"Experiment1:25 focused checks and108 cached source-threshold endpoint checks. Experiment2:24 exact baseline-centroid prediction comparisons and all48 realized-capacity checks. Experiment3:10 focused sampling/partition checks, then reconstruction of288 target records and their source-selected settings. Preservation check: {preservation['files_checked_size_mtime']:,} protected files unchanged by size/mtime; {preservation['baseline_artifacts_hashed']} baseline artifacts unchanged by SHA-256.",
             "", "All results remain conditional on the reviewed subset. C2 negatives and C5 positives each occur in one sequence. Sequence IDs do not establish patient independence. Unresolved release center mappings remain excluded. No new dataset or encoder was introduced.",
             "", "The exact frozen design is in [PROTOCOL.md](PROTOCOL.md). Raw evidence, code, logs, models, CSVs, figures, and TeX fragments are local to this folder. The previous data, runs, outputs, and open manuscript remain unchanged."]
    report = "\n".join(text) + "\n"
    for original, corrected in {
        "is43.9%": "is 43.9%", "=108": "= 108", "=48": "= 48", "=288": "= 288",
        "All140,967": "All 140,967", "from60.3% to57.7%": "from 60.3% to 57.7%",
        "All24 K1": "All 24 K=1", "The2,000": "The 2,000", "accepted1,026": "accepted 1,026",
        "seed17": "seed 17", "seeds101,211,307": "seeds 101, 211, and 307",
        "All288": "All 288", "and375,912": "and 375,912", "experiments1–2": "experiments 1–2",
        "Experiment1:25": "Experiment 1: 25", "and108": "and 108", "Experiment2:24": "Experiment 2: 24",
        "all48": "all 48", "Experiment3:10": "Experiment 3: 10", "of288": "of 288",
        "Conditional95%": "Conditional 95%"
    }.items():
        report = report.replace(original, corrected)
    (ROOT / "RESULTS.md").write_text(report, encoding="utf-8")
    environment = {"python": platform.python_version(), "platform": platform.platform(),
                   "libraries": {name: importlib.metadata.version(name) for name in ("numpy", "scipy", "scikit-learn", "joblib", "matplotlib", "threadpoolctl")}}
    dump_json(ROOT / "environment.json", environment)
    dump_json(ROOT / "FOLLOWUPS_COMPLETE.json", {"exploratory": True, "threshold_controls": 108,
                                                "factorial_final_evaluations": 48, "stress_target_evaluations": 288,
                                                "protocol_sha256": setup["protocol_sha256"], "baseline_preserved": True,
                                                "interpretation": "No consistent robustness advantage established for current CBMP on this subset"})
    hashes = {path.relative_to(ROOT).as_posix(): sha256(path) for path in ROOT.rglob("*")
              if path.is_file() and "__pycache__" not in path.parts and ".pytest_cache" not in path.parts
              and path.name != "artifacts_manifest.json"}
    dump_json(ROOT / "artifacts_manifest.json", {"file_sha256": hashes, "count": len(hashes),
                                                "scope": "All completed followup artifacts except mutable interpreter/test caches and this checksum ledger"})
    print(f"Integrated results saved: {ROOT / 'RESULTS.md'}; {len(hashes)} artifacts hashed")


if __name__ == "__main__":
    main()
