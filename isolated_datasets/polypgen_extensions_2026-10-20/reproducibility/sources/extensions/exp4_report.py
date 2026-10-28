"""Paired DINOv2/ResNet18 reports using completed, unchanged cohort fits."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from exp4_dinov2 import (BASE, CELLS, CENTERS, OUT, ROOT, SEEDS, dump_json,
                          load_cohort, manifest_digest, sha256, write_csv)

ENCODERS = ("ResNet18", "DINOv2 base")
LABELS = {"cbmp": "CBMP", "multi_prototype_unbalanced": "Unbalanced multiprototype",
          "linear_logistic": "Linear logistic", "trainable_cosine": "Trained cosine",
          "linear_group_center_weighted": "Weighted linear",
          "cosine_group_center_weighted": "Weighted cosine",
          "single_centroid_group_weighted": "Group weighted centroid",
          "single_centroid_center_balanced": "Center balanced centroid",
          "single_centroid_image_weighted": "Image weighted centroid"}
METHODS = tuple(LABELS)
REPORT = OUT / "report"
COMPONENT_CONTRASTS = {
    "multiplicity_at_beta0": np.array([-1, 0, 1, 0]),
    "multiplicity_at_beta1": np.array([0, -1, 0, 1]),
    "balancing_at_K1": np.array([-1, 1, 0, 0]),
    "balancing_at_K4": np.array([0, 0, -1, 1]),
    "interaction_difference_in_differences": np.array([1, -1, -1, 1]),
}


def read_csv(path):
    with Path(path).open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def load_predictions(rows, dirs, keys, keyfield):
    support = [i for i, row in enumerate(rows) if row["center"] in CENTERS]
    positions = {rows[i]["sample_id"]: j for j, i in enumerate(support)}
    expected = {rows[i]["sample_id"]: rows[i] for i in support}
    cube = np.full((len(SEEDS), len(keys), len(support)), -1, dtype=np.int8)
    for si, seed in enumerate(SEEDS):
        seen = set()
        for record in read_csv(dirs[seed] / "predictions.csv"):
            name, sample = record[keyfield], record["sample_id"]
            if name not in keys or sample not in positions:
                raise ValueError("Prediction contains unexpected method/sample")
            pair = (name, sample)
            if pair in seen:
                raise ValueError("Duplicate predictions")
            seen.add(pair)
            original = expected[sample]
            if (int(record["label"]) != original["label"] or record["center"] != original["center"]
                    or record["group_id"] != original["group_id"]):
                raise ValueError("Prediction cohort identity/label/group differs")
            cube[si, keys.index(name), positions[sample]] = int(record["prediction"])
        if len(seen) != len(support) * len(keys):
            raise ValueError("Incomplete paired predictions")
    if np.any(cube < 0):
        raise ValueError("Unfilled paired prediction cube")
    return cube, support


def paired_bootstrap(rows, support, predictions, attempts=2000, seed=2027):
    """The same whole-sequence multiplicities for every encoder/head/fit seed."""
    rng = np.random.default_rng(seed)
    y = np.array([rows[i]["label"] for i in support])
    centers = np.array([rows[i]["center"] for i in support])
    groups = np.array([rows[i]["group_id"] for i in support])
    values = np.zeros((attempts, predictions.shape[0], predictions.shape[1], len(CENTERS)))
    valid = np.ones(attempts, dtype=bool)
    invalid = {}
    for ci, center in enumerate(CENTERS):
        select = centers == center
        yy = y[select]
        names, inverse = np.unique(groups[select], return_inverse=True)
        draws = rng.multinomial(len(names), np.full(len(names), 1 / len(names)), size=attempts)
        positives = draws @ np.bincount(inverse, weights=yy == 1, minlength=len(names))
        negatives = draws @ np.bincount(inverse, weights=yy == 0, minlength=len(names))
        complete = (positives > 0) & (negatives > 0)
        valid &= complete
        invalid[center] = int((~complete).sum())
        for si in range(predictions.shape[0]):
            for mi in range(predictions.shape[1]):
                pred = predictions[si, mi, select]
                tp = draws @ np.bincount(inverse, weights=(yy == 1) & (pred == 1), minlength=len(names))
                tn = draws @ np.bincount(inverse, weights=(yy == 0) & (pred == 0), minlength=len(names))
                values[:, si, mi, ci] = .5 * (
                    np.divide(tp, positives, out=np.zeros_like(tp), where=positives > 0)
                    + np.divide(tn, negatives, out=np.zeros_like(tn), where=negatives > 0))
    if valid.sum() < 2:
        raise ValueError("Insufficient class-complete sequence bootstrap draws")
    values = values[valid].mean(axis=1)
    return values.mean(axis=2), values.min(axis=2), {
        "attempted": attempts, "accepted": int(valid.sum()), "seed": seed,
        "invalid_by_center_nonexclusive": invalid,
        "scope": "Whole sequence bootstrap within fixed centers; same resamples all encoders, heads, and seeds; conditional on both classes per center and fixed fitted models. Patient independence unverified; no population-level hospital claim."}


def main_metrics():
    records = []
    for encoder, root in ((ENCODERS[0], BASE / "results"), (ENCODERS[1], OUT / "main")):
        for seed in SEEDS:
            fits = json.loads((root / f"seed{seed}/metrics_by_center.json").read_text())
            if len(fits) != 36:
                raise ValueError("Expected36 final main evaluations per seed")
            for fit in fits:
                records.append({"encoder": encoder, "seed": seed, "center": fit["heldout_center"],
                                "method": fit["method"], "configuration": fit["configuration"],
                                "temperature": fit["temperature"], **fit["metrics"]})
    return records


def summarize(records, methods, bootstrap_mean, bootstrap_worst):
    summaries, center_records = [], []
    for ei, encoder in enumerate(ENCODERS):
        for mi, method in enumerate(methods):
            selected = [r for r in records if r["encoder"] == encoder and r["method"] == method]
            if len(selected) != len(CENTERS) * len(SEEDS):
                raise ValueError("Incomplete paired encoder/center/head-seed metric cohort")
            per_center = []
            for center in CENTERS:
                local = [r for r in selected if r["center"] == center]
                ba = [r["balanced_accuracy"] for r in local]
                per_center.append(float(np.mean(ba)))
                center_records.append({"encoder": encoder, "method": method, "center": center,
                    "balanced_accuracy": float(np.mean(ba)), "optimization_seed_sd": float(np.std(ba, ddof=1)),
                    "auroc": float(np.mean([r["auroc"] for r in local])),
                    "macro_f1": float(np.mean([r["macro_f1"] for r in local]))})
            index = ei * len(methods) + mi
            low, high = np.quantile(bootstrap_mean[:, index], [.025, .975])
            worst_low, worst_high = np.quantile(bootstrap_worst[:, index], [.025, .975])
            seed_means = [np.mean([r["balanced_accuracy"] for r in selected if r["seed"] == seed]) for seed in SEEDS]
            summaries.append({"encoder": encoder, "method": method,
                 "mean_center_balanced_accuracy": float(np.mean(per_center)),
                 "worst_center_balanced_accuracy": float(np.min(per_center)),
                 "mean_center_auroc": float(np.mean([r["auroc"] for r in selected])),
                 "mean_center_macro_f1": float(np.mean([r["macro_f1"] for r in selected])),
                 "optimization_seed_sd": float(np.std(seed_means, ddof=1)),
                 "conditional_ci_low": float(low), "conditional_ci_high": float(high),
                 "conditional_worst_ci_low": float(worst_low), "conditional_worst_ci_high": float(worst_high)})
    contrasts = []
    for mi, method in enumerate(methods):
        delta = bootstrap_mean[:, len(methods) + mi] - bootstrap_mean[:, mi]
        delta_worst = bootstrap_worst[:, len(methods) + mi] - bootstrap_worst[:, mi]
        before, after = summaries[mi], summaries[len(methods) + mi]
        low, high = np.quantile(delta, [.025, .975])
        wl, wh = np.quantile(delta_worst, [.025, .975])
        contrasts.append({"method": method,
              "DINOv2_minus_ResNet18_mean_BA": after["mean_center_balanced_accuracy"] - before["mean_center_balanced_accuracy"],
              "conditional_ci_low": float(low), "conditional_ci_high": float(high),
              "DINOv2_minus_ResNet18_worst_BA": after["worst_center_balanced_accuracy"] - before["worst_center_balanced_accuracy"],
              "conditional_worst_ci_low": float(wl), "conditional_worst_ci_high": float(wh)})
    return summaries, center_records, contrasts


def save_fig(fig, name):
    for suffix in ("png", "pdf"):
        fig.savefig(REPORT / f"{name}.{suffix}", dpi=240, bbox_inches="tight")
    plt.close(fig)


def component_effects(factorial, center_records, bootstrap):
    effects = []
    for ei, encoder in enumerate(ENCODERS):
        cells = [cell for cell, _, _ in CELLS]
        means = np.array([next(r for r in factorial if r["encoder"] == encoder and r["method"] == cell)
                          ["mean_center_balanced_accuracy"] for cell in cells])
        for name, weights in COMPONENT_CONTRASTS.items():
            values = bootstrap[:, ei * len(cells):(ei + 1) * len(cells)] @ weights
            low, high = np.quantile(values, [.025, .975])
            effects.append({"encoder": encoder, "contrast": name, "scope": "equal_center_mean",
                 "effect_balanced_accuracy": float(means @ weights),
                 "conditional_ci_low": float(low), "conditional_ci_high": float(high)})
            for center in CENTERS:
                local = np.array([next(r for r in center_records if r["encoder"] == encoder
                              and r["method"] == cell and r["center"] == center)["balanced_accuracy"] for cell in cells])
                effects.append({"encoder": encoder, "contrast": name, "scope": center,
                     "effect_balanced_accuracy": float(local @ weights),
                     "conditional_ci_low": None, "conditional_ci_high": None})
    return effects


def figures(summaries, center_metrics, factorial):
    palette = ("#267889", "#d17b27")
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 6), constrained_layout=True)
    y = np.arange(len(METHODS))
    for ei, encoder in enumerate(ENCODERS):
        values = [r for r in summaries if r["encoder"] == encoder]
        for ax, key in zip(axes, ("mean_center_balanced_accuracy", "worst_center_balanced_accuracy")):
            ax.barh(y + (ei - .5) * .33, [r[key] for r in values], height=.30, color=palette[ei], label=encoder)
    for ax, title in zip(axes, ("Equal center mean", "Worst center after averaging head seeds")):
        ax.set_yticks(y, [LABELS[m] for m in METHODS])
        ax.invert_yaxis()
        ax.set_xlim(0, 1)
        ax.set_xlabel("Balanced accuracy")
        ax.set_title(title)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="x", alpha=.2)
        ax.legend(frameon=False)
    fig.suptitle("Paired frozen encoders; reviewed PolypGen subset; 4 targets, 3 head seeds", fontsize=12)
    save_fig(fig, "paired_encoder_mean_worst")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), constrained_layout=True)
    for ei, encoder in enumerate(ENCODERS):
        for beta, style in ((0, "o-"), (1, "s--")):
            selected = [next(r for r in factorial if r["encoder"] == encoder and r["method"] == cell)
                        for cell, k, b in CELLS if b == beta]
            for ax, key in zip(axes, ("mean_center_balanced_accuracy", "worst_center_balanced_accuracy")):
                ax.plot([1, 4], [r[key] for r in selected], style, color=palette[ei], label=f"{encoder}, beta={beta}")
    for ax, title in zip(axes, ("Equal center mean", "Worst center")):
        ax.set_xticks([1, 4])
        ax.set_xlabel("Prototypes per class")
        ax.set_ylim(0, 1)
        ax.set_ylabel("Balanced accuracy")
        ax.set_title(title)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(alpha=.2)
        ax.legend(frameon=False, fontsize=8)
    fig.suptitle("Same fixed K x beta ablation across frozen encoders", fontsize=12)
    save_fig(fig, "paired_encoder_factorial")
    selected = ("cbmp", "linear_logistic", "single_centroid_group_weighted")
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharey=True, constrained_layout=True)
    for ax, method in zip(axes, selected):
        for ei, encoder in enumerate(ENCODERS):
            values = [next(r for r in center_metrics if r["method"] == method and r["encoder"] == encoder and r["center"] == c)
                      for c in CENTERS]
            ax.errorbar(np.arange(len(CENTERS)) + (ei - .5) * .12, [r["balanced_accuracy"] for r in values],
                        yerr=[r["optimization_seed_sd"] for r in values], fmt="o-", color=palette[ei], capsize=3, label=encoder)
        ax.set_xticks(np.arange(len(CENTERS)), CENTERS)
        ax.set_ylim(0, 1)
        ax.set_title(LABELS[method])
        ax.set_xlabel("Held-out center")
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(alpha=.2)
        ax.legend(frameon=False, fontsize=8)
    axes[0].set_ylabel("Balanced accuracy; bars: head-seed SD")
    fig.suptitle("Paired center-level replication; sequence subset; patient independence unverified", fontsize=11)
    save_fig(fig, "paired_encoder_per_center")


def tables_and_text(summaries, contrasts, factorial, bootstrap_meta, metadata):
    # Keep standard CVPR/Overleaf tabular commands and no extra packages.
    table = [r"\begin{table*}[t]", r"\centering\small", r"\begin{tabular}{lcccc}", r"\hline",
             r"& \multicolumn{2}{c}{ResNet18} & \multicolumn{2}{c}{DINOv2-B} \\",
             r"Classifier & Mean & Worst & Mean & Worst \\", r"\hline"]
    for method in METHODS:
        before, after = [next(r for r in summaries if r["method"] == method and r["encoder"] == enc) for enc in ENCODERS]
        table.append(f"{LABELS[method]} & {before['mean_center_balanced_accuracy']:.3f} & {before['worst_center_balanced_accuracy']:.3f} & {after['mean_center_balanced_accuracy']:.3f} & {after['worst_center_balanced_accuracy']:.3f} " + r"\\")
    table += [r"\hline\end{tabular}", r"\caption{Exploratory paired frozen-encoder comparison on the same partial reviewed PolypGen sequence subset: 5,444 frames, 37 sequences, 4 target centers and 3 head seeds. Source-only classifier selection and original partitions are unchanged. Scores are balanced accuracy. Patient independence is unverified.}",
              r"\label{tab:encoder_replication}", r"\end{table*}"]
    (REPORT / "table_encoder_replication.tex").write_text("\n".join(table) + "\n", encoding="utf-8")
    fac_tex = [r"\begin{table}[t]", r"\centering\small", r"\begin{tabular}{lccccc}", r"\hline",
               r"Encoder & $K$ & $\beta$ & Mean & Worst & AUROC \\", r"\hline"]
    for enc in ENCODERS:
        for cell, k, beta in CELLS:
            r = next(r for r in factorial if r["encoder"] == enc and r["method"] == cell)
            fac_tex.append(f"{enc} & {k} & {beta:.0f} & {r['mean_center_balanced_accuracy']:.3f} & {r['worst_center_balanced_accuracy']:.3f} & {r['mean_center_auroc']:.3f} " + r"\\")
    fac_tex += [r"\hline\end{tabular}", r"\caption{Exploratory fixed prototype-count and center-balancing ablation across frozen encoders. Group weighting, sequences, partitions and source-only calibration are held constant. Mean and worst scores are balanced accuracy after averaging 3 head seeds.}",
                r"\label{tab:encoder_factorial}", r"\end{table}"]
    (REPORT / "table_encoder_factorial.tex").write_text("\n".join(fac_tex) + "\n", encoding="utf-8")
    lines = ["# Experiment 4: paired frozen DINOv2 replication", "",
        "The DINOv2 base replication and fixed factorial ablation completed on exactly the original reviewed PolypGen subset: 5,444 frames, 37 sequences, held-out centers C2, C4, C5, C6, and fitting seeds 17, 29, 43. Source-only selection, candidate grids, calibration, and all frame/group/content partitions match the original ResNet18 experiment. Existing outputs remain unchanged.", "",
        "## Main matched classifier comparison", "",
        "| Classifier | ResNet18 mean BA | DINOv2 mean BA | ResNet18 worst BA | DINOv2 worst BA | Paired mean change (pp) |",
        "|---|---:|---:|---:|---:|---:|"]
    for method in METHODS:
        before, after = [next(r for r in summaries if r["method"] == method and r["encoder"] == enc) for enc in ENCODERS]
        contrast = next(r for r in contrasts if r["method"] == method)
        lines.append(f"| {LABELS[method]} | {before['mean_center_balanced_accuracy']:.1%} | {after['mean_center_balanced_accuracy']:.1%} | {before['worst_center_balanced_accuracy']:.1%} | {after['worst_center_balanced_accuracy']:.1%} | {100*contrast['DINOv2_minus_ResNet18_mean_BA']:+.2f} |")
    lines += ["", "![Paired mean/worst scores](paired_encoder_mean_worst.png)", "",
              "![Center-specific scores](paired_encoder_per_center.png)", "", "## Fixed component ablation", "",
              "| Encoder | K | beta | Mean BA | Worst BA | Mean AUROC |", "|---|---:|---:|---:|---:|---:|"]
    for enc in ENCODERS:
        for cell, k, beta in CELLS:
            r = next(r for r in factorial if r["encoder"] == enc and r["method"] == cell)
            lines.append(f"| {enc} | {k} | {beta:g} | {r['mean_center_balanced_accuracy']:.1%} | {r['worst_center_balanced_accuracy']:.1%} | {r['mean_center_auroc']:.1%} |")
    lines += ["", "![Fixed factorial across encoders](paired_encoder_factorial.png)", "", "## Interpretation and limits", "",
        "This comparison tests dependence on the pretrained representation. ResNet18 and DINOv2 differ in architecture, pretraining objective/data, and prescribed preprocessing; their difference does not isolate any one of these causes. No target results were used to change the predeclared DINOv2 checkpoint, preprocessing, classifier grids, or partitions.", "",
        "This PolypGen subset was already inspected during prior experiments. This is an exploratory paired replication rather than a new untouched test cohort. Unresolved sequence mappings remain excluded. Center C5 positives and C2 negatives each come from one sequence; sequence grouping does not establish independent patients.", "",
        f"Paired conditional sequence bootstrap: {bootstrap_meta['accepted']}/{bootstrap_meta['attempted']} class-complete draws accepted. Every encoder, head, and fitting seed uses the same sampled whole sequences. Intervals condition on the observed centers and fixed fitted heads; they do not measure patient-level independence or hospital-population uncertainty. See paired_encoder_contrasts.csv and bootstrap_metadata.json.", "",
        "108 DINOv2 final main fits and 48 fixed factorial final fits were evaluated. All 24 K=1 factorial prediction comparisons exactly matched the identical DINOv2 main centroid controls. Raw predictions, serialized heads, inner-source selection records, and exact feature/checkpoint hashes are saved beside this report.", "", "## Frozen encoder provenance", "",
        f"Official Hugging Face checkpoint: `facebook/dinov2-base`, revision `{metadata['revision']}`; weight SHA-256 `{metadata['checkpoint_sha256']}`. This checksum also matches the official Hugging Face pinned-revision LFS metadata; the public metadata and verification record are saved in ../checkpoint/. All model loads were offline and the architecture was required to resolve to Dinov2Model. Features are 768-dimensional final layer-normalized CLS vectors, L2 normalized. Evaluation mode, float32, no gradient updates. Processor configuration is pinned and hashed in ../experiment_config.json.", "",
        "[Official model](https://huggingface.co/facebook/dinov2-base) · [DINOv2 authors' implementation](https://github.com/facebookresearch/dinov2)", "",
        "Component effects for multiplicity, balancing and their interaction are in factorial_component_effects.csv, with paired conditional intervals for the equal-center mean and separate center-specific point effects. Positive or negative effects must be interpreted per encoder and center.", "",
        "Overleaf artifacts: table_encoder_replication.tex and table_encoder_factorial.tex; scientific figures are available as PNG and vector PDF. Manuscript source files were not modified."]
    (REPORT / "RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    if not (OUT / "FITS_COMPLETE").exists():
        raise FileNotFoundError("Complete all exp4 fits first")
    REPORT.mkdir(exist_ok=True)
    rows, digest, _ = load_cohort()
    before, support = load_predictions(rows, {s: BASE / "results" / f"seed{s}" for s in SEEDS}, METHODS, "method")
    after, support_after = load_predictions(rows, {s: OUT / "main" / f"seed{s}" for s in SEEDS}, METHODS, "method")
    if support != support_after:
        raise ValueError("Encoder supports differ")
    boot, boot_worst, boot_metadata = paired_bootstrap(rows, support, np.concatenate([before, after], axis=1))
    records = main_metrics()
    summaries, center_records, contrasts = summarize(records, METHODS, boot, boot_worst)
    write_csv(REPORT / "main_metrics_by_center_seed.csv", records)
    write_csv(REPORT / "summary_metrics.csv", summaries)
    write_csv(REPORT / "per_center_seedmean_metrics.csv", center_records)
    write_csv(REPORT / "paired_encoder_contrasts.csv", contrasts)
    cells = tuple(c[0] for c in CELLS)
    previous = ROOT.parent / "polypgen_followups_2026-10-17/exp2_factorial"
    # The old factorial stores all seeds in a single CSV.
    previous_predictions = read_csv(previous / "predictions.csv")
    positions = {rows[i]["sample_id"]: j for j, i in enumerate(support)}
    old_fac = np.full((len(SEEDS), len(cells), len(support)), -1, dtype=np.int8)
    new_fac = np.full_like(old_fac, -1)
    for source, cube in ((previous_predictions, old_fac), (read_csv(OUT / "factorial/predictions.csv"), new_fac)):
        seen = set()
        for row in source:
            key = (int(row["seed"]), row["cell"], row["sample_id"])
            if key in seen:
                raise ValueError("Duplicated factorial predictions")
            seen.add(key)
            cube[SEEDS.index(key[0]), cells.index(key[1]), positions[key[2]]] = int(row["prediction"])
        if len(seen) != np.prod(cube.shape) or np.any(cube < 0):
            raise ValueError("Incomplete factorial paired cohort")
    fb, fw, _ = paired_bootstrap(rows, support, np.concatenate([old_fac, new_fac], axis=1))
    factorial_records = []
    for encoder, path in ((ENCODERS[0], previous / "metrics_by_center_seed.csv"),
                          (ENCODERS[1], OUT / "factorial/metrics_by_center_seed.csv")):
        for r in read_csv(path):
            factorial_records.append({"encoder": encoder, "seed": int(r["seed"]), "center": r["center"],
                   "method": r["cell"], **{m: float(r[m]) for m in ("balanced_accuracy", "auroc", "macro_f1")}})
    factorial, factorial_centers, factorial_contrasts = summarize(factorial_records, cells, fb, fw)
    effects = component_effects(factorial, factorial_centers, fb)
    original_effects = read_csv(previous / "paired_effects.csv")
    for effect in effects:
        if effect["encoder"] == ENCODERS[0] and effect["scope"] == "equal_center_mean":
            reference = next(r for r in original_effects if r["contrast"] == effect["contrast"] and r["scope"] == effect["scope"])
            if not np.isclose(effect["effect_balanced_accuracy"], float(reference["effect_balanced_accuracy"]), atol=1e-12, rtol=0):
                raise ValueError("Paired report failed to reproduce original ResNet18 factorial effect")
    write_csv(REPORT / "factorial_encoder_summary.csv", factorial)
    write_csv(REPORT / "factorial_encoder_per_center.csv", factorial_centers)
    write_csv(REPORT / "factorial_encoder_contrasts.csv", factorial_contrasts)
    write_csv(REPORT / "factorial_component_effects.csv", effects)
    dump_json(REPORT / "bootstrap_metadata.json", boot_metadata)
    np.savez_compressed(REPORT / "paired_sequence_bootstrap.npz", mean=boot, worst=boot_worst,
                       factorial_mean=fb, factorial_worst=fw, methods=np.array(METHODS), cells=np.array(cells))
    metadata = json.loads((OUT / "experiment_config.json").read_text())["encoder"]
    external_check = json.loads((OUT / "checkpoint/external_checksum_verification.json").read_text())
    if (not external_check["match"] or external_check["revision"] != metadata["revision"]
            or external_check["expected_lfs_sha256"] != metadata["checkpoint_sha256"]
            or external_check["actual_cached_sha256"] != metadata["checkpoint_sha256"]):
        raise ValueError("Official checkpoint checksum verification differs from extraction metadata")
    figures(summaries, center_records, factorial)
    tables_and_text(summaries, contrasts, factorial, boot_metadata, metadata)
    dump_json(REPORT / "validation.json", {"exact_paired_manifest_digest": digest,
              "matched_main_predictions": int(np.prod(before.shape)),
              "matched_factorial_predictions": int(np.prod(old_fac.shape)),
              "all_prediction_pairs_complete": True, "bootstrap_metadata": boot_metadata,
              "main_final_evaluations": 108, "factorial_final_evaluations": 48,
              "official_pinned_revision_lfs_checksum_matches": True,
              "report_source_sha256": sha256(__file__)})
    (OUT / "COMPLETE").write_text("DINOv2 paired experiment, factorial, reports and validation complete.\n", encoding="utf-8")
    print(json.dumps({"main": summaries, "paired_changes": contrasts, "factorial": factorial}, indent=2), flush=True)


if __name__ == "__main__":
    main()
