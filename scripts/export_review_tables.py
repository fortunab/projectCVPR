"""Export numerical paper tables from completed review reports, without fitting."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path

import numpy as np

from review_protocol import read_csv, sha256, write_json


HEADS = {
    "cbmp": "CBMP", "multi_prototype_unbalanced": "Unbalanced multi-prototype",
    "linear_logistic": "Linear logistic", "trainable_cosine": "Trained cosine",
    "linear_group_center_weighted": "Weighted linear", "cosine_group_center_weighted": "Weighted cosine",
    "single_centroid_group_weighted": "Group-weighted centroid",
    "single_centroid_center_balanced": "Center-balanced centroid",
    "single_centroid_image_weighted": "Image-weighted centroid",
}
TASK_LABELS = {"polypgen_resnet18": "PolypGen / ResNet18", "polypgen_dinov2": "PolypGen / DINOv2",
               "camelyon_resnet18": "Custom CAMELYON17-WILDS / ResNet18"}


def number(value, multiplier=100, signed=False):
    if value in ("", None):
        return "--"
    result = float(value) * multiplier
    return f"{result:+.2f}" if signed else f"{result:.2f}"


def table(environment, columns, header, rows, caption, label, size="small", spacing=4):
    return (f"\\begin{{{environment}}}[!htbp]\n\\centering\\{size}\n"
            f"\\setlength{{\\tabcolsep}}{{{spacing}pt}}\n"
            f"\\begin{{tabular}}{{{columns}}}\n\\hline\n" + header + " \\\\\n\\hline\n"
            + "\n".join(" & ".join(row) + " \\\\" for row in rows)
            + "\n\\hline\n\\end{tabular}\n" + f"\\caption{{{caption}}}\n\\label{{{label}}}\n\\end{{{environment}}}\n")


def figures(summary, intervals):
    polyp = [r"\begin{figure*}[!t]", r"\centering", r"\begin{tikzpicture}[font=\scriptsize,y=.83cm]"]
    learned = ("linear_logistic", "trainable_cosine", "linear_group_center_weighted", "cosine_group_center_weighted")
    for task, title, shift in (("polypgen_resnet18", "ResNet18", 0), ("polypgen_dinov2", "DINOv2", 8)):
        results = summary[task]["historical"]
        comparator = max(learned, key=lambda name: results[name]["mean_ba"])
        polyp.extend([f"\\begin{{scope}}[xshift={shift}cm]",
                      f"\\node[font=\\small\\bfseries] at (3.1,3.8) {{{title}: {HEADS[comparator].lower()}}};"])
        for value in range(30, 101, 10):
            x = .075 * (value - 30)
            polyp.extend([f"\\draw[black!15] ({x:.6f},-.15)--({x:.6f},3.2);",
                          f"\\node[below] at ({x:.6f},-.15) {{{value}}};"])
        for center, y in zip(("C2", "C4", "C5", "C6"), (3, 2, 1, 0)):
            a = (float(results["cbmp"]["centers"][center]["balanced_accuracy"]) * 100 - 30) * .075
            b = (float(results[comparator]["centers"][center]["balanced_accuracy"]) * 100 - 30) * .075
            polyp.extend([f"\\node[left] at (-.15,{y}) {{{center}}};",
                          f"\\draw[black!45] ({a:.6f},{y})--({b:.6f},{y});",
                          f"\\fill[blue!75] ({a:.6f},{y}) circle (2.2pt);",
                          f"\\fill[orange!85!black] ({b-.035:.6f},{y-.035:.3f}) rectangle ({b+.035:.6f},{y+.035:.3f});"])
        polyp.extend([r"\node at (2.65,-.65) {Balanced accuracy (\%)};", r"\end{scope}"])
    polyp.extend([r"\fill[blue!75] (4.3,4.45) circle (2.2pt);",
                  r"\node[right] at (4.4,4.45) {CBMP};",
                  r"\fill[orange!85!black] (6.45,4.415) rectangle (6.52,4.485);",
                  r"\node[right] at (6.6,4.45) {Mean-leading learned comparator};", r"\end{tikzpicture}",
                  r"\caption{Center-specific PolypGen BA averaged across ten fitting seeds using the original grids. Connectors pair point estimates, not confidence intervals. The displayed learned comparator has the highest observed target mean within its encoder.}",
                  r"\label{fig:polypgen_comparison}", r"\end{figure*}"])
    values = {r["center"]: r for r in intervals if r["task"] == "camelyon_resnet18" and r["protocol"] == "historical"
              and r["comparison"] == "cbmp_minus_linear_logistic" and r["center"] in ("2", "1")}
    lower = 2 * math.floor(min(float(r["ci_low"]) * 100 for r in values.values()) / 2)
    upper = max(2, 2 * math.ceil(max(float(r["ci_high"]) * 100 for r in values.values()) / 2))
    hist = [r"\begin{figure}[t]", r"\centering", r"\begin{tikzpicture}[font=\scriptsize,x=.43cm,y=.9cm]"]
    for tick in range(lower, upper + 1, 2):
        hist.extend([f"\\draw[black!15] ({tick},-.3)--({tick},1.5);",
                     f"\\node[below] at ({tick},-.3) {{{tick}}};"])
    hist.append(r"\draw[dashed,black!65] (0,-.3)--(0,1.5);")
    for center, y in (("2", 1), ("1", 0)):
        row = values[center]
        lo, hi, point = (float(row[key]) * 100 for key in ("ci_low", "ci_high", "difference"))
        hist.extend([f"\\node[anchor=east] at ({lower-.5},{y}) {{H{center}}};",
                     f"\\draw[blue!75,line width=1pt] ({lo:.6f},{y})--({hi:.6f},{y});",
                     f"\\draw[blue!75] ({lo:.6f},{y-.08})--({lo:.6f},{y+.08});",
                     f"\\draw[blue!75] ({hi:.6f},{y-.08})--({hi:.6f},{y+.08});",
                     f"\\fill[blue!75] ({point:.6f},{y}) circle (2.3pt);"])
    hist.extend([f"\\node[align=center] at ({(lower+upper)/2},-1.0) {{CBMP minus linear logistic BA\\\\(percentage points)}};",
                 r"\end{tikzpicture}",
                 r"\caption{Histology BA differences over ten fitting seeds, with paired patient-bootstrap 95\% intervals conditional on sampled patients and fixed heads. H2 is primary; H1 is secondary.}",
                 r"\label{fig:camelyon_comparison}", r"\end{figure}"])
    return {"figure_polypgen.tex": "\n".join(polyp) + "\n", "figure_camelyon.tex": "\n".join(hist) + "\n"}


def export(report_dir, output):
    provenance = json.loads((report_dir / "provenance.json").read_text())
    if len(provenance["seeds"]) != 10:
        raise ValueError("The paper exporter requires the complete ten-seed report")
    means = read_csv(report_dir / "image_metrics_seed_means.csv")
    if {r["task"] for r in means} != set(TASK_LABELS):
        raise ValueError("Expected the three approved cached tasks")
    if any(int(r["fitting_seeds"]) != 10 for r in means):
        raise ValueError("Incomplete per-cell fitting seeds")
    cells = {(r["task"], r["protocol"], r["center"], r["method"]): r for r in means}
    by_seed = defaultdict(lambda: defaultdict(list))
    for row in read_csv(report_dir / "image_metrics_by_seed.csv"):
        by_seed[row["task"], row["protocol"], row["method"]][row["seed"]].append(float(row["balanced_accuracy"]))
    summary = {}
    for task in TASK_LABELS:
        summary[task] = {}
        for protocol in ("historical", "matched4", "factorial"):
            methods = sorted({r["method"] for r in means if r["task"] == task and r["protocol"] == protocol})
            summary[task][protocol] = {}
            for method in methods:
                rows = [r for r in means if r["task"] == task and r["protocol"] == protocol and r["method"] == method]
                scores = [float(r["balanced_accuracy"]) for r in rows]
                expected_centers = {"1", "2"} if task.startswith("camelyon") else {"C2", "C4", "C5", "C6"}
                if {r["center"] for r in rows} != expected_centers:
                    raise ValueError("Paper target coverage differs from the approved protocol")
                repeats = by_seed[task, protocol, method]
                if len(repeats) != 10 or any(len(v) != len(expected_centers) for v in repeats.values()):
                    raise ValueError("Incomplete seed-specific center summaries")
                summary[task][protocol][method] = {"mean_ba": float(np.mean(scores)), "worst_ba": min(scores),
                                                   "mean_ba_seed_sd": float(np.std([np.mean(v) for v in repeats.values()], ddof=1)),
                                                   "centers": {r["center"]: r for r in rows}}
    polyp_rows = []
    for method, label in HEADS.items():
        row = [label]
        for task in ("polypgen_resnet18", "polypgen_dinov2"):
            values = summary[task]["historical"][method]
            row += [f"${number(values['mean_ba'])}\\pm{number(values['mean_ba_seed_sd'])}$", number(values["worst_ba"])]
        polyp_rows.append(row)
    artifacts = {"main_polypgen.tex": table("table*", "lrrrr",
        "Classifier & ResNet18 mean BA & ResNet18 worst BA & DINOv2 mean BA & DINOv2 worst BA",
        polyp_rows, "PolypGen BA (percent) with the original grids. Mean BA is the mean $\\pm$ fitting-seed SD of ten "
        "equal-center averages over C2/C4/C5/C6; worst BA is the minimum center-specific seed mean. "
        "The reviewed cohort contains 5,444 frames and 37 recorded sequences; patient linkage is unresolved.",
        "tab:polypgen_results")}
    hist_rows = []
    for method, label in HEADS.items():
        row = [label]
        for center in ("2", "1"):
            values = cells["camelyon_resnet18", "historical", center, method]
            row += [number(values["balanced_accuracy"]), number(values["auroc"])]
        hist_rows.append(row)
    hist_header = (r" & \multicolumn{2}{c}{H2} & \multicolumn{2}{c}{H1} \\" + "\n"
                   + r"Classifier & BA & AUROC & BA & AUROC")
    artifacts["main_camelyon.tex"] = table("table", "lrrrr", hist_header,
        hist_rows, "Custom CAMELYON17-WILDS subset (percent), averaged over ten fitting seeds with the original grids. "
        "H2/H1 are fixed primary/secondary targets with 9/8 sampled patients. This is not the official full WILDS protocol.",
        "tab:camelyon_results", "footnotesize", 3)
    contrasts = read_csv(report_dir / "factorial_ranking_contrasts.csv")
    factorial_rows = []
    for task, label in (("polypgen_resnet18", "ResNet18"), ("polypgen_dinov2", "DINOv2")):
        for center in ("C2", "C4", "C5", "C6"):
            selected = {r["metric"]: r for r in contrasts if r["task"] == task and r["center"] == center and r["K"] == "4"}
            factorial_rows.append([label, center, number(selected["same_auc"]["beta1_minus_beta0"], signed=True),
                                   number(selected["across_auc"]["beta1_minus_beta0"], signed=True)])
    artifacts["main_factorial_ranking.tex"] = table("table", "llrr", "Encoder & Target & $\\Delta$Same & $\\Delta$Across",
        factorial_rows, "AUROC change (points) from $\\beta=0$ to $\\beta=1$ at fixed $K=4$, averaged over ten fitting seeds. "
        "Paired conditional intervals and valid-draw counts are supplied in the report and supplement.", "tab:factorial_ranking")
    rank_means = read_csv(report_dir / "group_ranking_seed_means.csv")
    rank_rows = []
    for task, center, label, comparator in (
            ("polypgen_resnet18", "C5", "C5 / ResNet18", "linear_logistic"),
            ("polypgen_dinov2", "C5", "C5 / DINOv2", "trainable_cosine"),
            ("camelyon_resnet18", "2", "H2 / ResNet18", "linear_logistic")):
        for method in ("cbmp", comparator):
            row = next(r for r in rank_means if r["task"] == task and r["center"] == center
                       and r["protocol"] == "historical" and r["method"] == method)
            rank_rows.append([label if method == "cbmp" else "", "CBMP" if method == "cbmp"
                              else "Cosine" if method == "trainable_cosine" else "Linear",
                              *[number(row[key]) for key in ("pooled_auc", "same_auc", "across_auc", "equal_group_pair_auc")]])
    artifacts["main_group_ranking.tex"] = table("table", "llrrrr", "Site/encoder & Head & Pooled & Same & Across & Equal",
        rank_rows, "Ranking diagnostics (percent), averaged over ten fitting seeds. Same/Across distinguish recorded "
        "group identity; Equal assigns uniform weight to observed group pairs. C5 has one positive-bearing sequence; H2 is the primary histology target.",
        "tab:group_ranking")
    full = ["% Generated from the completed ten-seed report; no manually entered numerical results."]
    for task, label in TASK_LABELS.items():
        for protocol in ("historical", "matched4", "factorial"):
            centers = sorted({r["center"] for r in means if r["task"] == task and r["protocol"] == protocol})
            for center in centers:
                full.append(f"\\subsection*{{{label}; {protocol}; target {center}}}")
                full.extend([r"\begingroup\small\setlength{\tabcolsep}{3pt}", r"\begin{longtable}{lrrrrrr}",
                             r"\toprule Head & BA & BA seed SD & F1 & AUC & NLL & Brier \\", r"\midrule"])
                local = [r for r in means if r["task"] == task and r["protocol"] == protocol and r["center"] == center]
                for row in local:
                    head = HEADS.get(row["method"], row["method"].replace("_", r"\_"))
                    full.append(" & ".join([head, number(row["balanced_accuracy"]), number(row["balanced_accuracy_seed_sd"]),
                                           number(row["macro_f1"]), number(row["auroc"]),
                                           f"{float(row['negative_log_likelihood']):.4f}",
                                           f"{float(row['brier_score']):.4f}"]) + r" \\")
                full.extend([r"\bottomrule", r"\end{longtable}", r"\endgroup"])
    artifacts["full_ten_seed_tables.tex"] = "\n".join(full) + "\n"
    artifacts.update(figures(summary, read_csv(report_dir / "paired_ba_intervals.csv")))
    if output.exists():
        raise FileExistsError("Choose a new table directory")
    output.mkdir(parents=True)
    for name, source in artifacts.items():
        (output / name).write_text(source, encoding="utf-8")
    write_json(output / "paper_values.json", summary)
    write_json(output / "provenance.json", {"report_provenance_sha256": sha256(report_dir / "provenance.json"),
               "image_means_sha256": sha256(report_dir / "image_metrics_seed_means.csv"),
               "factorial_contrasts_sha256": sha256(report_dir / "factorial_ranking_contrasts.csv"),
               "paired_ba_intervals_sha256": sha256(report_dir / "paired_ba_intervals.csv"),
               "group_ranking_sha256": sha256(report_dir / "group_ranking_seed_means.csv"),
               "exporter_sha256": sha256(Path(__file__)), "seeds": provenance["seeds"]})
    print(f"Exported {len(artifacts)} paper table sources")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    export(args.report_dir.resolve(), args.output_dir.resolve())
