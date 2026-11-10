"""Generate supplementary numerical tables from the completed review reports."""

import argparse
from pathlib import Path

import numpy as np

from export_review_tables import HEADS
from review_protocol import read_csv, sha256, write_json


def percent(value, signed=False):
    if value in ("", None):
        return "--"
    return f"{float(value)*100:+.2f}" if signed else f"{float(value)*100:.2f}"


def interval(row):
    return "--" if row["ci_low"] == "" else f"[{percent(row['ci_low'])},{percent(row['ci_high'])}]"


def longtable(header, columns, rows):
    return (r"\begingroup\small\setlength{\tabcolsep}{4pt}" + "\n"
            + f"\\begin{{longtable}}{{{columns}}}\n\\toprule\n" + header + r" \\ \midrule" + "\n"
            + "\n".join(" & ".join(row) + r" \\" for row in rows)
            + "\n" + r"\bottomrule\end{longtable}\endgroup" + "\n")


def export(report, aggregates, stability, output):
    names = {"polypgen_resnet18": "ResNet18", "polypgen_dinov2": "DINOv2", "camelyon_resnet18": "Histology"}
    files = {}
    rows = read_csv(report / "factorial_ranking_contrasts.csv")
    files["review_factorial_intervals.tex"] = longtable(
        "Encoder & Site & $K$ & Component & $\\Delta$ & 95\\% interval & Valid",
        "llrlrrr", [[names[r["task"]], r["center"], r["K"], r["metric"].replace("_auc", "").title(),
                    percent(r["beta1_minus_beta0"], True), interval(r), r["valid_draws"]] for r in rows])
    rows = read_csv(report / "factorial_ba_interactions.csv")
    files["review_ba_interactions.tex"] = longtable(
        "Task & Site & Interaction & 95\\% interval & Valid", "llrrr",
        [[names[r["task"]], r["center"].replace("equal_center_mean", "Mean"),
          percent(r["interaction"], True), interval(r), r["valid_draws"]] for r in rows])
    rows = read_csv(report / "protocol_sensitivity_intervals.csv")
    files["review_budget_intervals.tex"] = longtable(
        "Task & Head & $\\Delta$ BA & 95\\% interval & Valid", "llrrr",
        [[names[r["task"]], HEADS[r["method"]], percent(r["matched4_minus_historical"], True),
          interval(r), r["valid_draws"]] for r in rows if r["center"] == "equal_center_mean"])
    sequence = read_csv(report / "sequence_metrics_seed_means.csv")
    patients = read_csv(report / "patient_macro_seed_means.csv")
    sections = []
    for protocol in ("historical", "matched4"):
        sections.append(f"\\subsection*{{Sequence-presence summaries: {protocol}}}")
        table = []
        for method, label in HEADS.items():
            row = [label]
            for task in ("polypgen_resnet18", "polypgen_dinov2"):
                local = [r for r in sequence if r["task"]==task and r["protocol"]==protocol
                         and r["method"]==method and r["center"] in ("C4","C5","C6")]
                if len(local) != 3:
                    raise ValueError("Missing class-complete sequence targets")
                row.extend(percent(np.mean([float(r[m]) for r in local])) for m in ("balanced_accuracy","auroc"))
            table.append(row)
        sections.append(longtable("Head & R18 BA & R18 AUC & DINOv2 BA & DINOv2 AUC", "lrrrr", table))
        sections.append(f"\\subsection*{{Patient-macro patch summaries: {protocol}}}")
        table = []
        for method, label in HEADS.items():
            row = [label]
            for center in ("2","1"):
                r=next(r for r in patients if r["protocol"]==protocol and r["center"]==center and r["method"]==method)
                row.extend(percent(r[m]) for m in ("balanced_accuracy","auroc"))
            table.append(row)
        sections.append(longtable("Head & H2 BA & H2 AUC & H1 BA & H1 AUC", "lrrrr", table))
    files["review_group_endpoints.tex"]="\n".join(sections)
    rows = read_csv(aggregates / "paired_aggregate_intervals.csv")
    selected=[]
    for r in rows:
        comparator="cbmp_minus_trainable_cosine" if r["task"]=="polypgen_dinov2" else "cbmp_minus_linear_logistic"
        if r["comparison"]!=comparator:
            continue
        selected.append([names[r["task"]], "Orig." if r["protocol"]=="historical" else "Four",
                         r["center"].replace("equal_center_mean","Mean"), "BA" if r["metric"]=="balanced_accuracy" else "AUC",
                         percent(r["difference"],True),interval(r),r["valid_draws"]])
    files["review_aggregate_intervals.tex"]=longtable(
        "Task & Grid & Site & Metric & $\\Delta$ & 95\\% interval & Valid", "llllrrr", selected)
    rows=read_csv(stability/"prototype_stability_summary.csv")
    selected=[]
    for r in rows:
        if r["center"]!="C5" or not (r["protocol"]=="historical" and r["method"]=="cbmp"
                                    or r["protocol"]=="factorial" and r["method"].startswith("K4_")):
            continue
        selected.append([names[r["task"]],r["method"].replace("_",r"\_"),r["class"],r["comparable_pairs"],
                         f"{float(r['mean_matched_cosine']):.4f}" if r["mean_matched_cosine"] else "--",
                         f"{float(r['min_pair_mean_cosine']):.4f}" if r["min_pair_mean_cosine"] else "--"])
    files["review_stability.tex"]=longtable("Encoder & Head & Class & Pairs & Mean cosine & Min. pair mean",
                                           "llrrrr", selected)
    if output.exists():
        raise FileExistsError("Choose a new supplementary table directory")
    output.mkdir(parents=True)
    for name,content in files.items():
        (output/name).write_text(content,encoding="utf-8")
    write_json(output/"provenance.json",{"report_sha256":sha256(report/"provenance.json"),
               "aggregate_sha256":sha256(aggregates/"provenance.json"),"stability_sha256":sha256(stability/"provenance.json"),
               "exporter_sha256":sha256(Path(__file__))})
    print(f"Exported {len(files)} supplementary table sources")


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-dir",type=Path,required=True)
    parser.add_argument("--aggregate-dir",type=Path,required=True)
    parser.add_argument("--stability-dir",type=Path,required=True)
    parser.add_argument("--output-dir",type=Path,required=True)
    args=parser.parse_args()
    export(args.report_dir.resolve(),args.aggregate_dir.resolve(),args.stability_dir.resolve(),args.output_dir.resolve())
