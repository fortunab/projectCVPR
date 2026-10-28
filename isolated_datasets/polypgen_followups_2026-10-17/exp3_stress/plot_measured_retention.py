"""Supplementary measured-budget figure; no fitting or source changes."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from common import output_path, write_csv


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    folder = ROOT / "exp3_stress"
    document = json.loads((folder / "conditions_plan.json").read_text())
    plan = document["conditions"]
    rows = []
    for condition in plan:
        if condition["status"] != "ready":
            continue
        rows.append({"condition_id": condition["condition_id"], "heldout_center": condition["heldout_center"],
                     "arm": condition["arm"], "subset_seed": condition["subset_seed"],
                     "requested_dominant_retention": condition["requested_dominant_fraction"],
                     "actual_dominant_retention": condition["actual_dominant_group_retention"],
                     "actual_total_group_retention": condition["actual_total_group_retention"],
                     "actual_total_frame_retention": condition["composition"]["total_frames"] / condition["baseline_composition"]["total_frames"],
                     "group_center_hhi": condition["composition"]["group_center_hhi"],
                     "baseline_group_center_hhi": condition["baseline_composition"]["group_center_hhi"]})
    write_csv(folder / "measured_source_retention.csv", rows)
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.5), constrained_layout=True)
    colors = {"C2": "#377eb8", "C4": "#4daf4a", "C5": "#ff7f00", "C6": "#984ea3"}
    for center in colors:
        for arm, marker in (("composition", "o"), ("matched_scarcity", "^")):
            local = [r for r in rows if r["heldout_center"] == center and r["arm"] == arm]
            for ax, field in zip(axes, ("actual_total_group_retention", "actual_total_frame_retention", "group_center_hhi")):
                ax.scatter([r["requested_dominant_retention"] for r in local], [r[field] for r in local],
                           color=colors[center], marker=marker, alpha=0.65, label=f"{center} {arm}" if ax is axes[0] else None)
    for ax, ylabel in zip(axes, ("Fraction of all source groups retained", "Fraction of all source frames retained", "Measured group-center HHI")):
        ax.set_xlabel("Requested dominant-center retention")
        ax.set_xticks([0.25, 0.5]); ax.set_ylabel(ylabel); ax.grid(alpha=0.2)
    axes[0].legend(fontsize=6, ncol=2)
    fig.suptitle("Actual retained budgets and source composition; group budgets match within pairs", fontsize=10)
    for extension in ("png", "pdf"):
        fig.savefig(output_path(folder / "figures" / f"measured_source_retention.{extension}"), dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
