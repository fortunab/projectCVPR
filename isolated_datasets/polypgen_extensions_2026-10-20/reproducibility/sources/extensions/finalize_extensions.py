"""Assemble completed extension reports without editing earlier project files."""
from pathlib import Path
import csv
import hashlib
import json

ROOT = Path(__file__).resolve().parent
LABELS = {
    "cbmp": "CBMP",
    "multi_prototype_unbalanced": "Unbalanced multiprototype",
    "linear_logistic": "Linear logistic",
    "trainable_cosine": "Trained cosine",
    "linear_group_center_weighted": "Weighted linear",
    "cosine_group_center_weighted": "Weighted cosine",
    "single_centroid_group_weighted": "Group weighted centroid",
    "single_centroid_center_balanced": "Center balanced centroid",
    "single_centroid_image_weighted": "Image weighted centroid",
}


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def read_csv(path):
    with path.open(newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def main():
    for name in ("exp4_dinov2", "exp5_camelyon"):
        if not (ROOT / name / "COMPLETE").exists():
            raise RuntimeError(f"{name} is not complete; no combined completion claim allowed")
    preservation = json.loads((ROOT / "preservation_check.json").read_text())
    if not preservation["unchanged"]:
        raise RuntimeError("Previous project data/output preservation check failed")
    encoder = read_csv(ROOT / "exp4_dinov2/report/summary_metrics.csv")
    hospitals = read_csv(ROOT / "exp5_camelyon/report/summary_by_hospital.csv")
    cohort = json.loads((ROOT / "exp5_camelyon/audit/metadata_complete.json").read_text())
    evaluation = json.loads((ROOT / "exp5_camelyon/evaluation/complete.json").read_text())
    exp4 = json.loads((ROOT / "exp4_dinov2/report/validation.json").read_text())
    if exp4["main_final_evaluations"] != 108 or exp4["factorial_final_evaluations"] != 48:
        raise RuntimeError("Incomplete second-encoder evaluation counts")
    if evaluation["records"] != 54 or not evaluation["all27_heads_frozen_before_any_target_scoring"]:
        raise RuntimeError("Incomplete independent-dataset evaluation or missing freeze gate")
    if len(encoder) != 18 or len(hospitals) != 18:
        raise RuntimeError("Expected nine methods for both encoders and both target hospitals")
    enc = {(r["method"], r["encoder"]): r for r in encoder}
    hosp = {(r["method"], int(r["hospital"])): r for r in hospitals}
    lines = [
        "# Completed experiments 4–5", "",
        "Both approved extensions completed. All new data, cached features, trained classifier heads, source-only searches, predictions, scientific figures, CSV tables and Overleaf table fragments are in this folder. The existing manuscript was not edited.", "",
        "## 4. Frozen second encoder on the unchanged PolypGen cohort", "",
        "Frozen DINOv2 ViT-B/14 and the earlier ResNet18 use the same audited 5,444 frames, 37 sequences, four held-out centers and three fitting seeds. All nine classifier variants retain their original source-only tuning protocol. There are 108 new main evaluations and 48 new fixed factorial evaluations. Scores below are equally weighted center means after averaging fitting seeds; worst denotes the minimum of those center means.", "",
        "| Classifier | ResNet18 mean BA | DINOv2 mean BA | ResNet18 worst BA | DINOv2 worst BA |",
        "|---|---:|---:|---:|---:|",
    ]
    for method, label in LABELS.items():
        before, after = enc[method, "ResNet18"], enc[method, "DINOv2 base"]
        values = [before["mean_center_balanced_accuracy"], after["mean_center_balanced_accuracy"],
                  before["worst_center_balanced_accuracy"], after["worst_center_balanced_accuracy"]]
        lines.append(f"| {label} | " + " | ".join(f"{float(v):.2%}" for v in values) + " |")
    lines += [
        "", "[Detailed paired encoder report](exp4_dinov2/report/RESULTS.md) includes per-center scores, paired whole-sequence bootstrap intervals, fixed K/beta effects, saved predictions and checkpoint provenance.", "",
        "The encoder comparison is exploratory because the PolypGen cohort was previously inspected. It compares complete pretrained encoder packages, including their prescribed preprocessing. C2 negatives and C5 positives each occupy one sequence; patient independence has not been established. Sequence bootstrap intervals condition on class-complete draws and the observed centers and fitted models.", "",
        "## 5. Independent histology hospitals", "",
        f"The label-blind patient-capped CAMELYON17-WILDS subset contains {cohort['selected_patches']:,} patches from {cohort['selected_patients']} patients. Each eligible patient contributes at most 400 patches, ranked by a predeclared identity hash. A frozen ResNet18 ImageNet1K V1 encoder provides the common features. Original source-training hospitals 0, 3 and 4 alone select and fit all classifiers. These are new histology-trained classifier heads. All 27 final source-trained heads were saved and hashed before either target hospital was scored. Two target hospitals and three fitting seeds give 54 target records.", "",
        "Hospital 2 is the primary test. Hospital 1 is a prespecified secondary unseen-hospital evaluation. Balanced accuracy and AUROC below average three head-fitting seeds; patient bootstrap intervals are reported separately.", "",
        "| Classifier | Primary H2 BA | Primary H2 AUROC | Secondary H1 BA | Secondary H1 AUROC |",
        "|---|---:|---:|---:|---:|",
    ]
    for method, label in LABELS.items():
        first, second = hosp[method, 2], hosp[method, 1]
        values = [first["balanced_accuracy_mean"], first["auroc_mean"],
                  second["balanced_accuracy_mean"], second["auroc_mean"]]
        lines.append(f"| {label} | " + " | ".join(f"{float(v):.2%}" for v in values) + " |")
    lines += [
        "", "[Detailed independent-dataset report](exp5_camelyon/report/RESULTS.md) includes the patient audit, patient bootstrap intervals, paired classifier contrasts, selected settings, scientific figures and an Overleaf table.", "",
        "This is a custom frozen ImageNet-transfer subset using a pinned community redistribution, not a full official WILDS benchmark result. The author download was probed before mirror acquisition; all 21 mirror shards were verified against their pinned SHA-256 hashes. External ImageNet pretraining, patient caps and the custom source-only selection protocol must be disclosed in any paper. The two observed target hospitals do not identify uncertainty over a population of hospitals.", "",
        "## Reproducibility and preservation", "",
        "The frozen plan is [PROTOCOL.md](PROTOCOL.md). Separate experiment reports provide PNG and vector PDF figures plus standalone LaTeX table fragments; copy those artifacts into Overleaf only when the experiment interpretation is ready. No online Overleaf project was changed.", "",
        f"The final preservation check found no changes among {preservation['files_checked_size_mtime']:,} protected earlier files; {preservation['baseline_artifacts_hashed']:,} earlier experiment artifacts also match their saved SHA-256 values. See [preservation_check.json](preservation_check.json).", "",
        "The artifact manifest hashes scripts, fitted models, predictions, reports and provenance. Raw downloaded shards and selected image contents are covered by the acquisition and image-audit manifests rather than hashed a second time in that index.",
        "", "## Publication next steps", "",
        "[Publication roadmap in Romanian](PUBLICATION_ROADMAP.md) separates CVPR main's novelty/evidence burden from a medical CBMS contribution, explains ITiCSE's computing-education scope, identifies close prototype literature, and gives a provisional plan before the November 2026 CVPR deadlines. It proposes future work; additional experiments have not been started under that plan.",
    ]
    (ROOT / "RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    entries = []
    excluded = {"download", "shards", "images", "__pycache__", ".venv-repro"}
    for path in sorted(ROOT.rglob("*")):
        relative = path.relative_to(ROOT)
        if not path.is_file() or excluded.intersection(relative.parts):
            continue
        if path.name in {"artifacts_manifest.json", "EXTENSIONS_COMPLETE.json"} or path.suffix == ".part":
            continue
        entries.append({"path": relative.as_posix(), "bytes": path.stat().st_size, "sha256": digest(path)})
    manifest = ROOT / "artifacts_manifest.json"
    manifest.write_text(json.dumps({"files": entries, "raw_content_hashes_in_scoped_manifests": True}, indent=2), encoding="utf-8")
    marker = {"complete": True, "main_second_encoder_evaluations": 108,
              "second_encoder_factorial_evaluations": 48, "independent_hospital_evaluations": 54,
              "selected_camelyon_patches": cohort["selected_patches"], "selected_camelyon_patients": cohort["selected_patients"],
              "previous_project_files_unchanged": True, "indexed_artifacts": len(entries),
              "artifacts_manifest_sha256": digest(manifest), "combined_report_sha256": digest(ROOT / "RESULTS.md")}
    (ROOT / "EXTENSIONS_COMPLETE.json").write_text(json.dumps(marker, indent=2), encoding="utf-8")
    print(json.dumps(marker, indent=2))


if __name__ == "__main__":
    main()
