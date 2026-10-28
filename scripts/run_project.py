"""Safe wrappers for existing saved-prediction and cached-feature checks."""
import argparse
import datetime
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "isolated_datasets/polypgen_v3_copy_2026-10-17"
EXT = ROOT / "isolated_datasets/polypgen_extensions_2026-10-20"
BUNDLE = EXT / "reproducibility"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("step", choices=("verify", "reports", "level1", "level2", "all"))
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--seed", type=int, choices=(17, 29, 43), default=17)
    parser.add_argument("--encoder", choices=("resnet18", "dinov2"), default="resnet18")
    parser.add_argument("--allow-version-drift", action="store_true")
    args = parser.parse_args()
    out = (args.output_dir or ROOT / "reproduction_runs" / datetime.datetime.now().strftime("run_%Y%m%d_%H%M%S_%f")).resolve()
    if out.exists():
        raise FileExistsError("Use a new output directory; no earlier run is deleted")
    env = dict(os.environ, MPLBACKEND="Agg", PYTHONDONTWRITEBYTECODE="1")
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        env[name] = "4"

    def run(script, *arguments):
        command = [sys.executable, "-B", str(script), *map(str, arguments)]
        print("Running:", subprocess.list2cmdline(command), flush=True)
        subprocess.run(command, cwd=ROOT, env=env, check=True)

    check = ["--allow-version-drift"] if args.allow_version_drift else []
    run(ROOT / "scripts/check_environment.py", *check)
    out.mkdir(parents=True)
    env["MPLCONFIGDIR"] = str(out / "matplotlib")
    if args.step in ("verify", "reports", "all"):
        run(ROOT / "scripts/build_paper_results.py", "--output-dir", out / "verified_tables")
    if args.step in ("level1", "all"):
        for name, predictions, unit in (
            ("polypgen_resnet18", [(s, BASE / f"results/seed{s}/predictions.csv") for s in (17, 29, 43)], "group_id"),
            ("polypgen_dinov2", [(s, EXT / f"exp4_dinov2/main/seed{s}/predictions.csv") for s in (17, 29, 43)], "group_id"),
            ("camelyon", [(None, EXT / "exp5_camelyon/evaluation/predictions.csv")], "patient"),
        ):
            inputs = []
            for seed, path in predictions:
                inputs += ["--input", f"{seed}={path}" if seed is not None else str(path)]
            run(BUNDLE / "metrics_from_predictions.py", *inputs, "--unit", unit, "--output-dir", out / name)
    if args.step in ("level2", "all"):
        features = BASE / "features/resnet18_imagenet1k_v1.npz" if args.encoder == "resnet18" else EXT / "exp4_dinov2/features/dinov2_base_cls.npz"
        run(BUNDLE / "fit_cached_heads.py", "--manifest", BASE / "audit/manifest.csv", "--features", features,
            "--seed", args.seed, "--threads", 4, *check, "--output-dir", out / f"{args.encoder}_seed{args.seed}")
    print("Completed requested checks:", out)


if __name__ == "__main__":
    main()
