import argparse
from pathlib import Path


def main(argv=None):
    parser = argparse.ArgumentParser(description="CBMP research pipeline")
    sub = parser.add_subparsers(dest="command", required=True)
    audit = sub.add_parser("audit", help="Verify explicit image manifest; no label inference")
    audit.add_argument("manifest")
    audit.add_argument("output")
    extract = sub.add_parser("extract", help="Extract frozen timm features using a local checkpoint")
    extract.add_argument("manifest")
    extract.add_argument("audit")
    extract.add_argument("output")
    extract.add_argument("--model", required=True)
    extract.add_argument("--checkpoint", required=True)
    extract.add_argument("--device", default="cpu")
    extract.add_argument("--batch-size", type=int, default=16)
    run = sub.add_parser("run", help="Nested held-out-center benchmark")
    run.add_argument("cache")
    run.add_argument("output")
    run.add_argument("--methods", nargs="+", choices=["cbmp", "centroid", "balanced_centroid", "multiprototype", "image_centroid", "linear", "cosine"],
        default=["cbmp", "centroid", "balanced_centroid", "multiprototype", "image_centroid", "linear", "cosine"])
    run.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    run.add_argument("--bootstraps", type=int, default=500)
    run.add_argument("--allow-synthetic", action="store_true")
    export = sub.add_parser("export", help="Generate LaTeX table and PDF/SVG figures")
    export.add_argument("run_dir")
    demo = sub.add_parser("demo-features", help="Explicit synthetic pipeline-test fixture, not research data")
    demo.add_argument("output")
    args = parser.parse_args(argv)
    if args.command == "audit":
        from .data import audit_manifest
        print(audit_manifest(args.manifest, args.output))
    elif args.command == "extract":
        from .data import extract
        extract(args.manifest, args.audit, args.output, args.model, args.checkpoint, args.device, args.batch_size)
    elif args.command == "run":
        from .experiment import run
        if args.bootstraps < 1 or len(set(args.seeds)) != len(args.seeds):
            parser.error("Positive bootstrap count and unique seeds required")
        run(args.cache, args.output, args.methods, args.seeds, args.allow_synthetic, args.bootstraps)
    elif args.command == "export":
        from .report import export
        print(export(args.run_dir))
    elif args.command == "demo-features":
        import json
        import numpy as np
        from .models import normalize
        path = Path(args.output)
        if path.exists():
            raise FileExistsError(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        rng = np.random.default_rng(17)
        x, y, centers, groups, ids = [], [], [], [], []
        for s in range(4):
            for group in range(12):
                label = group % 2
                base = np.zeros(8)
                base[label] = 1.5
                base[2 + s] = .4
                for frame in range(2):
                    x.append(base + rng.normal(0, .4, 8))
                    y.append(label)
                    centers.append(f"demo_center_{s}")
                    groups.append(f"demo_patient_{s}_{group}")
                    ids.append(f"demo_{s}_{group}_{frame}")
        np.savez_compressed(path, x=normalize(x), y=y, centers=centers, groups=groups, sample_id=ids,
            provenance=json.dumps({"data_kind": "synthetic_pipeline_test", "seed": 17,
                "warning": "Not PolypGen; never use these outputs as paper evidence"}))


if __name__ == "__main__":
    main()
