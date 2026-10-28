"""Create a strict, inspectable manifest for real PolypGen data."""

import argparse
import json
from pathlib import Path

from cbmp_data import build_manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_root")
    parser.add_argument("--manifest", default="results/cbmp/manifest.csv")
    parser.add_argument("--sequence-metadata", help="Reviewed CSV: sequence_id,center,source[,patient_id]")
    parser.add_argument("--sequence-only", action="store_true", help="Index only grouped video frames")
    parser.add_argument("--verified-sequences-only", action="store_true",
                        help="Explicit preliminary subset: include only sequences in reviewed metadata")
    parser.add_argument("--mask-threshold", type=int, default=127)
    parser.add_argument("--min-foreground-pixels", type=int, default=1)
    args = parser.parse_args()
    if not Path(args.data_root).expanduser().is_dir():
        parser.exit(2, f"PolypGen data directory is missing: {Path(args.data_root).expanduser().resolve()}\n"
                    "Download and extract the ZIP successfully before running the audit.\n"
                    "From the polypgen directory run:\n"
                    "  python download_synapse_polypgen.py --dest-dir data/PolypGen --extract --prompt-token\n")
    if args.sequence_metadata and not Path(args.sequence_metadata).expanduser().is_file():
        parser.exit(2, f"Sequence metadata CSV is missing: {args.sequence_metadata}\n"
                    "Run the audit without --sequence-metadata first to generate a template.\n"
                    "Fill its center assignments from verified release metadata before rerunning.\n")
    try:
        audit = build_manifest(args.data_root, args.manifest, sequence_metadata=args.sequence_metadata,
                               sequence_only=args.sequence_only, mask_threshold=args.mask_threshold,
                               min_foreground_pixels=args.min_foreground_pixels,
                               verified_sequences_only=args.verified_sequences_only)
    except (FileNotFoundError, ValueError) as error:
        parser.exit(2, f"Data audit stopped: {error}\nNo experiment was started.\n")
    print(json.dumps(audit, indent=2))
    if not audit["samples_indexed"]:
        raise SystemExit("No usable rows; inspect the audit and rejected CSV before training")


if __name__ == "__main__":
    main()

