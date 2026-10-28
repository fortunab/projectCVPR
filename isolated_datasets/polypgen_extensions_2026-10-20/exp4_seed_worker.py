"""Optional independent seed worker; same frozen cache, grids, splits, math."""
import argparse
import contextlib
import json
import sys
import time
import torch
from threadpoolctl import threadpool_limits

from exp4_dinov2 import OUT, ROOT, SEEDS, dump_json, extract, load_cohort, sha256
from cbmp_evaluation import candidate_methods, evaluate_loco


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, choices=SEEDS, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    setup = json.loads((ROOT / "setup_complete.json").read_text())
    if setup["protocol_sha256"] != sha256(ROOT / "PROTOCOL.md"):
        raise ValueError("Protocol changed")
    for filename, expected in setup["copied_code_sha256"].items():
        if sha256(ROOT / "code" / filename) != expected:
            raise ValueError("Frozen classifier core changed")
    rows, digest, splits = load_cohort()
    if not (OUT / "features/dinov2_base_cls.npz").exists():
        raise FileNotFoundError("Independent worker requires the completed common feature cache")
    features, metadata = extract(rows, digest, batch_size=16)
    destination = (OUT / "main" / f"seed{args.seed}").resolve()
    private = (OUT / "parallel_private" / f"seed{args.seed}").resolve()
    if not destination.is_relative_to(ROOT.resolve()) or not private.is_relative_to(ROOT.resolve()):
        raise ValueError("Both source and target must stay inside the new experiment folder")
    if destination.exists() or private.exists():
        raise FileExistsError("Independent worker requires untouched private and final seed directories")
    started = time.perf_counter()
    dump_json(private / "worker_provenance.json", {
        "seed": args.seed, "threads": 4, "resource_only_parallelization": True,
        "manifest_digest": digest, "checkpoint_sha256": metadata["checkpoint_sha256"],
        "feature_cache_sha256": sha256(OUT / "features/dinov2_base_cls.npz"),
        "protocol_sha256": sha256(ROOT / "PROTOCOL.md"),
        "worker_source_sha256": sha256(__file__),
        "classifier_code_sha256": setup["copied_code_sha256"],
        "private_output": str(private), "promoted_output": str(destination)})
    dump_json(private / "outer_splits.json", {"manifest_digest": digest, "splits": splits[args.seed]})
    with threadpool_limits(limits=4):
        records = evaluate_loco(features, rows, splits[args.seed], private,
                                seed=args.seed, methods=candidate_methods())
    if len(records) != 36:
        raise ValueError("Incomplete independent seed worker")
    dump_json(private / "worker_completion.json", {"seed": args.seed,
              "final_evaluations": len(records), "elapsed_seconds": time.perf_counter() - started})
    (private / "COMPLETE").write_text("All 36 main comparisons complete; independent seed worker.\n", encoding="utf-8")
    if destination.exists():
        raise FileExistsError("Final seed directory now exists; preserving completed private outputs")
    destination.parent.mkdir(parents=True, exist_ok=True)
    # Both absolute paths were checked within ROOT; rename is on this same
    # volume and requires an absent destination. Never merge/replace outputs.
    private.rename(destination)
    print(f"[WORKER] seed{args.seed} COMPLETE", flush=True)


class Tee:
    def __init__(self, console, log):
        self.console, self.log = console, log

    def write(self, value):
        self.console.write(value)
        self.log.write(value)
        self.flush()
        return len(value)

    def flush(self):
        self.console.flush()
        self.log.flush()


if __name__ == "__main__":
    seed_arg = sys.argv[sys.argv.index("--seed") + 1]
    log_path = OUT / "logs" / f"parallel_seed{seed_arg}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("x", encoding="utf-8") as log:
        tee = Tee(sys.stdout, log)
        with contextlib.redirect_stdout(tee), contextlib.redirect_stderr(tee):
            print("Command:", sys.executable, *sys.argv, flush=True)
            main()
