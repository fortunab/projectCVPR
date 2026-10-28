"""Persist stdout and run the prespecified exp4 extraction, fits, and report."""
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "exp4_dinov2"
plan = json.loads((OUT / "extraction_resource_plan.json").read_text())
commands = [
    [sys.executable, "-u", str(ROOT / "exp4_dinov2.py"), "run",
     "--batch-size", str(plan["selected_batch_size"]), "--threads", "4",
     "--encoder-threads", str(plan["selected_threads"])],
    [sys.executable, "-u", str(ROOT / "exp4_report.py")],
]
env = os.environ.copy()
env.update(OMP_NUM_THREADS="4", MKL_NUM_THREADS="4", OPENBLAS_NUM_THREADS="4",
           NUMEXPR_NUM_THREADS="4", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
           MPLBACKEND="Agg", PYTHONUNBUFFERED="1")
for stage, command in zip(("processing", "report"), commands):
    path = OUT / "logs" / f"{stage}.log"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"Preserve previous log before restarting: {path}")
    with path.open("x", encoding="utf-8") as log:
        log.write("Command: " + subprocess.list2cmdline(command) + "\n")
        log.flush()
        process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, text=True, encoding="utf-8",
                                   errors="replace", bufsize=1)
        for line in process.stdout:
            print(line, end="", flush=True)
            log.write(line)
            log.flush()
        if process.wait():
            raise RuntimeError(f"Exp4 {stage} failed; inspect {path}")
print("[EXP4] COMPLETE processing and report", flush=True)
