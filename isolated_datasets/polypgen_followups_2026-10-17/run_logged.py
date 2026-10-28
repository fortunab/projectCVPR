"""Run one isolated processing script with a preserved UTF-8 console log."""
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if len(sys.argv) < 2:
        raise ValueError("Provide the processing script filename")
    script = (ROOT / sys.argv[1]).resolve()
    if not script.is_relative_to(ROOT) or not script.is_file():
        raise ValueError("Processing script must be inside the isolated follow-up folder")
    log = ROOT / "logs" / (script.stem + ".log")
    log.parent.mkdir(exist_ok=True)
    if log.exists():
        raise FileExistsError(f"Refusing to overwrite previous log: {log}")
    env = os.environ.copy()
    env.update(OMP_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2", MKL_NUM_THREADS="2", NUMEXPR_NUM_THREADS="2", PYTHONUTF8="1")
    command = [sys.executable, "-u", str(script), *sys.argv[2:]]
    with log.open("w", encoding="utf-8") as stream:
        stream.write(subprocess.list2cmdline(command) + "\n")
        stream.flush()
        child = subprocess.Popen(command, cwd=ROOT, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                 text=True, encoding="utf-8", errors="replace", bufsize=1)
        for line in child.stdout:
            stream.write(line)
            stream.flush()
            print(line, end="", flush=True)
        raise SystemExit(child.wait())


if __name__ == "__main__":
    main()
