"""Run the 33 archived classifier/manifest tests against frozen reference code."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import tempfile
import unittest

from review_protocol import BASE, ROOT, sha256, write_json


def run(output):
    if output.exists():
        raise FileExistsError("Choose a new test-report directory")
    output.mkdir(parents=True)
    temporary = (ROOT / "tmp/reference-tests").resolve()
    if not temporary.is_relative_to(ROOT):
        raise ValueError("Fixture directory must remain in this checkout")
    temporary.mkdir(parents=True, exist_ok=True)
    tempfile.tempdir = str(temporary)
    os.environ["TMP"] = os.environ["TEMP"] = str(temporary)
    names = ("test_cbmp_weighted_heads.py", "test_cbmp_evaluation.py", "test_cbmp_manifest.py")
    suite = unittest.TestSuite()
    sources = {}
    for name in names:
        path = BASE / "code/tests" / name
        sources[name] = sha256(path)
        suite.addTests(unittest.defaultTestLoader.discover(str(path.parent), pattern=name))
    log = output / "reference_tests.txt"
    with log.open("w", encoding="utf-8") as stream:
        result = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
    write_json(output / "verification.json", {
        "tests_run": result.testsRun, "failures": len(result.failures), "errors": len(result.errors),
        "successful": result.wasSuccessful(), "test_sources": sources, "log_sha256": sha256(log),
        "scope": "archived regression fixtures importing frozen reference classifier modules",
    })
    print(f"Reference tests: {result.testsRun}; failures={len(result.failures)}, errors={len(result.errors)}")
    if not result.wasSuccessful():
        raise RuntimeError(f"Reference regression failed; inspect {log}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    run(args.output_dir.resolve())
