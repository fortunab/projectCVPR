"""Losslessly store repeated prediction blocks and restore exact original CSVs."""

from __future__ import annotations

import argparse
import csv
import itertools
import json
from pathlib import Path
import re

import numpy as np

from review_protocol import TASKS, sha256, write_json


SUPPORT = ("sample_id", "group_id", "label", "dataset_index")
VALUES = ("prediction", "probability_class_1", "margin")
CONSTANTS = ("task", "seed", "protocol", "center", "method", "source_sequence_threshold")


def save_object(objects, rows, fields):
    objects.mkdir(parents=True, exist_ok=True)
    arrays = {key: np.asarray([row[key].encode("utf-8") for row in rows], dtype="S") for key in fields}
    temporary = objects / "object.partial.npz"
    if temporary.exists():
        raise FileExistsError("An incomplete prediction-object write already exists")
    np.savez_compressed(temporary, **arrays)
    digest = sha256(temporary)
    target = objects / f"{digest}.npz"
    if target.exists():
        if sha256(target) != digest:
            raise ValueError("Prediction object integrity mismatch")
        temporary.unlink()
    else:
        temporary.replace(target)
    return digest


def pack_file(source, index_path, objects):
    blocks = []
    with source.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        fields = reader.fieldnames
        if set(fields) != set(SUPPORT + VALUES + CONSTANTS):
            raise ValueError("Unexpected prediction CSV columns")
        group_key = lambda row: tuple(row[key] for key in CONSTANTS)
        for key, group in itertools.groupby(reader, key=group_key):
            rows = list(group)
            blocks.append({"constants": dict(zip(CONSTANTS, key)), "rows": len(rows),
                           "support_object": save_object(objects, rows, SUPPORT),
                           "prediction_object": save_object(objects, rows, VALUES)})
    if not blocks:
        raise ValueError("Cannot archive empty predictions")
    write_json(index_path, {"version": 1, "columns": fields, "blocks": blocks,
                            "rows": sum(block["rows"] for block in blocks),
                            "original_predictions_sha256": sha256(source),
                            "representation": "exact original UTF-8 field strings; no numeric quantization"})


def restore_file(index_path, objects, output):
    index = json.loads(index_path.read_text())
    if index["version"] != 1 or set(index["columns"]) != set(SUPPORT + VALUES + CONSTANTS):
        raise ValueError("Unsupported prediction archive schema")
    if output.exists():
        if sha256(output) == index["original_predictions_sha256"]:
            return index["rows"]
        raise FileExistsError("Existing prediction CSV differs; choose a new output")
    cache = {}

    def read_object(digest, fields):
        if not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise ValueError("Invalid content-addressed prediction object")
        if digest not in cache:
            path = objects / f"{digest}.npz"
            if sha256(path) != digest:
                raise ValueError("Prediction object bytes changed")
            with np.load(path, allow_pickle=False) as arrays:
                if set(arrays.files) != set(fields):
                    raise ValueError("Prediction object fields differ")
                cache[digest] = {key: [value.decode("utf-8") for value in arrays[key].tolist()] for key in fields}
        return cache[digest]

    temporary = output.with_name(output.name + ".partial")
    count = 0
    created = False
    try:
        with temporary.open("x", encoding="utf-8", newline="") as stream:
            created = True
            writer = csv.DictWriter(stream, fieldnames=index["columns"])
            writer.writeheader()
            for block in index["blocks"]:
                support = read_object(block["support_object"], SUPPORT)
                values = read_object(block["prediction_object"], VALUES)
                if (set(block["constants"]) != set(CONSTANTS)
                        or any(len(column) != block["rows"] for column in (*support.values(), *values.values()))):
                    raise ValueError("Prediction block length or constant fields differ")
                for position in range(block["rows"]):
                    writer.writerow({**block["constants"], **{key: support[key][position] for key in SUPPORT},
                                     **{key: values[key][position] for key in VALUES}})
                count += block["rows"]
        if count != index["rows"] or sha256(temporary) != index["original_predictions_sha256"]:
            raise ValueError("Reconstructed prediction CSV does not match its original SHA256")
        temporary.replace(output)
    except BaseException:
        # Only this newly created file is removed; original result files are never overwritten.
        if created:
            temporary.unlink(missing_ok=True)
        raise
    return count


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    root = args.run_dir.resolve()
    plan = json.loads((root / "run_manifest.json").read_text())
    if not set(plan["tasks"]) <= set(TASKS):
        parser.error("This archive format targets the three reviewed cached tasks")
    total = 0
    for task in plan["tasks"]:
        for seed in plan["seeds"]:
            folder = root / task / f"seed{int(seed)}"
            if not folder.resolve().is_relative_to(root):
                raise ValueError("Prediction output leaves the selected results directory")
            total += restore_file(folder / "predictions.index.json", root / "prediction_objects", folder / "predictions.csv")
    print(f"Verified/restored {total} original prediction rows with exact CSV hashes")
