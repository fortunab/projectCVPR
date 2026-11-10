"""Audit the complete local CAMELYON17-WILDS v1.0 official split.

Requires an independently installed WILDS runtime and a complete local dataset.
No download or patient/patch capping is performed by this command.
"""

from __future__ import annotations

import argparse
import csv
import importlib.metadata
import inspect
from pathlib import Path

import numpy as np

from external_review_data import official_splits
from review_protocol import sha256, write_csv, write_json


def prepare(dataset, output):
    if str(dataset.version) != "1.0" or dataset.split_scheme != "official":
        raise ValueError("Use CAMELYON17-WILDS version 1.0 with split_scheme=official")
    data_dir = Path(dataset.data_dir)
    metadata_path = data_dir / "metadata.csv"
    with metadata_path.open(encoding="utf-8-sig", newline="") as stream:
        metadata = list(csv.DictReader(stream))
    if len(metadata) != len(dataset):
        raise ValueError("Official metadata length differs from dataset")
    subsets = {name: np.asarray(dataset.get_subset(name).indices, dtype=int).tolist()
               for name in ("train", "val", "test")}
    partitions = {}
    for name, indices in subsets.items():
        if indices != sorted(indices) or any(i in partitions for i in indices):
            raise ValueError("Unexpected official dataset order or overlapping partitions")
        partitions.update({index: name for index in indices})
    fields = dataset.metadata_fields
    hospital_column, slide_column = fields.index("hospital"), fields.index("slide")
    labels = dataset.y_array.numpy()
    official_metadata = dataset.metadata_array.numpy()
    rows = []
    for count, index in enumerate(sorted(partitions)):
        record = metadata[index]
        patient, node, x, y = (record[key] for key in ("patient", "node", "x_coord", "y_coord"))
        if not all(value.isdigit() for value in (patient, node, x, y)):
            raise ValueError("Official patient/node/coordinate fields must be non-negative numeric IDs")
        relative = f"patches/patient_{patient}_node_{node}/patch_patient_{patient}_node_{node}_x_{x}_y_{y}.png"
        image = data_dir / relative
        if (not image.is_file() or int(record["tumor"]) != int(labels[index])
                or int(record["center"]) != int(official_metadata[index, hospital_column])
                or int(record["slide"]) != int(official_metadata[index, slide_column])):
            raise ValueError(f"Official image or metadata mismatch at dataset index {index}")
        rows.append({"sample_id": f"wilds:camelyon17:v1.0:{index}", "dataset_index": index,
                     "path": relative, "sha256": sha256(image), "center": record["center"],
                     "group_id": f"patient:{patient}", "slide_id": record["slide"],
                     "label": int(record["tumor"]), "partition": partitions[index]})
        if count % 10000 == 0:
            print(f"Audited {count}/{len(partitions)} official images", flush=True)
    official_splits(rows)
    output.mkdir(parents=True)
    manifest = output / "manifest.csv"
    write_csv(manifest, rows)
    source = Path(inspect.getfile(type(dataset)))
    write_json(output / "official_audit.json", {
        "dataset": "camelyon17", "version": "1.0", "split_scheme": "official",
        "official_indices": subsets, "manifest_sha256": sha256(manifest),
        "metadata_sha256": sha256(metadata_path), "dataset_implementation_sha256": sha256(source),
        "wilds_version": importlib.metadata.version("wilds"),
        "n_rows": len(rows), "id_val_used": False, "patient_or_patch_cap": None,
        "image_verification": "complete selected official image byte hashes and metadata identity",
        "patient_labels_inferred": False,
    })
    print(f"Prepared complete official manifest: {manifest}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True, help="Parent of camelyon17_v1.0")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        parser.error("Choose a new output directory")
    from wilds import get_dataset
    official = get_dataset(dataset="camelyon17", root_dir=str(args.data_root), download=False,
                           version="1.0", split_scheme="official")
    prepare(official, args.output_dir)
