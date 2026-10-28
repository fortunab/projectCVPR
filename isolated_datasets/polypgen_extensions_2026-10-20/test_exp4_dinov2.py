"""Meaningful frozen-feature/cohort safeguards for the encoder replication."""
import copy
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from exp4_dinov2 import (CENTERS, load_cohort, normalized_cls,
                          validate_feature_cache, validate_splits)


class StubEncoder(torch.nn.Module):
    def __init__(self, zero=False, mismatch=False):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(1.0), requires_grad=False)
        self.zero, self.mismatch = zero, mismatch
        self.eval()

    def forward(self, pixel_values):
        values = torch.ones((len(pixel_values), 768)) * (0 if self.zero else 1)
        hidden = values[:, None, :].clone()
        if self.mismatch:
            hidden[:, 0, 0] += 1
        return SimpleNamespace(pooler_output=values, last_hidden_state=hidden)


def test_exp4_official_cls_semantics_and_unit_norm():
    features = normalized_cls(StubEncoder(), torch.ones((2, 3, 224, 224)))
    assert features.shape == (2, 768)
    assert features.dtype == np.float32
    assert np.allclose(np.linalg.norm(features, axis=1), 1)


@pytest.mark.parametrize("mode", ("training", "gradient", "zero", "different_pooler"))
def test_exp4_rejects_unfrozen_or_wrong_representation(mode):
    model = StubEncoder(zero=mode == "zero", mismatch=mode == "different_pooler")
    if mode == "training":
        model.train()
    if mode == "gradient":
        model.weight.requires_grad_(True)
    with pytest.raises(ValueError):
        normalized_cls(model, torch.ones((1, 3, 224, 224)))


@pytest.mark.parametrize("change", ("metadata", "sample_order", "nonnormalized", "nonfinite"))
def test_exp4_cache_provenance_and_order_rejection(change):
    features = np.zeros((2, 768), dtype=np.float32)
    features[:, 0] = 1
    expected = {"checkpoint_sha256": "pinned", "manifest_digest": "paired"}
    metadata = expected.copy()
    ids, expected_ids = np.array(["a", "b"]), np.array(["a", "b"])
    if change == "metadata":
        metadata["checkpoint_sha256"] = "different"
    if change == "sample_order":
        ids = ids[::-1]
    if change == "nonnormalized":
        features *= 2
    if change == "nonfinite":
        features[0, 1] = np.nan
    with pytest.raises(ValueError):
        validate_feature_cache(features, metadata, expected, ids, expected_ids)


def test_exp4_exact_original_manifest_grids_and_partitions():
    rows, digest, splits = load_cohort()
    assert len(rows) == 5444
    assert len({r["group_id"] for r in rows}) == 37
    assert digest == "8a0b340e84aa28bab4d96011a38fc985d896291586b95afb4d67a2843cee5e44"
    assert tuple(s["heldout_center"] for s in splits[17] if s["status"] == "ready") == CENTERS


def test_exp4_rejects_target_in_source_validation():
    rows, _, splits = load_cohort()
    changed = copy.deepcopy(splits[17])
    outer = next(s for s in changed if s["status"] == "ready")
    outer["inner_folds"][0]["validation"].append(outer["outer_test"][0])
    with pytest.raises(ValueError, match="inner"):
        validate_splits(rows, changed)


def test_exp4_rejects_group_overlap_across_center_boundary():
    rows, _, splits = load_cohort()
    changed_rows = copy.deepcopy(rows)
    outer = next(s for s in splits[17] if s["status"] == "ready")
    changed_rows[outer["outer_test"][0]]["group_id"] = changed_rows[outer["outer_train"][0]]["group_id"]
    with pytest.raises(ValueError, match="outer"):
        validate_splits(changed_rows, splits[17])
