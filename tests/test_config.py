"""Tests for config loading and the label scheme."""

from __future__ import annotations

from pathlib import Path

import pytest

from iotml.config import deep_merge, load_config, load_labels


def test_deep_merge_overrides_nested_values_only() -> None:
    base = {"a": {"x": 1, "y": 2}, "b": 3}
    merged = deep_merge(base, {"a": {"y": 20}})
    assert merged == {"a": {"x": 1, "y": 20}, "b": 3}
    assert base["a"]["y"] == 2  # input is not changed


def test_profile_overrides_default(tmp_path: Path) -> None:
    (tmp_path / "default.yaml").write_text("data:\n  cap: 10\n  seed: 1\n", encoding="utf-8")
    (tmp_path / "quick.yaml").write_text("data:\n  cap: 5\n", encoding="utf-8")
    config = load_config("quick", config_dir=tmp_path)
    assert config["data"] == {"cap": 5, "seed": 1}


def test_project_label_scheme_has_34_labels_in_8_categories() -> None:
    labels = load_labels()
    assert len(labels) == 34
    assert len(set(labels.values())) == 8
    assert labels["VulnerabilityScan"] == "Recon"
    assert labels["BenignTraffic"] == "Benign"


def test_duplicate_label_is_rejected(tmp_path: Path) -> None:
    (tmp_path / "labels.yaml").write_text(
        "categories:\n  A: [x]\n  B: [x]\n", encoding="utf-8"
    )
    with pytest.raises(ValueError):
        load_labels(config_dir=tmp_path)
