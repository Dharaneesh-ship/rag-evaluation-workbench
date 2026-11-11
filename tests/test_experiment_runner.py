from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from rag.eval.experiment_runner import (
    load_experiment_config,
    _metric_record,
)


def write_config(tmp_path, **overrides):
    config_dir = tmp_path / "configs" / "experiments"
    config_dir.mkdir(parents=True, exist_ok=True)

    config = {
        "suite": "../../eval/dev.json",
        "db": "../../data/workbench.db",
        "index_dir": "../../data/default_indexes",
        "split": "dev",
        "k": 10,
        "experiments": [
            {"name": "baseline", "top_k": 10},
            {"name": "no_reranker", "top_k": 5},
        ],
    }
    config.update(overrides)

    path = config_dir / "test.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return path


def test_load_valid_config_and_resolve_paths(tmp_path):
    path = write_config(tmp_path)

    config = load_experiment_config(path)

    assert len(config["experiments"]) == 2
    assert config["split"] == "dev"
    assert Path(config["suite"]) == (tmp_path / "eval" / "dev.json").resolve()
    assert Path(config["db"]) == (tmp_path / "data" / "workbench.db").resolve()
    assert Path(config["index_dir"]) == (
        tmp_path / "data" / "default_indexes"
    ).resolve()


def test_reject_frozen_test_split(tmp_path):
    path = write_config(tmp_path, split="test")

    with pytest.raises(ValueError, match="only the dev split"):
        load_experiment_config(path)


def test_reject_duplicate_experiment_names(tmp_path):
    path = write_config(
        tmp_path,
        experiments=[
            {"name": "baseline", "top_k": 5},
            {"name": "baseline", "top_k": 10},
        ],
    )

    with pytest.raises(ValueError, match="unique"):
        load_experiment_config(path)


def test_reject_invalid_top_k(tmp_path):
    path = write_config(
        tmp_path,
        experiments=[{"name": "invalid", "top_k": 0}],
    )

    with pytest.raises(ValueError, match="Invalid top_k"):
        load_experiment_config(path)


def test_reject_empty_experiment_list(tmp_path):
    path = write_config(tmp_path, experiments=[])

    with pytest.raises(ValueError, match="non-empty list"):
        load_experiment_config(path)


@dataclass
class FakeQueryResult:
    query_id: str
    recall: float


def test_metric_record_contains_aggregate_and_per_query_metrics():
    report = SimpleNamespace(
        efficiency=SimpleNamespace(
            mean_query_latency_ms=35.5,
            mean_ingest_latency_ms=120.0,
            token_saved_est=50,
        ),
        num_queries=24,
        k=10,
        mean_recall_at_k=0.55,
        mrr=0.42,
        mean_ndcg_at_k=0.43,
        per_query=[FakeQueryResult("q1", 0.5)],
    )

    record = _metric_record(
        "bm25_baseline",
        {"top_k": 10},
        report,
    )

    assert record["status"] == "completed"
    assert record["num_queries"] == 24
    assert record["recall_at_k"] == 0.55
    assert record["mrr"] == 0.42
    assert record["ndcg_at_k"] == 0.43
    assert record["mean_query_latency_ms"] == 35.5
    assert record["per_query"][0]["query_id"] == "q1"
