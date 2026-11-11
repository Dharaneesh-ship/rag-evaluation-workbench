"""Configuration-driven RAG experiment orchestration."""

import argparse
import csv
import json
import logging
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)

from rag.cli.eval import _load_suite, _run_queries
from rag.eval.split_guard import filter_by_split
from rag.pipelines.eval_pipeline import run_eval


def load_experiment_config(path: Path) -> dict[str, Any]:
    """Load and validate an experiment YAML configuration."""
    path = path.resolve()
    config = yaml.safe_load(path.read_text(encoding="utf-8")) or {}

    if not isinstance(config, dict):
        raise ValueError("Experiment configuration must be a YAML mapping.")

    if config.get("split", "dev") != "dev":
        raise ValueError(
            "Experiment runner currently permits only the dev split. "
            "The frozen test split is reserved for independent evaluation."
        )

    if not config.get("suite"):
        raise ValueError("Configuration must specify 'suite'.")

    experiments = config.get("experiments")
    if not isinstance(experiments, list) or not experiments:
        raise ValueError("'experiments' must be a non-empty list.")

    names = []
    for experiment in experiments:
        if not isinstance(experiment, dict):
            raise ValueError("Each experiment must be a YAML mapping.")
        name = experiment.get("name")
        if not isinstance(name, str) or not name.strip():
            raise ValueError("Every experiment requires a non-empty name.")
        names.append(name)
        if int(experiment.get("top_k", 10)) < 1:
            raise ValueError(f"Invalid top_k in experiment '{name}'.")

    if len(names) != len(set(names)):
        raise ValueError("Experiment names must be unique.")

    base = path.parent
    for key in ("suite", "db", "index_dir"):
        if config.get(key):
            value = Path(config[key])
            config[key] = str(
                value if value.is_absolute() else (base / value).resolve()
            )

    config["_config_path"] = str(path)
    return config


def _metric_record(name: str, settings: dict, report: Any) -> dict:
    """Flatten aggregate metrics into a comparable experiment record."""
    efficiency = report.efficiency

    return {
        "experiment": name,
        "status": "completed",
        "settings": settings,
        "num_queries": report.num_queries,
        "k": report.k,
        "recall_at_k": report.mean_recall_at_k,
        "mrr": report.mrr,
        "ndcg_at_k": report.mean_ndcg_at_k,
        "mean_query_latency_ms": efficiency.mean_query_latency_ms,
        "mean_ingest_latency_ms": efficiency.mean_ingest_latency_ms,
        "token_saved_est": efficiency.token_saved_est,
        "per_query": [asdict(item) for item in report.per_query],
    }


def run_experiments(
    config_path: Path,
    output_dir: Path | None = None,
) -> dict:
    """Run all configured experiments and export JSON and CSV results."""
    config = load_experiment_config(config_path)

    suite_path = Path(config["suite"])
    if not suite_path.is_file():
        raise FileNotFoundError(f"Evaluation suite not found: {suite_path}")

    db_path = Path(config.get("db", "data/default.db"))
    index_dir = Path(config.get("index_dir", "data/default_indexes"))

    if not db_path.is_file():
        raise FileNotFoundError(
            f"Database not found: {db_path}. Ingest documents first."
        )
    if not index_dir.is_dir():
        raise FileNotFoundError(f"Index directory not found: {index_dir}")

    entries = _load_suite(suite_path)
    entries = filter_by_split(entries, "dev")

    if not entries:
        raise ValueError("No development-split evaluation queries found.")

    k = int(config.get("k", 10))
    if k < 1:
        raise ValueError("Metric cutoff k must be >= 1.")

    records = []

    for experiment in config["experiments"]:
        name = experiment["name"]
        settings = {
            "top_k": int(experiment.get("top_k", 10)),
            "embedding_provider": experiment.get(
                "embedding_provider", "server_default"
            ),
            "reranker_provider": experiment.get(
                "reranker_provider", "server_default"
            ),
        }

        logger.info(f"\nRunning experiment: {name}")
        logger.info(f"Settings: {settings}")

        try:
            results = _run_queries(
                entries=entries,
                db_path=db_path,
                index_dir=index_dir,
                top_k=settings["top_k"],
                embedding_provider=settings["embedding_provider"],
                reranker_provider=settings["reranker_provider"],
            )

            report = run_eval(results, k=k)
            record = _metric_record(name, settings, report)

            logger.info(
                f"Recall@{k}: {report.mean_recall_at_k:.4f} | "
                f"MRR: {report.mrr:.4f} | "
                f"nDCG@{k}: {report.mean_ndcg_at_k:.4f}"
            )

        except Exception as exc:
            record = {
                "experiment": name,
                "status": "failed",
                "settings": settings,
                "error": str(exc),
            }
            logger.info(f"Experiment failed: {exc}")

        records.append(record)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    destination = Path(output_dir) if output_dir else (
        Path(config["_config_path"]).parent / "../../outputs/experiments"
    )
    destination.mkdir(parents=True, exist_ok=True)

    payload = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "config": config,
        "results": records,
    }

    json_path = destination / f"experiment_{timestamp}.json"
    csv_path = destination / f"experiment_{timestamp}.csv"

    json_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    columns = [
        "experiment",
        "status",
        "num_queries",
        "k",
        "recall_at_k",
        "mrr",
        "ndcg_at_k",
        "mean_query_latency_ms",
        "mean_ingest_latency_ms",
        "token_saved_est",
        "error",
    ]

    with csv_path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        for record in records:
            writer.writerow({
                key: record.get(key, "")
                for key in columns
            })

    completed = sum(r["status"] == "completed" for r in records)

    logger.info("\nExperiment summary")
    logger.info(f"Completed: {completed}/{len(records)}")
    logger.info(f"JSON: {json_path}")
    logger.info(f"CSV:  {csv_path}")

    if completed == 0:
        raise RuntimeError("All experiments failed. Check the error details.")

    return payload


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run configuration-driven RAG experiments."
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to experiment YAML configuration.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Optional result output directory.",
    )
    args = parser.parse_args()

    try:
        run_experiments(args.config, args.output_dir)
        return 0
    except Exception as exc:
        parser.exit(1, f"Experiment runner error: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
