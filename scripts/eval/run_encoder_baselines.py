#!/usr/bin/env python3
from __future__ import annotations

"""Runner script for encoder baselines.

Usage (from project root)::

    PYTHONPATH=src python scripts/eval/run_encoder_baselines.py \\
        --split dev --limit 50 --methods finbert,encoder

    PYTHONPATH=src python scripts/eval/run_encoder_baselines.py \\
        --split dev --methods finbert,encoder,strong_llm
"""

import argparse
import csv
import json
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from astra.config.task_schema import (
    ANNOTATION_DEV_ALIGNED_WORKSET_ANNOTATED_NORMALIZED_PATH,
    PREDICTIONS_DIR,
    resolve_reports_input_path,
)
from astra.evaluation.encoder_baselines import (
    EncoderSentimentBaseline,
    FinBERTBaseline,
    StrongLLMBaseline,
    run_all_baselines,
)

logger = logging.getLogger(__name__)

AVAILABLE_METHODS = ("finbert", "encoder", "strong_llm")


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def _load_reports(
    split: str | None,
    limit: int | None,
    input_path: Path | None,
) -> list[dict[str, str]]:
    """Load report rows from the experiment CSV or a custom path."""
    resolved = input_path or resolve_reports_input_path(
        split, prefer_experiment_split=True,
    )
    if not resolved.exists():
        # Fall back to non-experiment path.
        resolved = resolve_reports_input_path(split, prefer_experiment_split=False)
    logger.info("Loading reports from %s", resolved)
    with resolved.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if split is not None and input_path is None:
        rows = [r for r in rows if r.get("split") == split]
    if limit is not None:
        rows = rows[:limit]
    logger.info("Loaded %d reports (split=%s, limit=%s)", len(rows), split, limit)
    return rows


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    logger.info("Wrote %d rows to %s", len(rows), path)


def _load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


# ---------------------------------------------------------------------------
# Evaluation helper
# ---------------------------------------------------------------------------

def _try_evaluate(prediction_paths: dict[str, Path]) -> dict | None:
    """Run NLP evaluation if gold annotations exist. Returns metrics or None."""
    gold_path = ANNOTATION_DEV_ALIGNED_WORKSET_ANNOTATED_NORMALIZED_PATH
    if not gold_path.exists():
        logger.info(
            "Gold annotations not found at %s; skipping evaluation.", gold_path,
        )
        return None

    try:
        from astra.evaluation.nlp_metrics import evaluate_predictions

        result = evaluate_predictions(
            gold_path=gold_path,
            prediction_paths=prediction_paths,
        )
        return result
    except Exception as exc:  # noqa: BLE001
        logger.warning("Evaluation failed: %s", exc)
        return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run encoder/lexicon/LLM baselines on ASTRA reports.",
    )
    parser.add_argument(
        "--split",
        choices=("dev", "test", "train"),
        default="dev",
        help="Data split to evaluate (default: dev).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of reports to process.",
    )
    parser.add_argument(
        "--methods",
        type=str,
        default="finbert,encoder",
        help=(
            "Comma-separated list of baselines to run. "
            "Choices: finbert, encoder, strong_llm (default: finbert,encoder)."
        ),
    )
    parser.add_argument(
        "--encoder-model",
        type=str,
        default="bert-base-chinese",
        help="HuggingFace model name for the encoder baseline.",
    )
    parser.add_argument(
        "--input-path",
        type=Path,
        default=None,
        help="Override report CSV input path.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PREDICTIONS_DIR,
        help="Directory to write prediction JSONL files.",
    )
    parser.add_argument(
        "--evaluate",
        action="store_true",
        default=False,
        help="Run NLP evaluation against gold annotations after prediction.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=False,
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
    )

    # Parse methods.
    selected = [m.strip() for m in args.methods.split(",") if m.strip()]
    for m in selected:
        if m not in AVAILABLE_METHODS:
            parser.error(f"Unknown method: {m}. Choose from: {AVAILABLE_METHODS}")

    # Load reports.
    reports = _load_reports(args.split, args.limit, args.input_path)
    if not reports:
        logger.error("No reports loaded. Exiting.")
        sys.exit(1)

    # Optionally build LLM client.
    llm_client = None
    if "strong_llm" in selected:
        try:
            from astra.llm.client import ClaudeJSONClient, load_llm_settings

            config = load_llm_settings()
            llm_client = ClaudeJSONClient(config)
            if not llm_client.enabled:
                logger.warning(
                    "LLM client not enabled (missing API key?). "
                    "Skipping strong_llm baseline.",
                )
                selected = [m for m in selected if m != "strong_llm"]
                llm_client = None
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to initialise LLM client: %s", exc)
            selected = [m for m in selected if m != "strong_llm"]

    if not selected:
        logger.error("No valid methods to run. Exiting.")
        sys.exit(1)

    # Run baselines selectively or via the aggregator.
    prediction_output_paths: dict[str, Path] = {}

    if set(selected) <= {"finbert", "encoder"}:
        # Only non-LLM methods requested -- run via run_all_baselines.
        all_results = run_all_baselines(
            reports,
            llm_client=None,
            encoder_model=args.encoder_model,
        )
        method_name_map = {"finbert": "finbert_lexicon", "encoder": "encoder_sentiment"}
        for method_key in selected:
            method_name = method_name_map[method_key]
            preds = all_results.get(method_name, [])
            out_path = args.output_dir / f"{method_name}_predictions.jsonl"
            _write_jsonl(out_path, preds)
            prediction_output_paths[method_name] = out_path
    else:
        # LLM method requested -- still run all via aggregator.
        all_results = run_all_baselines(
            reports,
            llm_client=llm_client,
            encoder_model=args.encoder_model,
        )
        method_name_map = {
            "finbert": "finbert_lexicon",
            "encoder": "encoder_sentiment",
            "strong_llm": "strong_llm",
        }
        for method_key in selected:
            method_name = method_name_map[method_key]
            preds = all_results.get(method_name, [])
            out_path = args.output_dir / f"{method_name}_predictions.jsonl"
            _write_jsonl(out_path, preds)
            prediction_output_paths[method_name] = out_path

    # Evaluation.
    if args.evaluate and prediction_output_paths:
        logger.info("Running NLP evaluation...")
        eval_result = _try_evaluate(prediction_output_paths)
        if eval_result and "metrics" in eval_result:
            print(json.dumps(eval_result["metrics"], ensure_ascii=False, indent=2))
        else:
            logger.info("No evaluation metrics produced.")

    # Summary.
    print(f"\nDone. Wrote predictions for: {', '.join(prediction_output_paths)}")
    for name, path in prediction_output_paths.items():
        count = len(_load_jsonl(path))
        print(f"  {name}: {count} rows -> {path}")


if __name__ == "__main__":
    main()
