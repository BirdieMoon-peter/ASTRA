"""Strict split-locked evaluation with provenance tracking.

Usage:
    PYTHONPATH=src python scripts/eval/run_split_locked_evaluation.py --split test
    PYTHONPATH=src python scripts/eval/run_split_locked_evaluation.py --split dev
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from astra.config.task_schema import (
    ANNOTATIONS_DIR,
    EVAL_OUTPUT_DIR,
    PREDICTIONS_DIR,
    ABLATION_OUTPUT_DIR,
    ANNOTATION_DEV_ALIGNED_WORKSET_ANNOTATED_NORMALIZED_PATH,
    ANNOTATION_GOLD_WORKSET_PATH,
)
from astra.evaluation.nlp_metrics import (
    _load_jsonl,
    _macro_f1,
    build_calibration_bins,
    build_phenomena_metrics,
    continuous_score_deltas,
    evidence_span_prf,
    expected_calibration_error,
    label_distribution_diff,
)


LOCKFILE_DIR = ROOT / "artifacts" / "eval_locks"

PREDICTION_PATHS: dict[str, Path] = {
    "rule_baseline": PREDICTIONS_DIR / "rule_baseline_predictions.jsonl",
    "direct_llm": PREDICTIONS_DIR / "direct_baseline_predictions.jsonl",
    "cot_llm": PREDICTIONS_DIR / "cot_baseline_predictions.jsonl",
    "react_llm": PREDICTIONS_DIR / "react_baseline_predictions.jsonl",
    "astra_mvp": PREDICTIONS_DIR / "astra_predictions.jsonl",
}

ABLATION_PATHS: dict[str, Path] = {
    "astra_minus_retrieval": ABLATION_OUTPUT_DIR / "minus_retrieval.jsonl",
    "astra_minus_neutralizer": ABLATION_OUTPUT_DIR / "minus_neutralizer.jsonl",
    "astra_minus_verifier": ABLATION_OUTPUT_DIR / "minus_verifier.jsonl",
    "astra_minus_uncertainty_gate": ABLATION_OUTPUT_DIR / "minus_uncertainty_gate.jsonl",
    "astra_minus_analyst_prior": ABLATION_OUTPUT_DIR / "minus_analyst_prior.jsonl",
}


def _sha256(path: Path) -> str:
    if not path.exists():
        return ""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _git_hash() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
        )
        return result.stdout.strip() if result.returncode == 0 else "unknown"
    except Exception:
        return "unknown"


def _resolve_gold_path(split: str) -> Path:
    """Resolve the gold annotation file for a given split."""
    candidates = [
        ANNOTATIONS_DIR / f"stratreportzh_{split}_gold.jsonl",
        ANNOTATIONS_DIR / f"stratreportzh_{split}_annotated.jsonl",
    ]
    if split == "dev":
        candidates.append(ANNOTATION_DEV_ALIGNED_WORKSET_ANNOTATED_NORMALIZED_PATH)
    candidates.append(ANNOTATION_GOLD_WORKSET_PATH)

    for candidate in candidates:
        if candidate.exists():
            rows = _load_jsonl(candidate)
            complete = [r for r in rows if r.get("annotation") and r["annotation"].get("fundamental_sentiment")]
            if complete:
                return candidate
    return candidates[0]


def _filter_by_split(rows: list[dict[str, Any]], split: str) -> list[dict[str, Any]]:
    """Filter rows to a specific split. Tries 'split' field first, then falls through."""
    filtered = [r for r in rows if r.get("split") == split]
    if filtered:
        return filtered
    return rows


def _check_lockfile(split: str, force: bool) -> bool:
    """Check if a previous evaluation exists for this split."""
    lockfile = LOCKFILE_DIR / f"{split}_eval_lock.json"
    if lockfile.exists() and not force:
        lock_data = json.loads(lockfile.read_text(encoding="utf-8"))
        print(f"[BLOCKED] Previous {split} evaluation found:")
        print(f"  Timestamp: {lock_data.get('timestamp', 'unknown')}")
        print(f"  Git hash:  {lock_data.get('git_hash', 'unknown')}")
        print(f"Use --force to override.")
        return False
    return True


def _write_lockfile(split: str, manifest: dict[str, Any]) -> None:
    LOCKFILE_DIR.mkdir(parents=True, exist_ok=True)
    lockfile = LOCKFILE_DIR / f"{split}_eval_lock.json"
    lockfile.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


def evaluate_locked(
    split: str,
    *,
    force: bool = False,
    include_ablations: bool = True,
) -> dict[str, Any]:
    """Run strict split-locked evaluation with full provenance."""

    if not _check_lockfile(split, force):
        return {"status": "blocked_by_lockfile"}

    gold_path = _resolve_gold_path(split)
    if not gold_path.exists():
        return {"status": "missing_gold", "gold_path": str(gold_path)}

    gold_rows = _load_jsonl(gold_path)
    gold_rows = _filter_by_split(gold_rows, split)
    complete_gold = [r for r in gold_rows if r.get("annotation") and r["annotation"].get("fundamental_sentiment")]

    if not complete_gold:
        return {"status": "no_complete_annotations", "split": split, "gold_path": str(gold_path)}

    gold_ids = {r["report_id"] for r in complete_gold}
    print(f"[INFO] Evaluating split={split}, {len(complete_gold)} gold annotations from {gold_path.name}")

    # Evaluate main methods
    all_paths = dict(PREDICTION_PATHS)
    if include_ablations:
        all_paths.update(ABLATION_PATHS)

    metrics: dict[str, dict[str, Any]] = {}
    for method, path in all_paths.items():
        if not path.exists():
            continue
        pred_rows = _load_jsonl(path)
        pred_rows = [r for r in pred_rows if r.get("report_id") in gold_ids]
        if not pred_rows:
            continue

        span_metrics = evidence_span_prf(complete_gold, pred_rows)
        cal_bins = build_calibration_bins(pred_rows, complete_gold)
        phen_metrics = build_phenomena_metrics(complete_gold, pred_rows)

        metrics[method] = {
            "fundamental_sentiment_macro_f1": _macro_f1(complete_gold, pred_rows, "fundamental_sentiment"),
            "strategic_optimism_macro_f1": _macro_f1(complete_gold, pred_rows, "strategic_optimism"),
            "phenomenon_macro_f1": _macro_f1(complete_gold, pred_rows, "phenomenon"),
            **span_metrics,
            "ece": expected_calibration_error(cal_bins),
            "matched_count": len(pred_rows),
            "phenomena": phen_metrics,
        }

    # Ablation score deltas and label distribution diffs
    ablation_analysis: dict[str, Any] = {}
    astra_path = PREDICTION_PATHS.get("astra_mvp")
    if astra_path and astra_path.exists():
        full_preds = [r for r in _load_jsonl(astra_path) if r.get("report_id") in gold_ids]
        for abl_name, abl_path in ABLATION_PATHS.items():
            if not abl_path.exists():
                continue
            abl_preds = [r for r in _load_jsonl(abl_path) if r.get("report_id") in gold_ids]
            if not abl_preds:
                continue
            ablation_analysis[abl_name] = {
                "score_deltas": continuous_score_deltas(full_preds, abl_preds),
                "label_diffs": label_distribution_diff(full_preds, abl_preds),
            }

    # Build provenance manifest
    manifest = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "split": split,
        "git_hash": _git_hash(),
        "gold_path": str(gold_path),
        "gold_sha256": _sha256(gold_path),
        "gold_count": len(complete_gold),
        "prediction_sha256": {
            method: _sha256(path)
            for method, path in all_paths.items()
            if path.exists()
        },
        "methods_evaluated": list(metrics.keys()),
    }

    # Write outputs
    output_dir = EVAL_OUTPUT_DIR / f"locked_{split}"
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics_path = output_dir / "nlp_metrics.json"
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    manifest_path = output_dir / "provenance_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    if ablation_analysis:
        ablation_path = output_dir / "ablation_analysis.json"
        ablation_path.write_text(json.dumps(ablation_analysis, ensure_ascii=False, indent=2), encoding="utf-8")

    _write_lockfile(split, manifest)

    print(f"\n[OK] Locked {split} evaluation complete.")
    print(f"  Metrics: {metrics_path}")
    print(f"  Manifest: {manifest_path}")
    print(f"\nResults summary:")
    for method, m in metrics.items():
        print(f"  {method}:")
        print(f"    sentiment_F1={m['fundamental_sentiment_macro_f1']:.4f}  "
              f"optimism_F1={m['strategic_optimism_macro_f1']:.4f}  "
              f"phenomenon_F1={m['phenomenon_macro_f1']:.4f}  "
              f"evidence_F1={m['evidence_f1']:.4f}  "
              f"ECE={m['ece']:.4f}")

    if ablation_analysis:
        print(f"\nAblation label flip rates:")
        for abl_name, analysis in ablation_analysis.items():
            diffs = analysis["label_diffs"]
            rates = [f"{k}={v['flip_rate']:.2%}" for k, v in diffs.items()]
            print(f"  {abl_name}: {', '.join(rates)}")

    return {
        "status": "ok",
        "split": split,
        "metrics": metrics,
        "ablation_analysis": ablation_analysis,
        "manifest": manifest,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Strict split-locked NLP evaluation")
    parser.add_argument("--split", required=True, choices=["dev", "test"])
    parser.add_argument("--force", action="store_true", help="Override existing lockfile")
    parser.add_argument("--no-ablations", action="store_true")
    args = parser.parse_args()

    result = evaluate_locked(
        args.split,
        force=args.force,
        include_ablations=not args.no_ablations,
    )

    if result["status"] != "ok":
        print(f"[FAIL] {result['status']}")
        sys.exit(1)


if __name__ == "__main__":
    main()
