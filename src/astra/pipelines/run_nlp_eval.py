from __future__ import annotations

import argparse
import json
from pathlib import Path

from astra.config.task_schema import (
    ANNOTATION_DEV_ALIGNED_WORKSET_ANNOTATED_NORMALIZED_PATH,
    ANNOTATION_MAIN_PATH,
    ANNOTATION_PILOT_PATH,
    MAIN_SAMPLE_PATH,
    PILOT_SAMPLE_PATH,
    resolve_eval_paths,
    resolve_prediction_paths,
)
from astra.evaluation.nlp_metrics import evaluate_predictions


def _load_report_ids(path: Path) -> set[str]:
    with path.open("r", encoding="utf-8") as handle:
        return {json.loads(line).get("report_id") for line in handle if line.strip()}


def _resolve_gold_path(prediction_paths: dict[str, Path]) -> Path:
    candidates = [
        ANNOTATION_DEV_ALIGNED_WORKSET_ANNOTATED_NORMALIZED_PATH,
        ANNOTATION_MAIN_PATH,
        ANNOTATION_PILOT_PATH,
        MAIN_SAMPLE_PATH,
        PILOT_SAMPLE_PATH,
    ]
    existing_candidates = [candidate for candidate in candidates if candidate.exists()]
    existing_predictions = [path for path in prediction_paths.values() if path.exists()]
    if not existing_candidates or not existing_predictions:
        return ANNOTATION_MAIN_PATH

    prediction_ids: set[str] = set()
    for path in existing_predictions:
        prediction_ids.update(_load_report_ids(path))

    best_candidate = existing_candidates[0]
    best_overlap = -1
    for candidate in existing_candidates:
        overlap = len(_load_report_ids(candidate) & prediction_ids)
        if overlap > best_overlap:
            best_candidate = candidate
            best_overlap = overlap
    return best_candidate


def main(gold_path: Path | None = None, outputs_root: Path | None = None) -> None:
    prediction_paths = resolve_prediction_paths(outputs_root)
    resolved_gold_path = gold_path or _resolve_gold_path(prediction_paths)
    eval_paths = resolve_eval_paths(outputs_root)
    result = evaluate_predictions(
        gold_path=resolved_gold_path,
        prediction_paths=prediction_paths,
        metrics_output_path=eval_paths["metrics"],
        calibration_output_path=eval_paths["calibration"],
        phenomena_output_path=eval_paths["phenomena"],
    )
    print("[OK] NLP evaluation completed.")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gold-path", type=Path, default=None)
    parser.add_argument("--outputs-root", type=Path, default=None)
    args = parser.parse_args()
    main(gold_path=args.gold_path, outputs_root=args.outputs_root)
