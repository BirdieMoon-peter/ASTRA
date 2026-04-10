from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import yaml

from astra.config.task_schema import ANNOTATION_MAIN_PATH, resolve_eval_paths, resolve_prediction_paths
from astra.evaluation.nlp_metrics import evaluate_predictions


def _now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def main(config_path: Path, outputs_root: Path) -> None:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    gold_path = ANNOTATION_MAIN_PATH if ANNOTATION_MAIN_PATH.exists() else ROOT / "data" / "annotations" / "stratreportzh_gold_workset.jsonl"
    prediction_paths = resolve_prediction_paths(outputs_root)
    eval_paths = resolve_eval_paths(outputs_root)
    result = evaluate_predictions(
        gold_path=gold_path,
        prediction_paths=prediction_paths,
        metrics_output_path=eval_paths["metrics"],
        calibration_output_path=eval_paths["calibration"],
        phenomena_output_path=eval_paths["phenomena"],
    )
    manifest = {
        "stage": "locked_test",
        "created_at": _now_iso(),
        "config_path": str(config_path),
        "outputs_root": str(outputs_root),
        "locked_split": config["selection"].get("locked_test_split", "test"),
        "gold_path": str(gold_path),
        "result_summary": result.get("metrics", {}),
    }
    manifest_path = outputs_root / "eval" / "locked_test_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"manifest_path": str(manifest_path), "status": "ok"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "locked_config.yaml")
    parser.add_argument("--outputs-root", type=Path, default=ROOT / "outputs")
    args = parser.parse_args()
    main(args.config, args.outputs_root)
