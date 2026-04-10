from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from astra.config.task_schema import (
    resolve_case_path,
    resolve_intermediate_path,
    resolve_prediction_paths,
)
from astra.pipelines.run_astra_inference import _build_case_rows, _write_jsonl
from astra.scoring.report_scorer import build_astra_prediction, route_prediction_labels


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main(outputs_root: Path) -> None:
    intermediate_path = resolve_intermediate_path(outputs_root)
    rows = _load_jsonl(intermediate_path)
    if not rows:
        raise FileNotFoundError(f"No intermediate rows found at {intermediate_path}")

    prediction_paths = resolve_prediction_paths(outputs_root)
    astra_predictions: list[dict[str, Any]] = []
    case_rows: list[dict[str, Any]] = []
    ablation_outputs: dict[str, list[dict[str, Any]]] = {
        "minus_retrieval": [],
        "minus_neutralizer": [],
        "minus_verifier": [],
        "minus_uncertainty_gate": [],
        "minus_analyst_prior": [],
    }

    for row in rows:
        report = {
            "report_id": row["report_id"],
            "report_date": row["report_date"],
            "stock_code": row["stock_code"],
            "split": row.get("split", ""),
            "title": row.get("title", ""),
            "summary": row.get("summary", ""),
        }
        rule_prediction = row.get("rule_prediction") or {}
        direct_prediction = row.get("direct_prediction") or {}
        cot_prediction = row.get("cot_prediction") or {}
        react_prediction = row.get("react_prediction") or {}
        routed_prediction = route_prediction_labels(
            direct_prediction=direct_prediction,
            cot_prediction=cot_prediction,
            react_prediction=react_prediction,
            rule_prediction=rule_prediction,
            title=report["title"],
            summary=report["summary"],
        )
        decomposition = dict(row.get("decomposition") or {})
        if row.get("history_context_used") is not None:
            decomposition["history_context_used"] = row.get("history_context_used")
        neutralization = row.get("neutralization") or {}
        verification = row.get("verification") or {}

        astra_prediction = build_astra_prediction(
            report=report,
            direct_prediction=routed_prediction,
            decomposition=decomposition,
            neutralization=neutralization,
            verifier_result=verification,
            method="astra_mvp",
        )
        astra_predictions.append(astra_prediction)
        case_rows.append(_build_case_rows(report, direct_prediction, astra_prediction))

        ablation_outputs["minus_retrieval"].append(
            build_astra_prediction(
                report=report,
                direct_prediction=routed_prediction,
                decomposition={**decomposition, "history_context_used": "无历史研报上下文"},
                neutralization=neutralization,
                verifier_result=verification,
                method="astra_minus_retrieval",
                use_retrieval=False,
            )
        )
        ablation_outputs["minus_neutralizer"].append(
            build_astra_prediction(
                report=report,
                direct_prediction=routed_prediction,
                decomposition=decomposition,
                neutralization={},
                verifier_result=verification,
                method="astra_minus_neutralizer",
                use_neutralizer=False,
            )
        )
        ablation_outputs["minus_verifier"].append(
            build_astra_prediction(
                report=report,
                direct_prediction=routed_prediction,
                decomposition=decomposition,
                neutralization=neutralization,
                verifier_result={},
                method="astra_minus_verifier",
                use_verifier=False,
            )
        )
        ablation_outputs["minus_uncertainty_gate"].append(
            build_astra_prediction(
                report=report,
                direct_prediction=routed_prediction,
                decomposition=decomposition,
                neutralization=neutralization,
                verifier_result=verification,
                method="astra_minus_uncertainty_gate",
                use_uncertainty_gate=False,
            )
        )
        ablation_outputs["minus_analyst_prior"].append(
            build_astra_prediction(
                report=report,
                direct_prediction=routed_prediction,
                decomposition=decomposition,
                neutralization=neutralization,
                verifier_result=verification,
                method="astra_minus_analyst_prior",
                use_analyst_prior=False,
            )
        )

    _write_jsonl(prediction_paths["astra_mvp"], astra_predictions)
    _write_jsonl(resolve_case_path(outputs_root), case_rows)
    for name, rows_for_name in ablation_outputs.items():
        _write_jsonl(prediction_paths[f"astra_{name}"], rows_for_name)

    print("[OK] Score rebuild completed.")
    print(json.dumps({
        "outputs_root": str(outputs_root),
        "rebuilt_rows": len(astra_predictions),
        "prediction_path": str(prediction_paths["astra_mvp"]),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--outputs-root", type=Path, required=True)
    args = parser.parse_args()
    main(outputs_root=args.outputs_root)
