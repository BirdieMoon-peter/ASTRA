from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from astra.config.prompts import DIRECT_BASELINE_SYSTEM_PROMPT, DIRECT_BASELINE_USER_PROMPT
from astra.config.task_schema import (
    ABLATION_OUTPUT_DIR,
    ASTRA_PREDICTIONS_PATH,
    CASE_STUDIES_PATH,
    CLEAN_REPORTS_PATH,
    COT_BASELINE_PATH,
    DIRECT_BASELINE_PATH,
    REACT_BASELINE_PATH,
    RULE_BASELINE_PATH,
    resolve_case_path,
    resolve_intermediate_path,
    resolve_prediction_paths,
    resolve_reports_input_path,
    resolve_snapshot_manifest_path,
    resolve_snapshot_root,
)
from astra.decomposition.decomposer import Decomposer
from astra.evaluation.baselines import rule_based_prediction
from astra.llm.client import ClaudeClientConfig, ClaudeJSONClient, load_llm_settings
from astra.neutralization.counterfactual_neutralizer import CounterfactualNeutralizer
from astra.retrieval.history_retriever import HistoryRetriever
from astra.scoring.report_scorer import build_astra_prediction, refine_labels_from_pipeline, route_prediction_labels
from astra.verification.verifier import Verifier


def _safe_llm_prediction(
    llm_client: ClaudeJSONClient,
    title: str,
    summary: str,
    *,
    style: str,
) -> dict[str, Any]:
    try:
        return _direct_llm_prediction(llm_client, title, summary, style=style)
    except Exception:
        fallback = rule_based_prediction(title, summary)
        return {
            "fundamental_sentiment": fallback.get("fundamental_sentiment", "neutral"),
            "strategic_optimism": fallback.get("strategic_optimism", "balanced"),
            "phenomenon": fallback.get("phenomenon", "none"),
            "uncertainty": 0.95,
            "evidence_spans": fallback.get("evidence_spans", []),
            "reasoning_summary": f"[{style} fallback] rule-based fallback after LLM failure",
        }


def _safe_decomposition(decomposer: Decomposer, *, title: str, summary: str, history_context: str) -> dict[str, Any]:
    try:
        return decomposer.run(title=title, summary=summary, history_context=history_context)
    except Exception:
        return {
            "factual_claims": [],
            "directional_cues": [],
            "hedge_cues": [],
            "optimistic_rhetoric": [],
            "risk_cues": [],
            "evidence_spans": [],
            "history_context_used": history_context,
            "fallback": "decomposer_error",
        }


def _safe_neutralization(
    neutralizer: CounterfactualNeutralizer,
    *,
    title: str,
    summary: str,
    decomposition: dict[str, Any],
) -> dict[str, Any]:
    try:
        return neutralizer.run(title=title, summary=summary, decomposition=decomposition)
    except Exception:
        return {
            "neutralized_text": summary,
            "removed_rhetoric": [],
            "preserved_facts": [],
            "fallback": "neutralizer_error",
        }


def _safe_verification(
    verifier: Verifier,
    *,
    title: str,
    summary: str,
    neutralized_text: str,
) -> dict[str, Any]:
    try:
        return verifier.run(title=title, summary=summary, neutralized_text=neutralized_text)
    except Exception:
        return {
            "numbers_preserved": True,
            "entities_preserved": True,
            "no_new_facts": True,
            "factual_consistency": 0.5,
            "verdict": "pass",
            "issues": ["verifier_error_fallback"],
        }


def _load_rows(
    limit: int | None = None,
    split: str | None = None,
    *,
    input_path: Path | None = None,
    prefer_experiment_split: bool = False,
) -> list[dict[str, str]]:
    resolved_input_path = input_path or resolve_reports_input_path(split, prefer_experiment_split=prefer_experiment_split)
    with resolved_input_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if split is not None and input_path is None and not prefer_experiment_split:
        rows = [row for row in rows if row["split"] == split]
    if limit is not None:
        rows = rows[:limit]
    return rows


def _direct_llm_prediction(llm_client: ClaudeJSONClient, title: str, summary: str, *, style: str = "direct") -> dict[str, Any]:
    prompt_suffix = {
        "direct": "当前模式：direct。请做保守判断，优先稳定 fundamental_sentiment；只有在证据非常清晰时才输出非 none 的 phenomenon。请直接输出最终 JSON。",
        "cot": "当前模式：cot。请在内部按以下顺序完成判断：先抽取基本面事实，再单独判断表述层面的乐观程度，重点检查是否存在“承认 downside 但被包装弱化”“标题强于正文”“前景结论强于事实支持”等现象。最终只输出 JSON。",
        "react": "当前模式：react。请在内部依次执行四步：1）列出支持基本面判断的事实；2）列出体现修辞与预期引导的词句；3）比较标题与摘要是否存在强度或方向落差；4）基于以上信息给出最终标签。最终只输出 JSON。",
    }[style]
    prompt = DIRECT_BASELINE_USER_PROMPT.format(title=title, summary=summary) + "\n\n" + prompt_suffix
    return llm_client.create_json(system=DIRECT_BASELINE_SYSTEM_PROMPT, user_prompt=prompt)


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _build_case_rows(
    report: dict[str, Any],
    direct_prediction: dict[str, Any],
    astra_prediction: dict[str, Any],
) -> dict[str, Any]:
    return {
        "report_id": report["report_id"],
        "report_date": report["report_date"],
        "stock_code": report["stock_code"],
        "title": report.get("title", ""),
        "summary": report.get("summary", ""),
        "direct_prediction": direct_prediction,
        "astra_prediction": astra_prediction,
        "neutralized_text": astra_prediction.get("neutralized_text", ""),
        "gap": astra_prediction.get("strategic_optimism_gap", 0.0),
        "uncertainty": astra_prediction.get("uncertainty", 0.0),
        "verifier_verdict": (astra_prediction.get("verification") or {}).get("verdict", ""),
    }


def _build_intermediate_row(
    report: dict[str, Any],
    *,
    split: str,
    llm_provider: str,
    llm_model: str,
    rule_prediction: dict[str, Any],
    direct_prediction: dict[str, Any],
    cot_prediction: dict[str, Any],
    react_prediction: dict[str, Any],
    routed_prediction: dict[str, Any],
    history_context_used: str,
    decomposition: dict[str, Any],
    neutralization: dict[str, Any],
    verification: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "report_id": report["report_id"],
        "report_date": report["report_date"],
        "stock_code": report["stock_code"],
        "split": split,
        "title": report.get("title", ""),
        "summary": report.get("summary", ""),
        "llm_provider": llm_provider,
        "llm_model": llm_model,
        "rule_prediction": rule_prediction,
        "direct_prediction": direct_prediction,
        "cot_prediction": cot_prediction,
        "react_prediction": react_prediction,
        "routed_prediction": routed_prediction,
        "history_context_used": history_context_used,
        "decomposition": decomposition,
        "neutralization": neutralization,
        "verification": verification,
    }


def _write_snapshot_manifest(
    *,
    outputs_root: Path,
    snapshot_id: str,
    split: str,
    limit: int,
    row_count: int,
    llm_enabled: bool,
    provider: str,
    model: str,
    source_reports_path: Path,
    write_intermediate_snapshot: bool,
    run_status: str = "completed",
    stage_status: dict[str, Any] | None = None,
) -> None:
    manifest_path = resolve_snapshot_manifest_path(outputs_root)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    prediction_paths = resolve_prediction_paths(outputs_root)
    payload = {
        "snapshot_id": snapshot_id,
        "snapshot_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "pipeline": "astra_inference",
        "split": split,
        "limit": limit,
        "row_count": row_count,
        "llm_enabled": llm_enabled,
        "provider": provider,
        "model": model,
        "source_reports_path": str(source_reports_path),
        "run_status": run_status,
        "stages": stage_status or {
            "inference": {"status": run_status, "completed": run_status == "completed"},
            "nlp_eval": {"status": "pending", "completed": False},
            "finance": {"status": "pending", "completed": False},
            "paper_export": {"status": "pending", "completed": False},
        },
        "paths": {
            "predictions": {key: str(value) for key, value in prediction_paths.items()},
            "case_studies": str(resolve_case_path(outputs_root)),
            "intermediate": str(resolve_intermediate_path(outputs_root)) if write_intermediate_snapshot else None,
        },
    }
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def main(
    limit: int = 100,
    split: str = "dev",
    *,
    input_path: Path | None = None,
    outputs_root: Path | None = None,
    snapshot_id: str | None = None,
    write_canonical: bool = True,
    write_intermediate_snapshot: bool = True,
    seed: int | None = None,
    prefer_experiment_split: bool = False,
    resume: bool = True,
) -> None:
    if snapshot_id and outputs_root is None:
        outputs_root = resolve_snapshot_root(snapshot_id)
    rows = _load_rows(
        limit=limit,
        split=split,
        input_path=input_path,
        prefer_experiment_split=prefer_experiment_split,
    )
    source_reports_path = input_path or resolve_reports_input_path(
        split,
        prefer_experiment_split=prefer_experiment_split,
    )
    total_rows = len(rows)
    llm_config = load_llm_settings()
    if seed is not None:
        llm_config.seed = seed
    llm_client = ClaudeJSONClient(llm_config)
    retriever = HistoryRetriever()
    decomposer = Decomposer(llm_client)
    neutralizer = CounterfactualNeutralizer(llm_client)
    verifier = Verifier(llm_client)

    print(
        json.dumps(
            {
                "event": "inference_started",
                "split": split,
                "limit": limit,
                "total_rows": total_rows,
                "llm_enabled": llm_client.enabled,
                "provider": llm_client.config.provider,
                "model": llm_client.config.model,
                "seed": llm_client.config.seed,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )

    rule_predictions: list[dict[str, Any]] = []
    direct_predictions: list[dict[str, Any]] = []
    cot_predictions: list[dict[str, Any]] = []
    react_predictions: list[dict[str, Any]] = []
    astra_predictions: list[dict[str, Any]] = []
    case_rows: list[dict[str, Any]] = []
    intermediate_rows: list[dict[str, Any]] = []
    ablation_outputs: dict[str, list[dict[str, Any]]] = {
        "minus_retrieval": [],
        "minus_neutralizer": [],
        "minus_verifier": [],
        "minus_uncertainty_gate": [],
        "minus_analyst_prior": [],
    }

    def _write_outputs(root: Path | None = None) -> None:
        prediction_paths = resolve_prediction_paths(root)
        _write_jsonl(prediction_paths["rule_baseline"], rule_predictions)
        if direct_predictions:
            _write_jsonl(prediction_paths["direct_llm"], direct_predictions)
            _write_jsonl(prediction_paths["cot_llm"], cot_predictions)
            _write_jsonl(prediction_paths["react_llm"], react_predictions)
        if astra_predictions:
            _write_jsonl(prediction_paths["astra_mvp"], astra_predictions)
            _write_jsonl(resolve_case_path(root), case_rows)
            for name, rows_for_name in ablation_outputs.items():
                _write_jsonl(prediction_paths[f"astra_{name}"], rows_for_name)
        if write_intermediate_snapshot and intermediate_rows and root is not None:
            _write_jsonl(resolve_intermediate_path(root), intermediate_rows)

    processed_report_ids: set[str] = set()
    if resume and outputs_root is not None:
        prediction_paths = resolve_prediction_paths(outputs_root)
        rule_predictions = _load_jsonl(prediction_paths["rule_baseline"])
        direct_predictions = _load_jsonl(prediction_paths["direct_llm"])
        cot_predictions = _load_jsonl(prediction_paths["cot_llm"])
        react_predictions = _load_jsonl(prediction_paths["react_llm"])
        astra_predictions = _load_jsonl(prediction_paths["astra_mvp"])
        case_rows = _load_jsonl(resolve_case_path(outputs_root))
        intermediate_rows = _load_jsonl(resolve_intermediate_path(outputs_root)) if write_intermediate_snapshot else []
        for name in ablation_outputs:
            ablation_outputs[name] = _load_jsonl(prediction_paths[f"astra_{name}"])
        processed_report_ids = {row.get("report_id", "") for row in astra_predictions if row.get("report_id")}
        if processed_report_ids:
            rows = [row for row in rows if row.get("report_id") not in processed_report_ids]
            total_rows = len(rows) + len(processed_report_ids)

    for index, row in enumerate(rows, start=1):
        print(
            json.dumps(
                {
                    "event": "row_started",
                    "index": index,
                    "total": total_rows,
                    "report_id": row.get("report_id", ""),
                    "stock_code": row.get("stock_code", ""),
                    "report_date": row.get("report_date", ""),
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
        title = row["title"]
        summary = row["summary"]
        rule_pred = rule_based_prediction(title, summary)
        rule_predictions.append({**row, "method": "rule_baseline", **rule_pred})

        if not llm_client.enabled:
            print(
                json.dumps(
                    {
                        "event": "row_completed",
                        "index": len(processed_report_ids) + len(rule_predictions),
                        "total": total_rows,
                        "report_id": row.get("report_id", ""),
                        "llm_used": False,
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
            if outputs_root is not None:
                _write_outputs(outputs_root)
            continue

        history_context = retriever.format_context(row["stock_code"], row["report_date"])
        direct_pred = _safe_llm_prediction(llm_client, title, summary, style="direct")
        cot_pred = _safe_llm_prediction(llm_client, title, summary, style="cot")
        react_pred = _safe_llm_prediction(llm_client, title, summary, style="react")
        direct_predictions.append({**row, "method": "direct_llm", **direct_pred})
        cot_predictions.append({**row, "method": "cot_llm", **cot_pred})
        react_predictions.append({**row, "method": "react_llm", **react_pred})
        routed_pred = route_prediction_labels(
            direct_prediction=direct_pred,
            cot_prediction=cot_pred,
            react_prediction=react_pred,
            rule_prediction=rule_pred,
            title=title,
            summary=summary,
        )

        decomposition = _safe_decomposition(decomposer, title=title, summary=summary, history_context=history_context)
        decomposition["history_context_used"] = history_context
        neutralized = _safe_neutralization(neutralizer, title=title, summary=summary, decomposition=decomposition)
        verification = _safe_verification(verifier, title=title, summary=summary, neutralized_text=neutralized.get("neutralized_text", ""))
        refined_pred = refine_labels_from_pipeline(
            routed_pred,
            decomposition=decomposition,
            neutralization=neutralized,
            verification=verification,
        )
        astra_prediction = build_astra_prediction(
            report=row,
            direct_prediction=refined_pred,
            decomposition=decomposition,
            neutralization=neutralized,
            verifier_result=verification,
            method="astra_mvp",
        )
        astra_predictions.append(astra_prediction)
        case_rows.append(_build_case_rows(row, direct_pred, astra_prediction))
        intermediate_rows.append(
            _build_intermediate_row(
                row,
                split=split,
                llm_provider=llm_client.config.provider,
                llm_model=llm_client.config.model,
                rule_prediction=rule_pred,
                direct_prediction=direct_pred,
                cot_prediction=cot_pred,
                react_prediction=react_pred,
                routed_prediction=routed_pred,
                history_context_used=history_context,
                decomposition=decomposition,
                neutralization=neutralized,
                verification=verification,
            )
        )

        # --- Ablation: minus_retrieval ---
        # Re-run decomposer without history, then neutralizer and verifier
        no_hist_decomp = _safe_decomposition(
            decomposer, title=title, summary=summary,
            history_context="无历史研报上下文",
        )
        no_hist_decomp["history_context_used"] = "无历史研报上下文"
        no_hist_neutral = _safe_neutralization(
            neutralizer, title=title, summary=summary,
            decomposition=no_hist_decomp,
        )
        no_hist_verif = _safe_verification(
            verifier, title=title, summary=summary,
            neutralized_text=no_hist_neutral.get("neutralized_text", ""),
        )
        no_hist_refined = refine_labels_from_pipeline(
            routed_pred,
            decomposition=no_hist_decomp,
            neutralization=no_hist_neutral,
            verification=no_hist_verif,
            use_retrieval=False,
        )
        ablation_outputs["minus_retrieval"].append(
            build_astra_prediction(
                report=row,
                direct_prediction=no_hist_refined,
                decomposition=no_hist_decomp,
                neutralization=no_hist_neutral,
                verifier_result=no_hist_verif,
                method="astra_minus_retrieval",
                use_retrieval=False,
            )
        )

        # --- Ablation: minus_neutralizer ---
        # Skip neutralization; verifier compares original against itself
        no_neutral_verif = _safe_verification(
            verifier, title=title, summary=summary,
            neutralized_text=summary,
        )
        no_neutral_refined = refine_labels_from_pipeline(
            routed_pred,
            decomposition=decomposition,
            neutralization={},
            verification=no_neutral_verif,
            use_neutralizer=False,
        )
        ablation_outputs["minus_neutralizer"].append(
            build_astra_prediction(
                report=row,
                direct_prediction=no_neutral_refined,
                decomposition=decomposition,
                neutralization={},
                verifier_result=no_neutral_verif,
                method="astra_minus_neutralizer",
                use_neutralizer=False,
            )
        )

        # --- Ablation: minus_verifier ---
        # Keep neutralization but skip verification
        no_verif_refined = refine_labels_from_pipeline(
            routed_pred,
            decomposition=decomposition,
            neutralization=neutralized,
            verification={},
            use_verifier=False,
        )
        ablation_outputs["minus_verifier"].append(
            build_astra_prediction(
                report=row,
                direct_prediction=no_verif_refined,
                decomposition=decomposition,
                neutralization=neutralized,
                verifier_result={},
                method="astra_minus_verifier",
                use_verifier=False,
            )
        )

        # --- Ablation: minus_uncertainty_gate (scoring-only) ---
        ablation_outputs["minus_uncertainty_gate"].append(
            build_astra_prediction(
                report=row,
                direct_prediction=refined_pred,
                decomposition=decomposition,
                neutralization=neutralized,
                verifier_result=verification,
                method="astra_minus_uncertainty_gate",
                use_uncertainty_gate=False,
            )
        )

        # --- Ablation: minus_analyst_prior (scoring-only) ---
        ablation_outputs["minus_analyst_prior"].append(
            build_astra_prediction(
                report=row,
                direct_prediction=refined_pred,
                decomposition=decomposition,
                neutralization=neutralized,
                verifier_result=verification,
                method="astra_minus_analyst_prior",
                use_analyst_prior=False,
            )
        )

        print(
            json.dumps(
                {
                    "event": "row_completed",
                    "index": len(processed_report_ids) + len(astra_predictions) if llm_client.enabled else len(processed_report_ids) + len(rule_predictions),
                    "total": total_rows,
                    "report_id": row.get("report_id", ""),
                    "llm_used": True,
                    "direct_predictions": len(direct_predictions),
                    "astra_predictions": len(astra_predictions),
                    "case_rows": len(case_rows),
                },
                ensure_ascii=False,
            ),
            flush=True,
        )

        if outputs_root is not None:
            _write_outputs(outputs_root)

    def _write_outputs(root: Path | None = None) -> None:
        prediction_paths = resolve_prediction_paths(root)
        _write_jsonl(prediction_paths["rule_baseline"], rule_predictions)
        if direct_predictions:
            _write_jsonl(prediction_paths["direct_llm"], direct_predictions)
            _write_jsonl(prediction_paths["cot_llm"], cot_predictions)
            _write_jsonl(prediction_paths["react_llm"], react_predictions)
        if astra_predictions:
            _write_jsonl(prediction_paths["astra_mvp"], astra_predictions)
            _write_jsonl(resolve_case_path(root), case_rows)
            for name, rows_for_name in ablation_outputs.items():
                _write_jsonl(prediction_paths[f"astra_{name}"], rows_for_name)
        if write_intermediate_snapshot and intermediate_rows and root is not None:
            _write_jsonl(resolve_intermediate_path(root), intermediate_rows)

    if write_canonical:
        _write_outputs()
    if outputs_root is not None:
        _write_outputs(outputs_root)
        _write_snapshot_manifest(
            outputs_root=outputs_root,
            snapshot_id=snapshot_id or outputs_root.name,
            split=split,
            limit=limit,
            row_count=total_rows,
            llm_enabled=llm_client.enabled,
            provider=llm_client.config.provider,
            model=llm_client.config.model,
            source_reports_path=source_reports_path,
            write_intermediate_snapshot=write_intermediate_snapshot,
        )

    print("[OK] Inference completed.")
    print(json.dumps({
        "rule_predictions": len(rule_predictions),
        "direct_predictions": len(direct_predictions),
        "cot_predictions": len(cot_predictions),
        "react_predictions": len(react_predictions),
        "astra_predictions": len(astra_predictions),
        "ablation_predictions": {key: len(value) for key, value in ablation_outputs.items()},
        "case_rows": len(case_rows),
        "llm_enabled": llm_client.enabled,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--split", type=str, default="dev")
    parser.add_argument("--input-path", type=Path, default=None)
    parser.add_argument("--outputs-root", type=Path, default=None)
    parser.add_argument("--snapshot-id", type=str, default=None)
    parser.add_argument("--no-write-canonical", action="store_true")
    parser.add_argument("--write-intermediate-snapshot", action="store_true")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--prefer-experiment-split", action="store_true")
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()
    main(
        limit=args.limit,
        split=args.split,
        input_path=args.input_path,
        outputs_root=args.outputs_root,
        snapshot_id=args.snapshot_id,
        write_canonical=not args.no_write_canonical,
        write_intermediate_snapshot=args.write_intermediate_snapshot,
        seed=args.seed,
        prefer_experiment_split=args.prefer_experiment_split,
        resume=not args.no_resume,
    )
