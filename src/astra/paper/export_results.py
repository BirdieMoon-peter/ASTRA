from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from astra.config.task_schema import (
    BACKTEST_CURVE_PATH,
    CALIBRATION_PATH,
    CASE_STUDIES_PATH,
    FINANCE_METRICS_PATH,
    CORPUS_STATS_PATH,
    NLP_METRICS_PATH,
    PAPER_FIGURE_BACKTEST_PATH,
    PAPER_FIGURE_CALIBRATION_PATH,
    PAPER_FIGURE_REGIME_PATH,
    PAPER_LATEX_ABLATION_PATH,
    PAPER_LATEX_CASE_PATH,
    PAPER_LATEX_DATA_PATH,
    PAPER_LATEX_FINANCE_PATH,
    PAPER_LATEX_MAIN_NLP_PATH,
    PAPER_LATEX_PHENOMENA_PATH,
    PAPER_OUTPUT_DIR,
    PAPER_TABLE_ABLATION_PATH,
    PAPER_TABLE_CASE_PATH,
    PAPER_TABLE_DATA_PATH,
    PAPER_TABLE_FINANCE_PATH,
    PAPER_TABLE_MAIN_NLP_PATH,
    PAPER_TABLE_PHENOMENA_PATH,
    PHENOMENA_METRICS_PATH,
    REGIME_HEATMAP_PATH,
    SPLIT_SUMMARY_PATH,
)
from astra.finance.signal_registry import paper_signal_rows, registry_metadata

_OUTPUTS_ROOT = PAPER_OUTPUT_DIR.parent
_PHENOMENA_LABELS = (
    "hedged_downside",
    "euphemistic_risk",
    "title_body_mismatch",
    "omitted_downside_context",
)
_FINANCE_SIGNAL_ROWS = paper_signal_rows()


def _resolve_path(default_path: Path, outputs_root: Path | None) -> Path:
    if outputs_root is None:
        return default_path
    return outputs_root / default_path.relative_to(_OUTPUTS_ROOT)


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in {path}: {exc}") from exc


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSONL in {path} at line {line_number}: {exc}") from exc
        if not isinstance(payload, dict):
            raise ValueError(f"Invalid JSONL object in {path} at line {line_number}: expected object")
        rows.append(payload)
    return rows


def _copy_csv_rows(path: Path) -> tuple[list[str], list[dict[str, Any]]]:
    if not path.exists():
        return [], []
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            fieldnames = list(reader.fieldnames or [])
            if not fieldnames:
                return [], []
            rows = [dict(row) for row in reader]
    except csv.Error as exc:
        raise ValueError(f"Invalid CSV in {path}: {exc}") from exc
    return fieldnames, rows


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        if rows:
            writer.writerows(rows)


def _safe_pct(value: Any) -> float:
    if value in (None, ""):
        return 0.0
    return round(float(value) * 100.0, 2)


def _safe_float(value: Any) -> float:
    if value in (None, ""):
        return 0.0
    return float(value)


def _escape_latex_text(value: str) -> str:
    return (
        value.replace("\\", r"\textbackslash{}")
        .replace("&", r"\&")
        .replace("%", r"\%")
        .replace("$", r"\$")
        .replace("#", r"\#")
        .replace("_", r"\_")
        .replace("{", r"\{")
        .replace("}", r"\}")
    )



def _format_latex_value(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.2f}"
    return _escape_latex_text(str(value))


def _render_latex_table(headers: list[str], rows: list[list[Any]]) -> str:
    column_spec = "l" + "r" * max(len(headers) - 1, 0)
    escaped_headers = [_escape_latex_text(header) for header in headers]
    lines = [f"\\begin{{tabular}}{{{column_spec}}}", "\\hline", " & ".join(escaped_headers) + r" \\", "\\hline"]
    for row in rows:
        lines.append(" & ".join(_format_latex_value(value) for value in row) + r" \\")
    lines.extend(["\\hline", "\\end{tabular}", ""])
    return "\n".join(lines)


_ABLATION_LABELS = {
    "astra_mvp": "Full ASTRA",
    "astra_minus_retrieval": "No retrieval context",
    "astra_minus_neutralizer": "No neutralizer",
    "astra_minus_verifier": "No verifier",
    "astra_minus_uncertainty_gate": "No uncertainty gate",
    "astra_minus_analyst_prior": "No analyst prior",
}


_CASE_LABELS = {
    "report_id": "Report ID",
    "stock_code": "Stock code",
    "report_date": "Report date",
    "title": "Original title",
    "summary": "Original summary",
    "gap": "Strategic optimism gap",
    "uncertainty": "Uncertainty",
    "verifier_verdict": "Verifier verdict",
    "neutralized_text": "Neutralized text",
}


def _render_data_table_latex(rows: list[dict[str, Any]]) -> str:
    newline = r"\\"
    lines = [r"\begin{tabularx}{\columnwidth}{@{}lX@{}}", r"\toprule", f"Item & Value {newline}", r"\midrule"]
    for row in rows:
        item = _escape_latex_text(str(row.get("item", "")))
        value = _escape_latex_text(str(row.get("value", "")))
        lines.append(f"{item} & {value} {newline}")
    lines.extend([r"\bottomrule", r"\end{tabularx}", ""])
    return "\n".join(lines)


def _render_ablation_latex(headers: list[str], rows: list[dict[str, Any]]) -> str:
    del headers
    header_cells = [
        "Setting",
        r"\shortstack[c]{Strat.\\F1}",
        r"\shortstack[c]{Evid.\\F1}",
        r"\shortstack[c]{Rank\\IC@20}",
        r"\shortstack[c]{Sharpe\\@20}",
    ]
    lines = [
        r"\begin{tabular}{@{}p{0.40\columnwidth}cccc@{}}",
        r"\toprule",
        " & ".join(header_cells) + r" \\",
        r"\midrule",
    ]
    for row in rows:
        label = _ABLATION_LABELS.get(str(row.get("setting", "")), str(row.get("setting", "")))
        values = [
            _escape_latex_text(label),
            _format_latex_value(row.get("strategic_f1", "")),
            _format_latex_value(row.get("evidence_f1", "")),
            _format_latex_value(row.get("mean_rank_ic_20", "")),
            _format_latex_value(row.get("ls_sharpe_20", "")),
        ]
        lines.append(" & ".join(values) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}", ""])
    return "\n".join(lines)


def _render_case_latex(row: dict[str, Any]) -> str:
    field_order = [
        "report_id",
        "stock_code",
        "report_date",
        "title",
        "summary",
        "gap",
        "uncertainty",
        "verifier_verdict",
        "neutralized_text",
    ]
    wide_fields = {"title", "summary", "neutralized_text"}
    lines = [r"\begin{tabularx}{\textwidth}{@{}lX@{}}", r"\toprule"]
    for index, field in enumerate(field_order):
        if index:
            lines.append(r"\midrule")
        label = _escape_latex_text(_CASE_LABELS[field])
        value = _format_latex_value(row.get(field, ""))
        if field in wide_fields:
            lines.append(rf"\multicolumn{{2}}{{@{{}}X@{{}}}}{{\textbf{{{label}}}: {value}}} \\")
        else:
            lines.append(rf"\textbf{{{label}}} & {value} \\")
    lines.extend([r"\bottomrule", r"\end{tabularx}", ""])
    return "\n".join(lines)


def _write_latex(path: Path, headers: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    body_rows = [[row.get(header, "") for header in headers] for row in rows]
    path.write_text(_render_latex_table(headers, body_rows), encoding="utf-8")


def _latex_escape_unicode_text(value: Any) -> str:
    text = str(value).strip()
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
    }
    for source, target in replacements.items():
        text = text.replace(source, target)
    return text


def _phenomena_row(phenomena_metrics: dict[str, Any], label: str) -> dict[str, Any]:
    astra_per_label = ((phenomena_metrics.get("astra_mvp") or {}).get("per_label") or {})
    row = astra_per_label.get(label)
    if row is None:
        row = phenomena_metrics.get(label) or {}
    return {
        "phenomenon": label,
        "support": int(row.get("count", row.get("support", 0)) or 0),
        "accuracy": _safe_pct(row.get("accuracy")),
    }


def export_paper_artifacts(outputs_root: Path | None = None) -> None:
    nlp_metrics_path = _resolve_path(NLP_METRICS_PATH, outputs_root)
    corpus_stats_path = _resolve_path(CORPUS_STATS_PATH, outputs_root)
    split_summary_path = _resolve_path(SPLIT_SUMMARY_PATH, outputs_root)
    phenomena_metrics_path = _resolve_path(PHENOMENA_METRICS_PATH, outputs_root)
    calibration_path = _resolve_path(CALIBRATION_PATH, outputs_root)
    finance_metrics_path = _resolve_path(FINANCE_METRICS_PATH, outputs_root)
    case_studies_path = _resolve_path(CASE_STUDIES_PATH, outputs_root)
    backtest_curve_path = _resolve_path(BACKTEST_CURVE_PATH, outputs_root)
    regime_heatmap_path = _resolve_path(REGIME_HEATMAP_PATH, outputs_root)

    paper_table_main_nlp_path = _resolve_path(PAPER_TABLE_MAIN_NLP_PATH, outputs_root)
    paper_table_data_path = _resolve_path(PAPER_TABLE_DATA_PATH, outputs_root)
    paper_table_phenomena_path = _resolve_path(PAPER_TABLE_PHENOMENA_PATH, outputs_root)
    paper_table_finance_path = _resolve_path(PAPER_TABLE_FINANCE_PATH, outputs_root)
    paper_table_ablation_path = _resolve_path(PAPER_TABLE_ABLATION_PATH, outputs_root)
    paper_table_case_path = _resolve_path(PAPER_TABLE_CASE_PATH, outputs_root)
    paper_figure_backtest_path = _resolve_path(PAPER_FIGURE_BACKTEST_PATH, outputs_root)
    paper_figure_calibration_path = _resolve_path(PAPER_FIGURE_CALIBRATION_PATH, outputs_root)
    paper_figure_regime_path = _resolve_path(PAPER_FIGURE_REGIME_PATH, outputs_root)
    paper_latex_main_nlp_path = _resolve_path(PAPER_LATEX_MAIN_NLP_PATH, outputs_root)
    paper_latex_data_path = _resolve_path(PAPER_LATEX_DATA_PATH, outputs_root)
    paper_latex_phenomena_path = _resolve_path(PAPER_LATEX_PHENOMENA_PATH, outputs_root)
    paper_latex_finance_path = _resolve_path(PAPER_LATEX_FINANCE_PATH, outputs_root)
    paper_latex_ablation_path = _resolve_path(PAPER_LATEX_ABLATION_PATH, outputs_root)
    paper_latex_case_path = _resolve_path(PAPER_LATEX_CASE_PATH, outputs_root)

    nlp_metrics = _load_json(nlp_metrics_path)
    corpus_stats = _load_json(corpus_stats_path)
    split_summary = _load_json(split_summary_path)
    phenomena_metrics = _load_json(phenomena_metrics_path)
    calibration_bins = _load_json(calibration_path)
    finance_metrics = _load_json(finance_metrics_path)
    case_rows = _load_jsonl(case_studies_path)

    data_headers = ["item", "value"]
    split_counts = split_summary.get("split_counts") or {}
    years_by_split = split_summary.get("years_by_split") or {}
    train_years = years_by_split.get("train") or {}
    dev_years = years_by_split.get("dev") or {}
    test_years = years_by_split.get("test") or {}
    title_length = corpus_stats.get("title_length") or {}
    summary_length = corpus_stats.get("summary_length") or {}
    astra_per_label = ((phenomena_metrics.get("astra_mvp") or {}).get("per_label") or {})
    aligned_count = int((nlp_metrics.get("astra_mvp") or {}).get("matched_gold_count", 0) or 0)
    data_rows = [
        {
            "item": "Raw corpus",
            "value": f"{int(corpus_stats.get('raw_report_count', 0)):,} title-summary pairs over {corpus_stats.get('date_range', {}).get('min', '')[:4]}--{corpus_stats.get('date_range', {}).get('max', '')[:4]}",
        },
        {
            "item": "Clean corpus",
            "value": f"{int(corpus_stats.get('clean_report_count', 0)):,} reports after deduplication",
        },
        {
            "item": "Covered stocks / companies",
            "value": f"{int(corpus_stats.get('unique_stock_count', 0)):,} stock codes / {int(corpus_stats.get('unique_company_count', 0)):,} companies",
        },
        {
            "item": "Split counts",
            "value": f"{int(split_counts.get('train', 0)):,} train ({'--'.join([next(iter(train_years.keys()), ''), list(train_years.keys())[-1] if train_years else ''])}) / {int(split_counts.get('dev', 0)):,} dev ({next(iter(dev_years.keys()), '')}) / {int(split_counts.get('test', 0)):,} test ({next(iter(test_years.keys()), '')})",
        },
        {
            "item": "Average title length",
            "value": f"{float(title_length.get('mean', 0.0)):.1f} Chinese characters",
        },
        {
            "item": "Average summary length",
            "value": f"{float(summary_length.get('mean', 0.0)):.1f} Chinese characters",
        },
        {
            "item": "Aligned NLP evaluation slice",
            "value": f"{aligned_count:,} manually aligned dev reports",
        },
        {
            "item": "Phenomenon support in aligned slice",
            "value": f"{int((astra_per_label.get('hedged_downside') or {}).get('count', 0))} hedged downside / {int((astra_per_label.get('euphemistic_risk') or {}).get('count', 0))} euphemistic risk / {int((astra_per_label.get('title_body_mismatch') or {}).get('count', 0))} title--body mismatch / {int((astra_per_label.get('omitted_downside_context') or {}).get('count', 0))} omitted-downside",
        },
    ]
    _write_csv(paper_table_data_path, data_headers, data_rows)
    paper_latex_data_path.parent.mkdir(parents=True, exist_ok=True)
    paper_latex_data_path.write_text(_render_data_table_latex(data_rows), encoding='utf-8')

    main_headers = ["model", "sentiment_f1", "strategic_f1", "evidence_f1", "ece"]
    main_rows = []
    for key in ("rule_baseline", "direct_llm", "cot_llm", "react_llm", "astra_mvp"):
        row = nlp_metrics.get(key) or {}
        main_rows.append(
            {
                "model": key,
                "sentiment_f1": _safe_pct(row.get("fundamental_sentiment_macro_f1")),
                "strategic_f1": _safe_pct(row.get("strategic_optimism_macro_f1")),
                "evidence_f1": _safe_pct(row.get("evidence_f1")),
                "ece": _safe_pct(row.get("ece")),
            }
        )
    _write_csv(paper_table_main_nlp_path, main_headers, main_rows)
    _write_latex(paper_latex_main_nlp_path, main_headers, main_rows)

    phenomena_headers = ["phenomenon", "support", "accuracy"]
    phenomena_rows = [_phenomena_row(phenomena_metrics, label) for label in _PHENOMENA_LABELS]
    _write_csv(paper_table_phenomena_path, phenomena_headers, phenomena_rows)
    _write_latex(paper_latex_phenomena_path, phenomena_headers, phenomena_rows)

    finance_headers = ["signal", "mean_rank_ic_5", "mean_rank_ic_10", "mean_rank_ic_20", "ls_sharpe_20", "turnover"]
    finance_rows = []
    signal_metrics = finance_metrics.get("signal_metrics") or {}
    for key in _FINANCE_SIGNAL_ROWS:
        row = signal_metrics.get(key) or {}
        finance_rows.append(
            {
                "signal": key,
                "mean_rank_ic_5": _safe_float(row.get("mean_rank_ic@5")),
                "mean_rank_ic_10": _safe_float(row.get("mean_rank_ic@10")),
                "mean_rank_ic_20": _safe_float(row.get("mean_rank_ic@20")),
                "ls_sharpe_20": _safe_float(row.get("ls_sharpe@20")),
                "turnover": _safe_float(row.get("turnover")),
            }
        )
    _write_csv(paper_table_finance_path, finance_headers, finance_rows)
    _write_latex(paper_latex_finance_path, finance_headers, finance_rows)

    ablation_headers = ["setting", "strategic_f1", "evidence_f1", "mean_rank_ic_20", "ls_sharpe_20"]
    ablation_rows = []
    ablation_metrics = finance_metrics.get("ablation_metrics") or {}
    for key in (
        "astra_mvp",
        "astra_minus_retrieval",
        "astra_minus_neutralizer",
        "astra_minus_verifier",
        "astra_minus_uncertainty_gate",
        "astra_minus_analyst_prior",
    ):
        nlp_row = nlp_metrics.get(key) or {}
        finance_row = ablation_metrics.get(key) or {}
        ablation_rows.append(
            {
                "setting": key,
                "strategic_f1": _safe_pct(nlp_row.get("strategic_optimism_macro_f1")),
                "evidence_f1": _safe_pct(nlp_row.get("evidence_f1")),
                "mean_rank_ic_20": _safe_float(finance_row.get("mean_rank_ic@20")),
                "ls_sharpe_20": _safe_float(finance_row.get("ls_sharpe@20")),
            }
        )
    _write_csv(paper_table_ablation_path, ablation_headers, ablation_rows)
    paper_latex_ablation_path.parent.mkdir(parents=True, exist_ok=True)
    paper_latex_ablation_path.write_text(_render_ablation_latex(ablation_headers, ablation_rows), encoding="utf-8")

    case_headers = [
        "report_id",
        "stock_code",
        "report_date",
        "title",
        "summary",
        "gap",
        "uncertainty",
        "verifier_verdict",
        "neutralized_text",
    ]
    first_case = case_rows[0] if case_rows else {}
    case_table_rows = [{header: first_case.get(header, "") for header in case_headers}]
    case_latex_rows = [
        {
            "report_id": first_case.get("report_id", ""),
            "stock_code": first_case.get("stock_code", ""),
            "report_date": first_case.get("report_date", ""),
            "title": _latex_escape_unicode_text(first_case.get("title", "")),
            "summary": _latex_escape_unicode_text(first_case.get("summary", "")),
            "gap": first_case.get("gap", ""),
            "uncertainty": first_case.get("uncertainty", ""),
            "verifier_verdict": first_case.get("verifier_verdict", ""),
            "neutralized_text": _latex_escape_unicode_text(first_case.get("neutralized_text", "")),
        }
    ]
    _write_csv(paper_table_case_path, case_headers, case_table_rows)
    paper_latex_case_path.parent.mkdir(parents=True, exist_ok=True)
    paper_latex_case_path.write_text(_render_case_latex(case_latex_rows[0]), encoding="utf-8")

    backtest_headers, backtest_rows = _copy_csv_rows(backtest_curve_path)
    _write_csv(paper_figure_backtest_path, backtest_headers or ["date"], backtest_rows)

    calibration_headers = ["model", "bin", "accuracy"]
    calibration_rows = []
    for key in ("direct_llm", "astra_mvp"):
        for row in calibration_bins.get(key) or []:
            calibration_rows.append(
                {
                    "model": key,
                    "bin": _safe_float(row.get("bin")),
                    "accuracy": _safe_float(row.get("accuracy")),
                }
            )
    _write_csv(paper_figure_calibration_path, calibration_headers, calibration_rows)

    regime_headers, regime_rows = _copy_csv_rows(regime_heatmap_path)
    _write_csv(
        paper_figure_regime_path,
        regime_headers or ["volatility_quintile", "horizon", "mean_rank_ic", "observation_count"],
        regime_rows,
    )
