from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from astra.finance.backtest_cross_sectional import run_cross_sectional_backtest

PRED_PATH = ROOT / "outputs" / "backups" / "20260405_submission_snapshot" / "predictions" / "astra_predictions.jsonl"
OUT_CSV = ROOT / "artifacts" / "finance" / "signal_search_results.csv"
OUT_JSON = ROOT / "artifacts" / "finance" / "signal_search_best.json"


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _score_prediction(row: dict[str, Any], alpha: float, beta: float, gamma: float) -> float:
    sentiment = float(row.get("sentiment_score", 0.0))
    gap = float(row.get("strategic_optimism_gap", 0.0))
    uncertainty = float(row.get("uncertainty", 0.5))
    return alpha * sentiment - beta * gap - gamma * uncertainty


def _evaluate_combo(rows: list[dict[str, Any]], alpha: float, beta: float, gamma: float) -> dict[str, Any]:
    temp_path = ROOT / "artifacts" / "finance" / "tmp_signal_predictions.jsonl"
    temp_rows = []
    for row in rows:
        clone = dict(row)
        custom_score = _score_prediction(row, alpha, beta, gamma)
        clone["astra_composite_score"] = round(custom_score, 6)
        clone["astra_finance_blend_score"] = round(custom_score, 6)
        temp_rows.append(clone)
    _write_jsonl(temp_path, temp_rows)
    result = run_cross_sectional_backtest(temp_path)
    signal_metrics = result.get("signal_metrics", {})
    headline = signal_metrics.get("astra_finance_plus_gap", {})
    return {
        "alpha": alpha,
        "beta": beta,
        "gamma": gamma,
        "mean_rank_ic_20": headline.get("mean_rank_ic@20", 0.0),
        "ls_sharpe_20": headline.get("ls_sharpe@20", 0.0),
        "turnover": headline.get("turnover", 0.0),
        "composite_j_proxy": round(
            0.15 * headline.get("mean_rank_ic@20", 0.0)
            + 0.15 * headline.get("ls_sharpe@20", 0.0),
            6,
        ),
    }


def main() -> None:
    rows = _load_jsonl(PRED_PATH)
    grid = []
    for alpha in (0.6, 0.8, 1.0, 1.2):
        for beta in (0.2, 0.4, 0.6, 0.8, 1.0):
            for gamma in (0.0, 0.1, 0.2, 0.4):
                grid.append(_evaluate_combo(rows, alpha, beta, gamma))

    grid.sort(key=lambda item: (item["composite_j_proxy"], item["ls_sharpe_20"], item["mean_rank_ic_20"]), reverse=True)

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUT_CSV.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["alpha", "beta", "gamma", "mean_rank_ic_20", "ls_sharpe_20", "turnover", "composite_j_proxy"])
        writer.writeheader()
        writer.writerows(grid)

    OUT_JSON.write_text(json.dumps(grid[0], ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"best": grid[0], "top5": grid[:5]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
