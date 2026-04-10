from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from astra.config.task_schema import resolve_backtest_paths, resolve_prediction_paths, resolve_snapshot_manifest_path
from astra.finance.backtest_cross_sectional import SIGNAL_NAMES, run_cross_sectional_backtest
from astra.finance.config import default_backtest_config


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        if rows:
            writer.writerows(rows)


def main(outputs_root: Path | None = None) -> None:
    prediction_paths = resolve_prediction_paths(outputs_root)
    backtest_paths = resolve_backtest_paths(outputs_root)
    config = default_backtest_config()
    main_result = run_cross_sectional_backtest(prediction_paths["astra_mvp"], config=config)
    ablation_paths = {
        "astra_mvp": prediction_paths["astra_mvp"],
        "astra_minus_retrieval": prediction_paths["astra_minus_retrieval"],
        "astra_minus_neutralizer": prediction_paths["astra_minus_neutralizer"],
        "astra_minus_verifier": prediction_paths["astra_minus_verifier"],
        "astra_minus_uncertainty_gate": prediction_paths["astra_minus_uncertainty_gate"],
        "astra_minus_analyst_prior": prediction_paths["astra_minus_analyst_prior"],
    }

    ablation_metrics = {}
    for name, path in ablation_paths.items():
        result = main_result if name == "astra_mvp" else run_cross_sectional_backtest(path, config=config)
        ablation_metrics[name] = {
            "status": result.get("status"),
            **(result.get("headline_metrics") or {}),
        }

    manifest_path = resolve_snapshot_manifest_path(outputs_root)
    payload = {
        "status": main_result.get("status"),
        "prediction_path": main_result.get("prediction_path", str(prediction_paths["astra_mvp"])),
        "report_count": main_result.get("report_count", 0),
        "aligned_stock_day_count": main_result.get("aligned_stock_day_count", 0),
        "trade_date_count": main_result.get("trade_date_count", 0),
        "config": main_result.get("config", config.to_dict()),
        "diagnostics": main_result.get("diagnostics", {}),
        "source_manifest_path": str(manifest_path) if manifest_path.exists() else None,
        "signal_metrics": main_result.get("signal_metrics", {}),
        "cost_stress": main_result.get("cost_stress_rows", []),
        "ablation_metrics": ablation_metrics,
    }

    backtest_paths["metrics"].parent.mkdir(parents=True, exist_ok=True)
    with backtest_paths["metrics"].open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    with backtest_paths["diagnostics"].open("w", encoding="utf-8") as handle:
        json.dump(main_result.get("diagnostics", {}), handle, ensure_ascii=False, indent=2)
    with backtest_paths["config"].open("w", encoding="utf-8") as handle:
        json.dump(main_result.get("config", config.to_dict()), handle, ensure_ascii=False, indent=2)

    _write_csv(
        backtest_paths["curve"],
        main_result.get("curve_rows", []),
        ["date", *SIGNAL_NAMES],
    )
    _write_csv(
        backtest_paths["regime_heatmap"],
        main_result.get("regime_heatmap_rows", []),
        ["volatility_quintile", "horizon", "mean_rank_ic", "observation_count"],
    )
    _write_csv(
        backtest_paths["robustness_grid"],
        main_result.get("cost_stress_rows", []),
        ["signal", "fee_bps", "slippage_bps", "impact_bps", "total_cost_bps", "net_sharpe@20", "turnover", "observation_count"],
    )
    portfolio_returns = []
    primary_signal = config.primary_signal
    for row in main_result.get("curve_rows", []):
        portfolio_returns.append({
            "date": row.get("date"),
            "signal": primary_signal,
            "cumulative_return": row.get(primary_signal),
        })
    _write_csv(
        backtest_paths["portfolio_returns"],
        portfolio_returns,
        ["date", "signal", "cumulative_return"],
    )

    print("[OK] Finance evaluation completed.")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--outputs-root", type=Path, default=None)
    args = parser.parse_args()
    main(outputs_root=args.outputs_root)
