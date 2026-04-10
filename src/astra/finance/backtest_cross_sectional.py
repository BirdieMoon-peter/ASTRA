from __future__ import annotations

import csv
import json
import math
import random
from bisect import bisect_left
from collections import defaultdict
from pathlib import Path
from typing import Any

from astra.config.task_schema import resolve_market_prices_path
from astra.finance.build_signals import build_report_signal
from astra.finance.config import BacktestConfig, default_backtest_config
from astra.finance.signal_registry import primary_backtest_signal, signal_names

FORWARD_HORIZONS = (5, 10, 20)
SIGNAL_NAMES = signal_names()
PRIMARY_SIGNAL = primary_backtest_signal()


def _safe_float(value: Any, default: float | None = None) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _average(values: list[float]) -> float:
    if not values:
        return 0.0
    return float(sum(values)) / float(len(values))


def _stdev(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean = _average(values)
    variance = sum((value - mean) ** 2 for value in values) / float(len(values) - 1)
    return math.sqrt(max(variance, 0.0))


def _bootstrap_ci(
    values: list[float], n_bootstrap: int = 1000, ci: float = 0.95
) -> tuple[float, float]:
    """Compute bootstrap confidence interval for the mean of *values*."""
    if not values:
        return (0.0, 0.0)
    n = len(values)
    rng = random.Random(42)
    means: list[float] = []
    for _ in range(n_bootstrap):
        sample = [values[rng.randint(0, n - 1)] for _ in range(n)]
        means.append(sum(sample) / n)
    means.sort()
    alpha = 1.0 - ci
    lo_idx = max(0, int(math.floor((alpha / 2.0) * len(means))))
    hi_idx = min(len(means) - 1, int(math.floor((1.0 - alpha / 2.0) * len(means))))
    return (round(means[lo_idx], 6), round(means[hi_idx], 6))


def _newey_west_tstat(
    values: list[float], lag: int | None = None
) -> tuple[float, float]:
    """Newey-West t-statistic and p-value for H0: mean == 0."""
    n = len(values)
    if n < 2:
        return (0.0, 1.0)
    mean = _average(values)
    if lag is None:
        lag = int(n ** (1 / 3))
    # demeaned residuals
    resid = [v - mean for v in values]
    # gamma_0
    gamma_0 = sum(r * r for r in resid) / n
    # Newey-West weighted autocovariances
    nw_var = gamma_0
    for j in range(1, lag + 1):
        gamma_j = sum(resid[t] * resid[t - j] for t in range(j, n)) / n
        weight = 1.0 - j / (lag + 1.0)
        nw_var += 2.0 * weight * gamma_j
    se = math.sqrt(max(nw_var / n, 1e-18))
    tstat = mean / se
    # two-sided p-value approximation using the normal CDF
    p_value = 2.0 * _normal_survival(abs(tstat))
    return (round(tstat, 6), round(p_value, 6))


def _normal_survival(x: float) -> float:
    """Approximate P(Z > x) for standard normal using Abramowitz & Stegun."""
    # Uses the rational approximation (formula 26.2.17) for the complementary
    # error function, accurate to ~1e-7 over the full range.
    t = 1.0 / (1.0 + 0.2316419 * abs(x))
    d = 0.3989422804014327  # 1/sqrt(2*pi)
    poly = t * (0.319381530 + t * (-0.356563782 + t * (1.781477937 + t * (-1.821255978 + t * 1.330274429))))
    survival = d * math.exp(-0.5 * x * x) * poly
    return max(0.0, min(1.0, survival))


def diagnostic_summary(records: list[dict], min_dates: int = 20) -> dict:
    """Produce a diagnostic summary dict for a list of backtest records."""
    trade_dates: set[str] = set()
    stocks_per_date: dict[str, int] = defaultdict(int)
    for rec in records:
        td = str(rec.get("trade_date", ""))
        if td:
            trade_dates.add(td)
            stocks_per_date[td] += 1
    n_trade_dates = len(trade_dates)
    n_total = len(records)
    counts = list(stocks_per_date.values())
    n_stocks_mean = round(_average(counts), 4) if counts else 0.0
    adequate = n_trade_dates >= min_dates
    warning = "" if adequate else (
        f"Only {n_trade_dates} trade dates available (minimum {min_dates}); "
        "results may be unreliable."
    )
    return {
        "n_trade_dates": n_trade_dates,
        "n_stocks_per_date_mean": n_stocks_mean,
        "n_total_observations": n_total,
        "sample_adequate": adequate,
        "warning": warning,
    }


def _load_prediction_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _load_market_series(stock_codes: set[str]) -> dict[str, list[dict[str, float | str]]]:
    market_prices_path = resolve_market_prices_path()
    if not market_prices_path.exists() or not stock_codes:
        return {}
    series_by_stock: dict[str, list[dict[str, float | str]]] = defaultdict(list)
    with market_prices_path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            stock_code = row.get("stock_code")
            if stock_code not in stock_codes:
                continue
            close = _safe_float(row.get("close"))
            if close is None:
                continue
            series_by_stock[stock_code].append(
                {
                    "trade_date": row["trade_date"],
                    "close": close,
                    "pct_change": _safe_float(row.get("pct_change"), 0.0) or 0.0,
                }
            )
    for rows in series_by_stock.values():
        rows.sort(key=lambda item: str(item["trade_date"]))
    return dict(series_by_stock)


def _rolling_volatility(rows: list[dict[str, float | str]], window: int = 20) -> dict[str, float]:
    history: list[float] = []
    vol_by_date: dict[str, float] = {}
    for row in rows:
        history.append(float(row.get("pct_change", 0.0)) / 100.0)
        window_values = history[-window:]
        mean = _average(window_values)
        variance = _average([(value - mean) ** 2 for value in window_values])
        vol_by_date[str(row["trade_date"])] = round(math.sqrt(max(variance, 0.0)), 8)
    return vol_by_date


def _build_market_context(stock_codes: set[str]) -> dict[str, dict[str, Any]]:
    context = {}
    for stock_code, rows in _load_market_series(stock_codes).items():
        dates = [str(row["trade_date"]) for row in rows]
        context[stock_code] = {
            "rows": rows,
            "dates": dates,
            "volatility": _rolling_volatility(rows),
        }
    return context


def _forward_return(
    rows: list[dict[str, float | str]], index: int, horizon: int, *, entry_field: str = "close", exit_field: str = "close"
) -> float | None:
    exit_index = index + horizon
    if index >= len(rows) or exit_index >= len(rows):
        return None
    entry_close = _safe_float(rows[index].get(entry_field))
    exit_close = _safe_float(rows[exit_index].get(exit_field))
    if entry_close in (None, 0.0) or exit_close is None:
        return None
    return round(exit_close / entry_close - 1.0, 6)


def _entry_index_for_policy(index: int, policy_name: str) -> int:
    if policy_name == "same_day_close":
        return index
    if policy_name in {"next_open", "next_close"}:
        return index + 1
    if policy_name == "t_plus_1_close":
        return index + 1
    return index + 1


def _entry_field_for_policy(policy_name: str) -> str:
    if policy_name == "next_open":
        return "open"
    return "close"


def _is_excluded_by_universe(
    prediction: dict[str, Any],
    market_row: dict[str, float | str],
    entry_index: int,
    config: BacktestConfig,
) -> bool:
    universe = config.universe
    if universe.exclude_st and str(prediction.get("is_st", market_row.get("is_st", "false"))).lower() in {"1", "true", "yes"}:
        return True
    if universe.exclude_pt and str(prediction.get("is_pt", market_row.get("is_pt", "false"))).lower() in {"1", "true", "yes"}:
        return True
    if universe.exclude_suspended and str(market_row.get("is_suspended", "false")).lower() in {"1", "true", "yes"}:
        return True
    if universe.exclude_limit_up_down:
        limit_state = str(market_row.get("limit_state", "")).lower()
        if limit_state in {"limit_up", "limit_down", "up", "down"}:
            return True
    return entry_index < universe.min_prior_trading_days


def _report_weight(prediction: dict[str, Any], config: BacktestConfig) -> float:
    aggregation = config.report_aggregation
    weight = aggregation.default_report_weight
    if aggregation.use_recency_weight:
        weight *= _safe_float(prediction.get("recency_weight"), 1.0) or 1.0
    if aggregation.use_broker_coverage_weight:
        weight *= _safe_float(prediction.get("broker_coverage_weight"), 1.0) or 1.0
    if aggregation.use_analyst_reliability_weight:
        weight *= _safe_float(prediction.get("analyst_reliability_weight"), 1.0) or 1.0
    return max(float(weight), 0.0)


def _aggregate_records(predictions: list[dict[str, Any]], config: BacktestConfig | None = None) -> list[dict[str, Any]]:
    config = config or default_backtest_config()
    stock_codes = {str(row.get("stock_code", "")).strip() for row in predictions if str(row.get("stock_code", "")).strip()}
    market_context = _build_market_context(stock_codes)
    aggregated: dict[tuple[str, str], dict[str, Any]] = {}

    policy_name = config.execution_policy.name
    entry_field = _entry_field_for_policy(policy_name)

    for prediction in predictions:
        stock_code = str(prediction.get("stock_code", "")).strip()
        report_date = str(prediction.get("report_date", "")).strip()
        series_context = market_context.get(stock_code)
        if not series_context or not report_date:
            continue
        dates = series_context["dates"]
        raw_index = bisect_left(dates, report_date)
        if raw_index >= len(dates):
            continue
        index = _entry_index_for_policy(raw_index, policy_name)
        if index >= len(dates):
            continue
        trade_date = dates[index]
        rows = series_context["rows"]
        market_row = rows[index]
        if _is_excluded_by_universe(prediction, market_row, index, config):
            continue
        signals = build_report_signal(prediction, prediction.get("title", ""), prediction.get("summary", ""))
        report_weight = _report_weight(prediction, config)
        if report_weight <= 0.0:
            continue
        key = (trade_date, stock_code)
        bucket = aggregated.setdefault(
            key,
            {
                "trade_date": trade_date,
                "stock_code": stock_code,
                "volatility": series_context["volatility"].get(trade_date, 0.0),
                "report_count": 0,
                "weight_sum": 0.0,
                "signal_sums": {name: 0.0 for name in SIGNAL_NAMES},
                "forward_return_5": _forward_return(rows, index, 5, entry_field=entry_field),
                "forward_return_10": _forward_return(rows, index, 10, entry_field=entry_field),
                "forward_return_20": _forward_return(rows, index, 20, entry_field=entry_field),
            },
        )
        bucket["report_count"] += 1
        bucket["weight_sum"] += report_weight
        for signal_name, signal_value in signals.items():
            bucket["signal_sums"][signal_name] += report_weight * float(signal_value)

    records = []
    for bucket in aggregated.values():
        report_count = max(int(bucket["report_count"]), 1)
        weight_sum = float(bucket.get("weight_sum") or report_count)
        record = {
            "trade_date": bucket["trade_date"],
            "stock_code": bucket["stock_code"],
            "volatility": float(bucket["volatility"]),
            "report_count": report_count,
            "weight_sum": round(weight_sum, 6),
            "forward_return_5": bucket["forward_return_5"],
            "forward_return_10": bucket["forward_return_10"],
            "forward_return_20": bucket["forward_return_20"],
        }
        for signal_name in SIGNAL_NAMES:
            record[signal_name] = round(bucket["signal_sums"][signal_name] / weight_sum, 6)
        records.append(record)
    records.sort(key=lambda item: (item["trade_date"], item["stock_code"]))
    return records


def _group_by_trade_date(records: list[dict[str, Any]]) -> list[tuple[str, list[dict[str, Any]]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[str(record["trade_date"])].append(record)
    return [(trade_date, grouped[trade_date]) for trade_date in sorted(grouped)]


def _rank(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda index: values[index])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        rank_value = (i + j + 2) / 2.0
        for position in range(i, j + 1):
            ranks[order[position]] = rank_value
        i = j + 1
    return ranks


def _cross_sectional_rank_scale(values: list[float]) -> list[float]:
    if not values:
        return []
    if len(values) == 1:
        return [0.0]
    ranks = _rank(values)
    denominator = len(values) - 1
    return [round(((rank - 1.0) / denominator) * 2.0 - 1.0, 6) for rank in ranks]


def _winsorize(values: list[float], lower: float, upper: float) -> list[float]:
    if not values:
        return []
    sorted_values = sorted(values)
    lower_index = min(len(sorted_values) - 1, max(0, int(math.floor(lower * (len(sorted_values) - 1)))))
    upper_index = min(len(sorted_values) - 1, max(0, int(math.ceil(upper * (len(sorted_values) - 1)))))
    lower_value = sorted_values[lower_index]
    upper_value = sorted_values[upper_index]
    return [min(max(value, lower_value), upper_value) for value in values]


def _zscore(values: list[float]) -> list[float]:
    if not values:
        return []
    mean = _average(values)
    std = _stdev(values)
    if std == 0.0:
        return [0.0 for _ in values]
    return [(value - mean) / std for value in values]


def _normalize_signal_rows(rows: list[dict[str, Any]], config: BacktestConfig | None = None) -> list[dict[str, Any]]:
    config = config or default_backtest_config()
    normalized_rows = [dict(row) for row in rows]
    neutralization = config.neutralization
    for signal_name in SIGNAL_NAMES:
        signal_values = [_safe_float(row.get(signal_name)) for row in rows]
        valid_pairs = [(index, value) for index, value in enumerate(signal_values) if value is not None]
        if not valid_pairs:
            continue
        values = [value for _, value in valid_pairs]
        if config.normalization_method in {"winsor_zscore_residual_rank", "winsor_zscore_rank"}:
            values = _winsorize(values, neutralization.winsor_lower, neutralization.winsor_upper)
            if neutralization.standardize_within_date:
                values = _zscore(values)
        scaled_values = _cross_sectional_rank_scale(values)
        for (index, _), scaled_value in zip(valid_pairs, scaled_values):
            normalized_rows[index][signal_name] = scaled_value
    return normalized_rows


def _pearson(x: list[float], y: list[float]) -> float | None:
    if len(x) < 2 or len(y) < 2 or len(x) != len(y):
        return None
    mean_x = _average(x)
    mean_y = _average(y)
    numerator = sum((x_value - mean_x) * (y_value - mean_y) for x_value, y_value in zip(x, y))
    denominator_x = math.sqrt(sum((value - mean_x) ** 2 for value in x))
    denominator_y = math.sqrt(sum((value - mean_y) ** 2 for value in y))
    denominator = denominator_x * denominator_y
    if denominator == 0:
        return None
    return numerator / denominator


def _spearman(x: list[float], y: list[float]) -> float | None:
    return _pearson(_rank(x), _rank(y))


def _mean_rank_ic(rows_by_date: list[tuple[str, list[dict[str, Any]]]], signal_name: str, horizon: int) -> dict[str, Any]:
    per_date_ics: list[float] = []
    return_key = f"forward_return_{horizon}"
    for _, rows in rows_by_date:
        xs: list[float] = []
        ys: list[float] = []
        for row in rows:
            signal = _safe_float(row.get(signal_name))
            forward_return = _safe_float(row.get(return_key))
            if signal is None or forward_return is None:
                continue
            xs.append(signal)
            ys.append(forward_return)
        if len(xs) < 2:
            continue
        ic = _spearman(xs, ys)
        if ic is not None:
            per_date_ics.append(ic)
    ci_lower, ci_upper = _bootstrap_ci(per_date_ics)
    tstat, pval = _newey_west_tstat(per_date_ics)
    return {
        "mean_ic": round(_average(per_date_ics), 6),
        "n_dates": len(per_date_ics),
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "tstat": tstat,
        "pval": pval,
    }


def _portfolio_members(rows: list[dict[str, Any]], signal_name: str, horizon: int, bucket_count: int = 5) -> dict[str, Any] | None:
    return_key = f"forward_return_{horizon}"
    candidates = [
        row
        for row in rows
        if _safe_float(row.get(signal_name)) is not None and _safe_float(row.get(return_key)) is not None
    ]
    if len(candidates) < 4:
        return None
    candidates.sort(key=lambda item: (float(item[signal_name]), str(item["stock_code"])))
    bucket_size = max(1, len(candidates) // bucket_count)
    bucket_size = min(bucket_size, len(candidates) // 2)
    if bucket_size == 0:
        return None
    short_bucket = candidates[:bucket_size]
    long_bucket = candidates[-bucket_size:]
    long_return = _average([float(row[return_key]) for row in long_bucket])
    short_return = _average([float(row[return_key]) for row in short_bucket])
    return {
        "long": {str(row["stock_code"]) for row in long_bucket},
        "short": {str(row["stock_code"]) for row in short_bucket},
        "ls_return": round(long_return - short_return, 6),
    }


def _apply_costs(
    gross_return: float,
    turnover: float,
    cost_bps: float,
) -> float:
    cost = (turnover * cost_bps) / 10000.0
    return round(gross_return - cost, 6)


def _long_short_summary(
    rows_by_date: list[tuple[str, list[dict[str, Any]]]],
    signal_name: str,
    horizon: int = 20,
    *,
    cost_bps: float = 0.0,
    config: BacktestConfig | None = None,
) -> dict[str, Any]:
    previous_portfolio: dict[str, Any] | None = None
    turnover_values: list[float] = []
    returns_series: list[dict[str, Any]] = []

    for trade_date, rows in rows_by_date:
        portfolio = _portfolio_members(rows, signal_name, horizon, bucket_count=config.portfolio_quantiles if config else 5)
        if not portfolio:
            continue
        turnover_value = 0.0
        if previous_portfolio is not None:
            long_overlap = len(previous_portfolio["long"] & portfolio["long"])
            short_overlap = len(previous_portfolio["short"] & portfolio["short"])
            long_turnover = 1.0 - (long_overlap / max(len(portfolio["long"]), 1))
            short_turnover = 1.0 - (short_overlap / max(len(portfolio["short"]), 1))
            turnover_value = (long_turnover + short_turnover) / 2.0
            turnover_values.append(turnover_value)
        gross_return = portfolio["ls_return"]
        net_return = _apply_costs(gross_return, turnover_value, cost_bps)
        returns_series.append({"date": trade_date, "gross_return": gross_return, "net_return": net_return})
        previous_portfolio = portfolio

    gross_returns = [float(item["gross_return"]) for item in returns_series]
    net_returns = [float(item["net_return"]) for item in returns_series]
    gross_sharpe = 0.0
    net_sharpe = 0.0
    gross_volatility = _stdev(gross_returns)
    net_volatility = _stdev(net_returns)
    if gross_volatility > 0:
        gross_sharpe = round((_average(gross_returns) / gross_volatility) * math.sqrt(252.0 / float(max(horizon, 1))), 4)
    if net_volatility > 0:
        net_sharpe = round((_average(net_returns) / net_volatility) * math.sqrt(252.0 / float(max(horizon, 1))), 4)

    ci_lower, ci_upper = _bootstrap_ci(net_returns)
    tstat, pval = _newey_west_tstat(net_returns)

    cumulative_gross = 0.0
    cumulative_net = 0.0
    cumulative_series = []
    for item in returns_series:
        cumulative_gross += float(item["gross_return"])
        cumulative_net += float(item["net_return"])
        cumulative_series.append({
            "date": item["date"],
            "cumulative_return": round(cumulative_net, 6),
            "gross_cumulative_return": round(cumulative_gross, 6),
        })

    return {
        "gross_sharpe": gross_sharpe,
        "net_sharpe": net_sharpe,
        "sharpe": net_sharpe,
        "turnover": round(_average(turnover_values), 4) if turnover_values else 0.0,
        "return_series": returns_series,
        "cumulative_series": cumulative_series,
        "observation_count": len(returns_series),
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "tstat": tstat,
        "pval": pval,
    }


def _assign_quintiles(values: list[float], bucket_count: int = 5) -> list[int]:
    if not values:
        return []
    order = sorted(range(len(values)), key=lambda index: values[index])
    buckets = [1] * len(values)
    for rank_position, original_index in enumerate(order):
        buckets[original_index] = min(bucket_count, int(rank_position * bucket_count / len(values)) + 1)
    return buckets


def _regime_heatmap(rows_by_date: list[tuple[str, list[dict[str, Any]]]], signal_name: str = PRIMARY_SIGNAL) -> list[dict[str, Any]]:
    grouped_scores: dict[tuple[int, int], list[float]] = defaultdict(list)
    for horizon in FORWARD_HORIZONS:
        return_key = f"forward_return_{horizon}"
        for _, rows in rows_by_date:
            candidates = [
                row
                for row in rows
                if _safe_float(row.get(signal_name)) is not None
                and _safe_float(row.get(return_key)) is not None
                and _safe_float(row.get("volatility")) is not None
            ]
            if len(candidates) < 5:
                continue
            quintiles = _assign_quintiles([float(row["volatility"]) for row in candidates])
            for quintile in range(1, 6):
                subset = [row for row, row_quintile in zip(candidates, quintiles) if row_quintile == quintile]
                if len(subset) < 2:
                    continue
                ic = _spearman(
                    [float(row[signal_name]) for row in subset],
                    [float(row[return_key]) for row in subset],
                )
                if ic is not None:
                    grouped_scores[(quintile, horizon)].append(ic)
    results = []
    for quintile in range(1, 6):
        for horizon in FORWARD_HORIZONS:
            values = grouped_scores.get((quintile, horizon), [])
            results.append(
                {
                    "volatility_quintile": quintile,
                    "horizon": horizon,
                    "mean_rank_ic": round(_average(values), 6) if values else 0.0,
                    "observation_count": len(values),
                }
            )
    return results


def _curve_rows(curves_by_signal: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    all_dates = sorted({item["date"] for rows in curves_by_signal.values() for item in rows})
    curve_lookup = {
        signal_name: {item["date"]: item["cumulative_return"] for item in rows}
        for signal_name, rows in curves_by_signal.items()
    }
    rows = []
    for trade_date in all_dates:
        row = {"date": trade_date}
        for signal_name in SIGNAL_NAMES:
            row[signal_name] = curve_lookup.get(signal_name, {}).get(trade_date)
        rows.append(row)
    return rows


def _cost_stress_rows(
    rows_by_date: list[tuple[str, list[dict[str, Any]]]],
    signal_name: str,
    config: BacktestConfig,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    stress = config.cost_stress
    for fee_bps in stress.one_way_fee_bps:
        for slippage_bps in stress.slippage_bps:
            impact_bps = stress.participation_rate * 100.0 * stress.impact_bps_per_1pct_participation
            total_bps = fee_bps + slippage_bps + impact_bps
            summary = _long_short_summary(
                rows_by_date,
                signal_name,
                horizon=config.portfolio.holding_horizon_days,
                cost_bps=total_bps,
                config=config,
            )
            rows.append(
                {
                    "signal": signal_name,
                    "fee_bps": fee_bps,
                    "slippage_bps": slippage_bps,
                    "impact_bps": round(impact_bps, 6),
                    "total_cost_bps": round(total_bps, 6),
                    "net_sharpe@20": summary["net_sharpe"],
                    "turnover": summary["turnover"],
                    "observation_count": summary["observation_count"],
                }
            )
    return rows


def run_cross_sectional_backtest(prediction_path: Path, config: BacktestConfig | None = None) -> dict[str, Any]:
    config = config or default_backtest_config()
    if not prediction_path.exists():
        return {"status": "missing_prediction_file", "prediction_path": str(prediction_path), "config": config.to_dict()}
    predictions = _load_prediction_rows(prediction_path)
    if not predictions:
        return {"status": "empty_prediction_file", "prediction_path": str(prediction_path), "config": config.to_dict()}
    market_prices_path = resolve_market_prices_path()
    if not market_prices_path.exists():
        return {
            "status": "missing_market_data",
            "prediction_path": str(prediction_path),
            "market_prices_path": str(market_prices_path),
            "config": config.to_dict(),
        }

    records = _aggregate_records(predictions, config=config)
    if not records:
        return {
            "status": "no_aligned_records",
            "prediction_path": str(prediction_path),
            "market_prices_path": str(market_prices_path),
            "config": config.to_dict(),
        }

    rows_by_date = [
        (trade_date, _normalize_signal_rows(rows, config=config))
        for trade_date, rows in _group_by_trade_date(records)
    ]
    signal_metrics: dict[str, dict[str, Any]] = {}
    curves_by_signal: dict[str, list[dict[str, Any]]] = {}

    total_cost_bps = (
        config.cost_model.commission_bps
        + config.cost_model.sell_tax_bps
        + config.cost_model.slippage_bps
    )

    for signal_name in SIGNAL_NAMES:
        long_short = _long_short_summary(rows_by_date, signal_name, horizon=20, cost_bps=total_cost_bps, config=config)
        ic5 = _mean_rank_ic(rows_by_date, signal_name, 5)
        ic10 = _mean_rank_ic(rows_by_date, signal_name, 10)
        ic20 = _mean_rank_ic(rows_by_date, signal_name, 20)
        metrics = {
            "mean_rank_ic@5": ic5["mean_ic"],
            "mean_rank_ic@10": ic10["mean_ic"],
            "mean_rank_ic@20": ic20["mean_ic"],
            "ic@20_ci_lower": ic20["ci_lower"],
            "ic@20_ci_upper": ic20["ci_upper"],
            "ic@20_tstat": ic20["tstat"],
            "ic@20_pval": ic20["pval"],
            "ls_sharpe@20": long_short["net_sharpe"],
            "ls_gross_sharpe@20": long_short["gross_sharpe"],
            "ls_net_sharpe@20": long_short["net_sharpe"],
            "ls_tstat": long_short["tstat"],
            "ls_pval": long_short["pval"],
            "ls_ci_lower": long_short["ci_lower"],
            "ls_ci_upper": long_short["ci_upper"],
            "turnover": long_short["turnover"],
            "portfolio_observation_count": long_short["observation_count"],
            "cost_bps": total_cost_bps,
        }
        signal_metrics[signal_name] = metrics
        curves_by_signal[signal_name] = long_short["cumulative_series"]

    headline = signal_metrics.get(config.primary_signal, {})
    diagnostics = diagnostic_summary(records, min_dates=config.min_trade_dates)
    return {
        "status": "ok",
        "prediction_path": str(prediction_path),
        "report_count": len(predictions),
        "aligned_stock_day_count": len(records),
        "trade_date_count": len(rows_by_date),
        "config": config.to_dict(),
        "diagnostics": diagnostics,
        "signal_metrics": signal_metrics,
        "curve_rows": _curve_rows(curves_by_signal),
        "regime_heatmap_rows": _regime_heatmap(rows_by_date, config.primary_signal),
        "cost_stress_rows": _cost_stress_rows(rows_by_date, config.primary_signal, config),
        "headline_metrics": {
            "mean_rank_ic@20": headline.get("mean_rank_ic@20", 0.0),
            "ls_sharpe@20": headline.get("ls_sharpe@20", 0.0),
            "ls_gross_sharpe@20": headline.get("ls_gross_sharpe@20", 0.0),
            "ls_net_sharpe@20": headline.get("ls_net_sharpe@20", 0.0),
            "ls_tstat": headline.get("ls_tstat", 0.0),
            "ls_pval": headline.get("ls_pval", 1.0),
            "turnover": headline.get("turnover", 0.0),
        },
    }
