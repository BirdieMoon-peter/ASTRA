import unittest
from pathlib import Path
from unittest.mock import patch

from astra.finance.backtest_cross_sectional import (
    PRIMARY_SIGNAL,
    SIGNAL_NAMES,
    _bootstrap_ci,
    _cross_sectional_rank_scale,
    _newey_west_tstat,
    _normalize_signal_rows,
    _portfolio_members,
    diagnostic_summary,
    run_cross_sectional_backtest,
)


class BootstrapCITest(unittest.TestCase):
    def test_empty_values_return_zero_zero(self) -> None:
        self.assertEqual(_bootstrap_ci([]), (0.0, 0.0))

    def test_single_value_returns_that_value(self) -> None:
        lo, hi = _bootstrap_ci([5.0])
        self.assertAlmostEqual(lo, 5.0, places=4)
        self.assertAlmostEqual(hi, 5.0, places=4)

    def test_ci_contains_sample_mean(self) -> None:
        values = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
        lo, hi = _bootstrap_ci(values)
        mean = sum(values) / len(values)
        self.assertLessEqual(lo, mean)
        self.assertGreaterEqual(hi, mean)

    def test_wider_ci_with_lower_confidence(self) -> None:
        values = list(range(1, 51))
        values_f = [float(v) for v in values]
        lo_95, hi_95 = _bootstrap_ci(values_f, ci=0.95)
        lo_80, hi_80 = _bootstrap_ci(values_f, ci=0.80)
        self.assertGreaterEqual(hi_95 - lo_95, hi_80 - lo_80)


class NeweyWestTStatTest(unittest.TestCase):
    def test_empty_returns_zero_tstat(self) -> None:
        tstat, pval = _newey_west_tstat([])
        self.assertEqual(tstat, 0.0)
        self.assertEqual(pval, 1.0)

    def test_single_value_returns_zero_tstat(self) -> None:
        tstat, pval = _newey_west_tstat([3.0])
        self.assertEqual(tstat, 0.0)
        self.assertEqual(pval, 1.0)

    def test_strong_positive_mean_yields_positive_tstat(self) -> None:
        values = [1.0] * 100
        tstat, pval = _newey_west_tstat(values)
        self.assertGreater(tstat, 0)
        self.assertLess(pval, 0.05)

    def test_zero_mean_yields_zero_tstat(self) -> None:
        values = [1.0, -1.0] * 50
        tstat, _pval = _newey_west_tstat(values)
        self.assertAlmostEqual(tstat, 0.0, places=2)


class DiagnosticSummaryTest(unittest.TestCase):
    def test_empty_records(self) -> None:
        result = diagnostic_summary([])
        self.assertEqual(result["n_trade_dates"], 0)
        self.assertEqual(result["n_total_observations"], 0)
        self.assertFalse(result["sample_adequate"])
        self.assertIn("Only 0", result["warning"])

    def test_adequate_sample(self) -> None:
        records = [{"trade_date": f"2024-01-{d:02d}", "stock_code": f"S{i}"}
                   for d in range(1, 22) for i in range(3)]
        result = diagnostic_summary(records, min_dates=20)
        self.assertEqual(result["n_trade_dates"], 21)
        self.assertTrue(result["sample_adequate"])
        self.assertEqual(result["warning"], "")
        self.assertAlmostEqual(result["n_stocks_per_date_mean"], 3.0)

    def test_inadequate_sample(self) -> None:
        records = [{"trade_date": "2024-01-01"}]
        result = diagnostic_summary(records, min_dates=20)
        self.assertFalse(result["sample_adequate"])
        self.assertIn("Only 1", result["warning"])


class CrossSectionalBacktestTest(unittest.TestCase):
    def _signal_row(self, stock_code: str, signal_value: object, forward_return: float = 0.1) -> dict[str, object]:
        row: dict[str, object] = {
            "trade_date": "2024-01-02",
            "stock_code": stock_code,
            "forward_return_20": forward_return,
        }
        for signal_name in SIGNAL_NAMES:
            row[signal_name] = signal_value
        return row

    def test_cross_sectional_rank_scale_maps_sorted_values_to_minus_one_one(self) -> None:
        self.assertEqual(
            _cross_sectional_rank_scale([1, 2, 3, 4]),
            [-1.0, -0.333333, 0.333333, 1.0],
        )

    def test_cross_sectional_rank_scale_returns_empty_list_for_empty_input(self) -> None:
        self.assertEqual(_cross_sectional_rank_scale([]), [])

    def test_cross_sectional_rank_scale_returns_zero_for_singleton(self) -> None:
        self.assertEqual(_cross_sectional_rank_scale([42.0]), [0.0])

    def test_cross_sectional_rank_scale_assigns_zero_when_all_values_tie(self) -> None:
        self.assertEqual(_cross_sectional_rank_scale([5.0, 5.0, 5.0]), [0.0, 0.0, 0.0])

    def test_cross_sectional_rank_scale_assigns_shared_ranks_for_partial_ties(self) -> None:
        self.assertEqual(
            _cross_sectional_rank_scale([1.0, 1.0, 3.0, 4.0]),
            [-0.666667, -0.666667, 0.333333, 1.0],
        )

    def test_primary_signal_prefers_quality_weighted_signal(self) -> None:
        self.assertEqual(PRIMARY_SIGNAL, "astra_quality_weighted")

    def test_normalize_signal_rows_rank_scales_each_signal_by_trade_date_slice(self) -> None:
        rows = [
            self._signal_row("000001", 1.0),
            self._signal_row("000002", 2.0),
            self._signal_row("000003", 3.0),
        ]

        normalized = _normalize_signal_rows(rows)

        self.assertIn("astra_finance_plus_gap", SIGNAL_NAMES)
        for signal_name in SIGNAL_NAMES:
            self.assertEqual(
                [row[signal_name] for row in normalized],
                [-1.0, 0.0, 1.0],
            )

        self.assertEqual(rows[0]["astra_finance_plus_gap"], 1.0)
        self.assertEqual(rows[0]["astra_quality_weighted"], 1.0)

    def test_normalize_signal_rows_leaves_missing_and_non_numeric_values_unchanged(self) -> None:
        rows = [
            self._signal_row("000001", 1.0),
            self._signal_row("000002", "not-a-number"),
            self._signal_row("000003", None),
            self._signal_row("000004", 3.0),
        ]

        normalized = _normalize_signal_rows(rows)

        for signal_name in SIGNAL_NAMES:
            self.assertEqual(normalized[0][signal_name], -1.0)
            self.assertEqual(normalized[1][signal_name], "not-a-number")
            self.assertIsNone(normalized[2][signal_name])
            self.assertEqual(normalized[3][signal_name], 1.0)

    def test_portfolio_members_breaks_signal_ties_by_stock_code(self) -> None:
        rows = [
            self._signal_row("000004", 0.0, 0.04),
            self._signal_row("000002", 0.0, 0.02),
            self._signal_row("000003", 1.0, 0.03),
            self._signal_row("000001", 1.0, 0.01),
        ]

        portfolio = _portfolio_members(rows, "astra_quality_weighted", 20, bucket_count=2)

        self.assertEqual(portfolio["short"], {"000002", "000004"})
        self.assertEqual(portfolio["long"], {"000001", "000003"})
        self.assertEqual(portfolio["ls_return"], -0.01)

    @patch("astra.finance.backtest_cross_sectional.resolve_market_prices_path", return_value=Path("/tmp/market.csv"))
    @patch("astra.finance.backtest_cross_sectional._regime_heatmap", return_value=[])
    @patch(
        "astra.finance.backtest_cross_sectional._curve_rows",
        side_effect=lambda curves_by_signal: [{"date": "2024-01-02", **{name: 0.0 for name in curves_by_signal}}],
    )
    @patch(
        "astra.finance.backtest_cross_sectional._long_short_summary",
        side_effect=lambda rows_by_date, signal_name, horizon=20, cost_bps=0.0: {
            "sharpe": 0.42 if signal_name == "astra_quality_weighted" else -0.1,
            "gross_sharpe": 0.52 if signal_name == "astra_quality_weighted" else -0.05,
            "net_sharpe": 0.42 if signal_name == "astra_quality_weighted" else -0.1,
            "turnover": 0.15 if signal_name == "astra_quality_weighted" else 0.3,
            "observation_count": 2,
            "ci_lower": -0.01,
            "ci_upper": 0.05,
            "tstat": 1.1,
            "pval": 0.27,
            "cumulative_series": [{"date": "2024-01-02", "cumulative_return": 0.07 if signal_name == "astra_quality_weighted" else 0.01}],
        },
    )
    @patch(
        "astra.finance.backtest_cross_sectional._mean_rank_ic",
        side_effect=lambda rows_by_date, signal_name, horizon: {
            "mean_ic": {("astra_quality_weighted", 5): 0.11, ("astra_quality_weighted", 10): 0.22, ("astra_quality_weighted", 20): 0.33}.get(
                (signal_name, horizon), -0.05
            ),
            "n_dates": 1,
            "ci_lower": -0.10,
            "ci_upper": 0.50,
            "tstat": 1.5,
            "pval": 0.13,
        },
    )
    @patch("astra.finance.backtest_cross_sectional._group_by_trade_date", return_value=[("2024-01-02", [{"stub": True}])])
    @patch("astra.finance.backtest_cross_sectional._normalize_signal_rows", side_effect=lambda rows: rows)
    @patch("astra.finance.backtest_cross_sectional._aggregate_records", return_value=[{"trade_date": "2024-01-02", "stock_code": "000001"}])
    @patch("astra.finance.backtest_cross_sectional._load_prediction_rows", return_value=[{"stock_code": "000001"}])
    @patch("pathlib.Path.exists", return_value=True)
    def test_run_cross_sectional_backtest_surfaces_primary_signal_outputs(
        self,
        _path_exists_mock,
        _load_predictions_mock,
        _aggregate_records_mock,
        _normalize_rows_mock,
        _group_by_trade_date_mock,
        _mean_rank_ic_mock,
        _long_short_summary_mock,
        _curve_rows_mock,
        _regime_heatmap_mock,
        _resolve_market_prices_path_mock,
    ) -> None:
        result = run_cross_sectional_backtest(Path("/tmp/predictions.jsonl"))

        self.assertEqual(result["status"], "ok")
        self.assertIn("astra_finance_plus_gap", result["signal_metrics"])
        afpg = result["signal_metrics"]["astra_finance_plus_gap"]
        self.assertEqual(afpg["mean_rank_ic@5"], -0.05)
        self.assertEqual(afpg["mean_rank_ic@10"], -0.05)
        self.assertEqual(afpg["mean_rank_ic@20"], -0.05)
        self.assertEqual(afpg["ls_sharpe@20"], -0.1)
        self.assertEqual(afpg["turnover"], 0.3)
        self.assertEqual(afpg["portfolio_observation_count"], 2)
        self.assertIn("ic@20_ci_lower", afpg)
        self.assertIn("ic@20_ci_upper", afpg)
        self.assertIn("ls_ci_lower", afpg)
        self.assertIn("ls_ci_upper", afpg)
        self.assertEqual(
            result["headline_metrics"],
            {
                "mean_rank_ic@20": 0.33,
                "ls_sharpe@20": 0.42,
                "ls_gross_sharpe@20": 0.52,
                "ls_net_sharpe@20": 0.42,
                "ls_tstat": 1.1,
                "ls_pval": 0.27,
                "turnover": 0.15,
            },
        )
        self.assertEqual(result["curve_rows"], [{"date": "2024-01-02", **{name: 0.0 for name in SIGNAL_NAMES}}])
        self.assertIn("astra_finance_plus_gap", result["curve_rows"][0])


if __name__ == "__main__":
    unittest.main()
