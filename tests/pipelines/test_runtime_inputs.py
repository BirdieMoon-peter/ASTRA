import csv
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import MagicMock, patch

from astra.config import task_schema
from astra.finance.backtest_cross_sectional import SIGNAL_NAMES
from astra.pipelines import run_astra_inference, run_finance_eval
from astra.retrieval.history_retriever import HistoryRetriever


class RuntimeInputsTest(unittest.TestCase):
    def test_safe_llm_prediction_falls_back_to_rule_based_output(self) -> None:
        llm_client = MagicMock()
        with patch("astra.pipelines.run_astra_inference._direct_llm_prediction", side_effect=RuntimeError("boom")):
            result = run_astra_inference._safe_llm_prediction(llm_client, "标题", "摘要短期承压", style="react")

        self.assertEqual(result["uncertainty"], 0.95)
        self.assertIn("fallback", result["reasoning_summary"])
        self.assertIn("fundamental_sentiment", result)

    def test_safe_component_wrappers_return_fallback_payloads(self) -> None:
        decomposer = MagicMock()
        decomposer.run.side_effect = RuntimeError("boom")
        neutralizer = MagicMock()
        neutralizer.run.side_effect = RuntimeError("boom")
        verifier = MagicMock()
        verifier.run.side_effect = RuntimeError("boom")

        decomposition = run_astra_inference._safe_decomposition(decomposer, title="标题", summary="摘要", history_context="历史")
        neutralization = run_astra_inference._safe_neutralization(neutralizer, title="标题", summary="摘要", decomposition={})
        verification = run_astra_inference._safe_verification(verifier, title="标题", summary="摘要", neutralized_text="摘要")

        self.assertEqual(decomposition["fallback"], "decomposer_error")
        self.assertEqual(neutralization["fallback"], "neutralizer_error")
        self.assertEqual(verification["issues"], ["verifier_error_fallback"])

    def test_load_rows_uses_clean_reports_split_by_default_even_when_experiment_split_exists(self) -> None:
        with TemporaryDirectory() as temp_dir:
            clean_reports_path = Path(temp_dir) / "reports_clean.csv"
            clean_reports_path.write_text(
                "\n".join(
                    [
                        "report_id,report_date,split,stock_code,title,summary",
                        "clean-dev,2025-01-02,dev,000001,Clean title,Clean summary",
                        "clean-test,2026-01-02,test,000002,Other title,Other summary",
                    ]
                ),
                encoding="utf-8",
            )
            reports_dev_path = Path(temp_dir) / "reports_dev.csv"
            reports_dev_path.write_text(
                "\n".join(
                    [
                        "report_id,report_date,split,stock_code,title,summary",
                        "experiment-dev,2025-01-02,dev,000001,Experiment title,Experiment summary",
                    ]
                ),
                encoding="utf-8",
            )

            with patch.object(task_schema, "CLEAN_REPORTS_PATH", clean_reports_path), patch.object(
                task_schema,
                "EXPERIMENT_DEV_PATH",
                reports_dev_path,
            ):
                rows = run_astra_inference._load_rows(limit=None, split="dev")

        self.assertEqual([row["report_id"] for row in rows], ["clean-dev"])

    def test_load_rows_uses_experiment_split_when_explicitly_requested(self) -> None:
        with TemporaryDirectory() as temp_dir:
            clean_reports_path = Path(temp_dir) / "reports_clean.csv"
            clean_reports_path.write_text(
                "\n".join(
                    [
                        "report_id,report_date,split,stock_code,title,summary",
                        "clean-dev,2025-01-02,dev,000001,Clean title,Clean summary",
                    ]
                ),
                encoding="utf-8",
            )
            reports_dev_path = Path(temp_dir) / "reports_dev.csv"
            reports_dev_path.write_text(
                "\n".join(
                    [
                        "report_id,report_date,split,stock_code,title,summary",
                        "experiment-dev,2025-01-02,dev,000001,Experiment title,Experiment summary",
                    ]
                ),
                encoding="utf-8",
            )

            with patch.object(task_schema, "CLEAN_REPORTS_PATH", clean_reports_path), patch.object(
                task_schema,
                "EXPERIMENT_DEV_PATH",
                reports_dev_path,
            ):
                rows = run_astra_inference._load_rows(limit=None, split="dev", prefer_experiment_split=True)

        self.assertEqual([row["report_id"] for row in rows], ["experiment-dev"])

    def test_load_rows_uses_explicit_input_path_without_split_filtering(self) -> None:
        with TemporaryDirectory() as temp_dir:
            input_path = Path(temp_dir) / "aligned.csv"
            input_path.write_text(
                "\n".join(
                    [
                        "report_id,report_date,split,stock_code,title,summary",
                        "gold-test-1,2026-01-02,test,000001,Title 1,Summary 1",
                        "gold-test-2,2026-01-03,test,000002,Title 2,Summary 2",
                    ]
                ),
                encoding="utf-8",
            )

            rows = run_astra_inference._load_rows(limit=None, split="test", input_path=input_path)

        self.assertEqual([row["report_id"] for row in rows], ["gold-test-1", "gold-test-2"])

    def test_history_retriever_prefers_existing_experiment_master_before_clean_reports(self) -> None:
        with TemporaryDirectory() as temp_dir:
            clean_reports_path = Path(temp_dir) / "reports_clean.csv"
            reports_master_path = Path(temp_dir) / "reports_experiment_master.csv"
            reports_master_path.write_text(
                "\n".join(
                    [
                        "report_id,report_date,split,stock_code,title,summary",
                        "r1,2025-01-01,dev,000001,Earlier,Earlier summary",
                        "r2,2025-01-02,dev,000001,Later,Later summary",
                    ]
                ),
                encoding="utf-8",
            )

            with patch.object(task_schema, "CLEAN_REPORTS_PATH", clean_reports_path), patch.object(
                task_schema,
                "REPORTS_EXPERIMENT_MASTER_PATH",
                reports_master_path,
            ):
                retriever = HistoryRetriever()

        rows = retriever.retrieve("000001", "2025-01-02")
        self.assertEqual([row["report_id"] for row in rows], ["r1"])

    def test_run_finance_eval_curve_csv_uses_signal_names_header_including_astra_finance_plus_gap(self) -> None:
        with TemporaryDirectory() as temp_dir:
            outputs_root = Path(temp_dir)
            curve_path = outputs_root / "backtest" / "backtest_curve.csv"
            metrics_path = outputs_root / "backtest" / "finance_metrics.json"
            regime_path = outputs_root / "backtest" / "regime_heatmap.csv"
            fake_result = {
                "status": "ok",
                "prediction_path": str(outputs_root / "predictions" / "astra_mvp.jsonl"),
                "report_count": 1,
                "aligned_stock_day_count": 1,
                "trade_date_count": 1,
                "signal_metrics": {name: {"mean_rank_ic@20": 0.0, "ls_sharpe@20": 0.0, "turnover": 0.0} for name in SIGNAL_NAMES},
                "curve_rows": [{"date": "2026-01-01", **{name: 0.0 for name in SIGNAL_NAMES}}],
                "regime_heatmap_rows": [{"volatility_quintile": 1, "horizon": 20, "mean_rank_ic": 0.0, "observation_count": 1}],
                "headline_metrics": {"mean_rank_ic@20": 0.11, "ls_sharpe@20": 0.22, "turnover": 0.33},
            }

            with patch("astra.pipelines.run_finance_eval.run_cross_sectional_backtest", return_value=fake_result):
                run_finance_eval.main(outputs_root=outputs_root)

            with curve_path.open("r", encoding="utf-8", newline="") as handle:
                reader = csv.DictReader(handle)
                self.assertEqual(reader.fieldnames, ["date", *SIGNAL_NAMES])
                first_row = next(reader)
            self.assertIn("astra_finance_plus_gap", first_row)
            self.assertEqual(first_row["astra_finance_plus_gap"], "0.0")
            self.assertTrue(metrics_path.exists())
            self.assertTrue(regime_path.exists())


if __name__ == "__main__":
    unittest.main()
