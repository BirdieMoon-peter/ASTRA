import json
import tempfile
import unittest
from pathlib import Path

from astra.paper.export_results import export_paper_artifacts


class PaperExportTests(unittest.TestCase):
    def test_export_paper_artifacts_writes_csv_and_latex(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "eval").mkdir()
            (root / "backtest").mkdir()
            (root / "cases").mkdir()
            (root / "paper").mkdir()
            (root / "paper" / "latex").mkdir()

            (root / "eval" / "nlp_metrics.json").write_text(
                json.dumps({
                    "astra_mvp": {
                        "fundamental_sentiment_macro_f1": 0.61,
                        "strategic_optimism_macro_f1": 0.62,
                        "evidence_f1": 0.63,
                        "ece": 0.07,
                    }
                }),
                encoding="utf-8",
            )
            (root / "eval" / "phenomena_metrics.json").write_text(
                json.dumps({"astra_mvp": {"per_label": {"hedged_downside": {"accuracy": 0.8}}}}),
                encoding="utf-8",
            )
            (root / "eval" / "calibration_bins.json").write_text(
                json.dumps({"astra_mvp": [{"bin": 0.5, "count": 2, "accuracy": 0.5}]}),
                encoding="utf-8",
            )
            (root / "backtest" / "finance_metrics.json").write_text(
                json.dumps({
                    "signal_metrics": {
                        "astra_uncertainty_gated": {
                            "mean_rank_ic@5": 0.01,
                            "mean_rank_ic@10": 0.02,
                            "mean_rank_ic@20": 0.03,
                            "ls_sharpe@20": 1.4,
                            "turnover": 0.5,
                        }
                    },
                    "ablation_metrics": {},
                }),
                encoding="utf-8",
            )
            (root / "backtest" / "backtest_curve.csv").write_text(
                "date,astra_uncertainty_gated\n2025-01-02,0.01\n",
                encoding="utf-8",
            )
            (root / "backtest" / "regime_heatmap.csv").write_text(
                "volatility_quintile,horizon,mean_rank_ic,observation_count\n1,20,0.02,10\n",
                encoding="utf-8",
            )
            (root / "cases" / "case_studies.jsonl").write_text(
                json.dumps({
                    "title": "原始标题",
                    "summary": "原始摘要",
                    "neutralized_text": "中性改写",
                    "sentiment_score": 0.3,
                    "counterfactual_sentiment_score": -0.1,
                    "strategic_optimism_gap": 0.4,
                    "reasoning_summary": "解释",
                }) + "\n",
                encoding="utf-8",
            )

            export_paper_artifacts(outputs_root=root)

            self.assertTrue((root / "paper" / "table_main_nlp.csv").exists())
            self.assertTrue((root / "paper" / "latex" / "tab_main_nlp.tex").exists())
            self.assertTrue((root / "paper" / "figure_backtest.csv").exists())

            case_latex = (root / "paper" / "latex" / "tab_case.tex").read_text(encoding="utf-8")
            self.assertIn("原始标题", case_latex)
            self.assertIn("原始摘要", case_latex)
            self.assertIn("中性改写", case_latex)


if __name__ == "__main__":
    unittest.main()
