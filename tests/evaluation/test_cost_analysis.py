from __future__ import annotations

import csv
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from astra.evaluation.cost_analysis import PipelineTimer


class PipelineTimerTest(unittest.TestCase):
    def test_empty_summarize(self) -> None:
        timer = PipelineTimer()
        summary = timer.summarize()
        self.assertEqual(summary["total_cost"], 0.0)
        self.assertEqual(summary["report_count"], 0)
        self.assertEqual(summary["stages"], {})

    def test_single_stage_latency_and_tokens(self) -> None:
        timer = PipelineTimer()
        timer.begin_report()
        # Simulate a stage that takes ~0 seconds (mocked by direct recording)
        timer._record_latency("retrieval", 1.5)
        timer._attempts["retrieval"] = 1
        timer._record_tokens("retrieval", 500, 100)

        summary = timer.summarize()
        self.assertIn("retrieval", summary["stages"])
        stage = summary["stages"]["retrieval"]
        self.assertAlmostEqual(stage["mean_latency_s"], 1.5, places=4)
        self.assertAlmostEqual(stage["mean_input_tokens"], 500.0)
        self.assertAlmostEqual(stage["mean_output_tokens"], 100.0)
        self.assertEqual(stage["failure_rate"], 0.0)
        # Cost: 500/1000 * 0.001 + 100/1000 * 0.003 = 0.0005 + 0.0003 = 0.0008
        self.assertAlmostEqual(stage["total_cost"], 0.0008, places=6)

    def test_failure_rate_tracking(self) -> None:
        timer = PipelineTimer()
        timer._attempts["decomposition"] = 10
        timer._failures["decomposition"] = 3

        summary = timer.summarize()
        self.assertAlmostEqual(summary["stages"]["decomposition"]["failure_rate"], 0.3, places=4)

    def test_agentic_overhead(self) -> None:
        timer = PipelineTimer()
        timer.begin_report()
        timer._attempts["retrieval"] = 1
        timer._record_tokens("retrieval", 1000, 500)
        timer._attempts["direct_llm"] = 1
        timer._record_tokens("direct_llm", 2000, 1000)

        summary = timer.summarize()
        direct_cost = summary["direct_llm_only_cost"]
        total_cost = summary["total_cost"]
        self.assertAlmostEqual(summary["agentic_overhead_cost"], total_cost - direct_cost, places=6)

    def test_context_manager_stage(self) -> None:
        timer = PipelineTimer()
        timer.begin_report()
        with timer.stage("neutralization") as s:
            s.record_tokens(input_tokens=200, output_tokens=50)

        summary = timer.summarize()
        self.assertIn("neutralization", summary["stages"])
        self.assertEqual(summary["stages"]["neutralization"]["attempts"], 1)
        self.assertGreaterEqual(summary["stages"]["neutralization"]["mean_latency_s"], 0.0)

    def test_context_manager_failure_tracking(self) -> None:
        timer = PipelineTimer()
        timer.begin_report()
        try:
            with timer.stage("verification"):
                raise ValueError("test error")
        except ValueError:
            pass

        summary = timer.summarize()
        self.assertEqual(summary["stages"]["verification"]["failures"], 1)
        self.assertAlmostEqual(summary["stages"]["verification"]["failure_rate"], 1.0)

    def test_export_table(self) -> None:
        timer = PipelineTimer()
        timer.begin_report()
        timer._attempts["retrieval"] = 2
        timer._record_latency("retrieval", 0.5)
        timer._record_latency("retrieval", 0.7)
        timer._record_tokens("retrieval", 100, 50)
        timer._record_tokens("retrieval", 200, 60)

        with TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "cost.csv"
            timer.export_table(csv_path)

            with csv_path.open("r", encoding="utf-8") as fh:
                reader = list(csv.DictReader(fh))

            stage_names = [row["stage"] for row in reader]
            self.assertIn("retrieval", stage_names)
            self.assertIn("TOTAL", stage_names)
            self.assertIn("per_report", stage_names)
            self.assertIn("direct_llm_only", stage_names)
            self.assertIn("agentic_overhead", stage_names)

    def test_cost_per_report_divides_by_report_count(self) -> None:
        timer = PipelineTimer()
        timer.begin_report()
        timer.begin_report()
        timer._attempts["cot_llm"] = 2
        timer._record_tokens("cot_llm", 1000, 500)
        timer._record_tokens("cot_llm", 1000, 500)

        summary = timer.summarize()
        self.assertEqual(summary["report_count"], 2)
        self.assertAlmostEqual(summary["cost_per_report"], summary["total_cost"] / 2, places=6)
