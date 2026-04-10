from __future__ import annotations

import csv
import time
from collections import defaultdict
from pathlib import Path
from typing import Any


PIPELINE_STAGES = (
    "retrieval",
    "decomposition",
    "neutralization",
    "verification",
    "direct_llm",
    "cot_llm",
    "react_llm",
)

DEFAULT_INPUT_PRICE_PER_1K = 0.001
DEFAULT_OUTPUT_PRICE_PER_1K = 0.003


class _StageTimer:
    """Context manager returned by ``PipelineTimer.stage()``."""

    def __init__(self, tracker: PipelineTimer, stage: str) -> None:
        self._tracker = tracker
        self._stage = stage
        self._start: float | None = None

    def __enter__(self) -> _StageTimer:
        self._start = time.monotonic()
        return self

    def __exit__(self, exc_type: type | None, exc_val: BaseException | None, exc_tb: Any) -> None:
        elapsed = time.monotonic() - (self._start or time.monotonic())
        self._tracker._record_latency(self._stage, elapsed)
        if exc_type is not None:
            self._tracker._record_failure(self._stage)

    # Convenience: let the caller record token counts within the block
    def record_tokens(self, input_tokens: int = 0, output_tokens: int = 0) -> None:
        self._tracker._record_tokens(self._stage, input_tokens, output_tokens)


class PipelineTimer:
    """Track per-stage latency, token usage, and failures for the ASTRA pipeline.

    Usage::

        timer = PipelineTimer()
        with timer.stage("retrieval") as s:
            ...
            s.record_tokens(input_tokens=500, output_tokens=120)
        summary = timer.summarize()
    """

    def __init__(
        self,
        input_price_per_1k: float = DEFAULT_INPUT_PRICE_PER_1K,
        output_price_per_1k: float = DEFAULT_OUTPUT_PRICE_PER_1K,
    ) -> None:
        self.input_price_per_1k = input_price_per_1k
        self.output_price_per_1k = output_price_per_1k

        # Internal accumulators keyed by stage name.
        self._latencies: dict[str, list[float]] = defaultdict(list)
        self._input_tokens: dict[str, list[int]] = defaultdict(list)
        self._output_tokens: dict[str, list[int]] = defaultdict(list)
        self._failures: dict[str, int] = defaultdict(int)
        self._attempts: dict[str, int] = defaultdict(int)
        self._report_count: int = 0

    # -- context manager for a whole report ----------------------------------

    def begin_report(self) -> None:
        """Mark the start of processing a single report."""
        self._report_count += 1

    # -- per-stage context manager -------------------------------------------

    def stage(self, name: str) -> _StageTimer:
        """Return a context manager that times *name* and counts attempts."""
        self._attempts[name] += 1
        return _StageTimer(self, name)

    # -- internal recorders --------------------------------------------------

    def _record_latency(self, stage: str, seconds: float) -> None:
        self._latencies[stage].append(seconds)

    def _record_tokens(self, stage: str, input_tokens: int, output_tokens: int) -> None:
        self._input_tokens[stage].append(input_tokens)
        self._output_tokens[stage].append(output_tokens)

    def _record_failure(self, stage: str) -> None:
        self._failures[stage] += 1

    # -- helpers -------------------------------------------------------------

    @staticmethod
    def _mean(values: list[int] | list[float]) -> float:
        if not values:
            return 0.0
        return sum(values) / len(values)

    def _stage_cost(self, stage: str) -> float:
        inp = sum(self._input_tokens.get(stage, []))
        out = sum(self._output_tokens.get(stage, []))
        return (inp / 1000.0) * self.input_price_per_1k + (out / 1000.0) * self.output_price_per_1k

    # -- public API ----------------------------------------------------------

    def summarize(self) -> dict[str, Any]:
        """Return a nested dict summarising all recorded stages."""
        stages: dict[str, dict[str, Any]] = {}
        all_stages = sorted(
            set(self._latencies) | set(self._input_tokens) | set(self._attempts)
        )

        total_cost = 0.0
        total_latency = 0.0

        for stage in all_stages:
            mean_lat = round(self._mean(self._latencies.get(stage, [])), 6)
            mean_inp = round(self._mean(self._input_tokens.get(stage, [])), 2)
            mean_out = round(self._mean(self._output_tokens.get(stage, [])), 2)
            attempts = self._attempts.get(stage, 0)
            failures = self._failures.get(stage, 0)
            failure_rate = round(failures / max(attempts, 1), 4)
            cost = round(self._stage_cost(stage), 6)

            stages[stage] = {
                "mean_latency_s": mean_lat,
                "mean_input_tokens": mean_inp,
                "mean_output_tokens": mean_out,
                "attempts": attempts,
                "failures": failures,
                "failure_rate": failure_rate,
                "total_cost": cost,
            }

            total_cost += cost
            total_latency += sum(self._latencies.get(stage, []))

        report_count = max(self._report_count, 1)
        cost_per_report = round(total_cost / report_count, 6)
        latency_per_report = round(total_latency / report_count, 4)

        # Direct-LLM-only cost for comparison
        direct_cost = round(self._stage_cost("direct_llm"), 6)
        agentic_overhead = round(total_cost - direct_cost, 6) if total_cost > 0 else 0.0

        return {
            "stages": stages,
            "total_cost": round(total_cost, 6),
            "cost_per_report": cost_per_report,
            "latency_per_report_s": latency_per_report,
            "report_count": self._report_count,
            "direct_llm_only_cost": direct_cost,
            "agentic_overhead_cost": agentic_overhead,
        }

    def export_table(self, path: Path) -> None:
        """Write a CSV summary suitable for paper inclusion."""
        summary = self.summarize()
        stages = summary["stages"]
        fieldnames = [
            "stage",
            "mean_latency_s",
            "mean_input_tokens",
            "mean_output_tokens",
            "attempts",
            "failures",
            "failure_rate",
            "total_cost",
        ]

        with path.open("w", encoding="utf-8", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            for stage_name in sorted(stages):
                row = {"stage": stage_name, **stages[stage_name]}
                writer.writerow(row)
            # Totals row
            writer.writerow(
                {
                    "stage": "TOTAL",
                    "mean_latency_s": "",
                    "mean_input_tokens": "",
                    "mean_output_tokens": "",
                    "attempts": "",
                    "failures": "",
                    "failure_rate": "",
                    "total_cost": summary["total_cost"],
                }
            )
            writer.writerow(
                {
                    "stage": "per_report",
                    "mean_latency_s": summary["latency_per_report_s"],
                    "mean_input_tokens": "",
                    "mean_output_tokens": "",
                    "attempts": "",
                    "failures": "",
                    "failure_rate": "",
                    "total_cost": summary["cost_per_report"],
                }
            )
            writer.writerow(
                {
                    "stage": "direct_llm_only",
                    "mean_latency_s": "",
                    "mean_input_tokens": "",
                    "mean_output_tokens": "",
                    "attempts": "",
                    "failures": "",
                    "failure_rate": "",
                    "total_cost": summary["direct_llm_only_cost"],
                }
            )
            writer.writerow(
                {
                    "stage": "agentic_overhead",
                    "mean_latency_s": "",
                    "mean_input_tokens": "",
                    "mean_output_tokens": "",
                    "attempts": "",
                    "failures": "",
                    "failure_rate": "",
                    "total_cost": summary["agentic_overhead_cost"],
                }
            )
