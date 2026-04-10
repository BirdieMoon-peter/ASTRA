#!/usr/bin/env python3
"""Aggregate completed human evaluations and print a summary.

Usage
-----
    python scripts/eval/aggregate_human_eval.py \
        --input data/annotations/human_eval_completed_annotator1.jsonl \
               data/annotations/human_eval_completed_annotator2.jsonl \
        --output outputs/eval/human_eval_results.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Ensure the project root is importable when running as a script.
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from astra.evaluation.human_eval import aggregate_eval_results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Aggregate completed human evaluation JSONL files.",
    )
    parser.add_argument(
        "--input",
        type=Path,
        nargs="+",
        required=True,
        help="One or more completed evaluation JSONL paths (one per evaluator).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=_PROJECT_ROOT / "outputs" / "eval" / "human_eval_results.json",
        help="Output path for aggregated results JSON.",
    )
    args = parser.parse_args()

    print(f"Aggregating {len(args.input)} evaluator file(s):")
    for p in args.input:
        print(f"  - {p}")

    results = aggregate_eval_results(args.input)

    # Write output.
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as fh:
        json.dump(results, fh, ensure_ascii=False, indent=2)
    print(f"\nResults written to {args.output}")

    # Print summary to console.
    print("\n===== Human Evaluation Summary =====")
    print(f"Evaluators:               {results['evaluator_count']}")
    print(f"Reports evaluated:        {results['report_count']}")
    print(f"Factual preservation:     {results['factual_preservation_rate']:.1%}")
    faith = results["faithfulness"]
    print(f"Faithfulness (mean/std):  {faith['mean']:.2f} / {faith['std']:.2f}  (n={faith['n']})")
    tone = results["tone_removal"]
    print(f"Tone removal  (mean/std): {tone['mean']:.2f} / {tone['std']:.2f}  (n={tone['n']})")

    iaa = results.get("inter_annotator_agreement", {})
    print(f"\nKrippendorff alpha (faithfulness): {iaa.get('krippendorff_alpha_faithfulness', 'N/A')}")
    print(f"Krippendorff alpha (tone_removal): {iaa.get('krippendorff_alpha_tone_removal', 'N/A')}")

    per_phenom = results.get("per_phenomenon", {})
    if per_phenom:
        print("\n--- Per-phenomenon breakdown ---")
        for phenom, stats in sorted(per_phenom.items()):
            fp_rate = stats.get("factual_preservation_rate", 0.0)
            f_mean = stats.get("faithfulness", {}).get("mean", 0.0)
            t_mean = stats.get("tone_removal", {}).get("mean", 0.0)
            print(f"  {phenom:30s}  fact_pres={fp_rate:.1%}  faith={f_mean:.2f}  tone={t_mean:.2f}")

    print("\nDone.")


if __name__ == "__main__":
    main()
