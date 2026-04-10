#!/usr/bin/env python3
"""Prepare a human-evaluation batch from ASTRA predictions and gold annotations.

Usage
-----
    python scripts/eval/prepare_human_eval_batch.py \
        --predictions-path outputs/predictions/astra_predictions.jsonl \
        --gold-path data/annotations/stratreportzh_dev_aligned_workset_annotated_normalized.jsonl \
        --n 50 \
        --output data/annotations/human_eval_batch.jsonl
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure the project root is importable when running as a script.
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from astra.evaluation.human_eval import (
    export_eval_batch,
    prepare_eval_packet,
    select_eval_sample,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare a human-evaluation JSONL batch for ASTRA neutralisation quality.",
    )
    parser.add_argument(
        "--predictions-path",
        type=Path,
        default=_PROJECT_ROOT / "outputs" / "predictions" / "astra_predictions.jsonl",
        help="Path to ASTRA predictions JSONL.",
    )
    parser.add_argument(
        "--gold-path",
        type=Path,
        default=(
            _PROJECT_ROOT
            / "data"
            / "annotations"
            / "stratreportzh_dev_aligned_workset_annotated_normalized.jsonl"
        ),
        help="Path to gold annotations JSONL.",
    )
    parser.add_argument(
        "--n",
        type=int,
        default=50,
        help="Number of examples to include in the evaluation batch.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible sampling.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=_PROJECT_ROOT / "data" / "annotations" / "human_eval_batch.jsonl",
        help="Output path for the evaluation batch JSONL.",
    )
    args = parser.parse_args()

    print(f"Loading predictions from {args.predictions_path}")
    print(f"Loading gold annotations from {args.gold_path}")

    sample = select_eval_sample(
        predictions_path=args.predictions_path,
        gold_path=args.gold_path,
        n=args.n,
        seed=args.seed,
    )
    print(f"Selected {len(sample)} reports for evaluation.")

    packets = prepare_eval_packet(sample)
    export_eval_batch(packets, args.output)
    print(f"Wrote {len(packets)} evaluation packets to {args.output}")


if __name__ == "__main__":
    main()
