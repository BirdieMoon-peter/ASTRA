from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]

DEFAULT_ANNOTATOR_A = ROOT / "data" / "annotations" / "stratreportzh_dev_aligned_workset_annotated.jsonl"
DEFAULT_ANNOTATOR_B = ROOT / "data" / "annotations" / "stratreportzh_dev_aligned_workset_annotated_normalized.jsonl"
OUT_JSON = ROOT / "artifacts" / "benchmark" / "annotation_agreement.json"
OUT_MD = ROOT / "docs" / "benchmark" / "annotation_agreement_report.md"

# Import the canonical IAA module (available when PYTHONPATH includes src/).
from astra.labeling.iaa_protocol import compute_iaa


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build annotation-agreement report between two annotator JSONL files.",
    )
    parser.add_argument(
        "--annotator-a",
        type=Path,
        default=DEFAULT_ANNOTATOR_A,
        help="Path to annotator-A JSONL file (default: raw annotated dev workset).",
    )
    parser.add_argument(
        "--annotator-b",
        type=Path,
        default=DEFAULT_ANNOTATOR_B,
        help="Path to annotator-B JSONL file (default: normalized annotated dev workset).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    annotator_a: Path = args.annotator_a
    annotator_b: Path = args.annotator_b

    # Use the canonical IAA module for all metric computation.
    agreement = compute_iaa(annotator_a, annotator_b)

    # ── persist JSON artifact ────────────────────────────────────────────
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(
        json.dumps(agreement, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # ── build Markdown report ────────────────────────────────────────────
    common_count = agreement["common_report_count"]
    lines = [
        "# Annotation Agreement Report",
        "",
        f"- Annotator A file: `{annotator_a}`",
        f"- Annotator B file: `{annotator_b}`",
        f"- Common report count: **{common_count}**",
        f"- Disagreement count: **{agreement['disagreement_count']}**",
        f"- Overall (micro) Cohen's kappa: **{agreement['overall_kappa']:.4f}**",
        "",
        "## Label agreement",
        "",
        "| Field | Accuracy | Cohen's kappa |",
        "|---|---:|---:|",
    ]
    for field, metrics in agreement["label_metrics"].items():
        lines.append(
            f"| {field} | {metrics['accuracy']:.4f} | {metrics['cohen_kappa']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Evidence span overlap",
            "",
            f"- Precision: **{agreement['span_metrics']['precision']:.4f}**",
            f"- Recall: **{agreement['span_metrics']['recall']:.4f}**",
            f"- F1: **{agreement['span_metrics']['f1']:.4f}**",
            "",
            "## Interpretation",
            "",
            "This report compares the raw annotated dev-aligned file with the "
            "normalized annotated file. It is a useful audit artifact, but it is "
            "not yet a substitute for a true multi-annotator benchmark release "
            "with formal adjudication records.",
        ]
    )

    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(
        json.dumps(
            {
                "agreement_json": str(OUT_JSON),
                "report_md": str(OUT_MD),
                "common_report_count": common_count,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
