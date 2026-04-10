import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from astra.pipelines.run_nlp_eval import _resolve_gold_path


class ResolveGoldPathTest(unittest.TestCase):
    def test_prefers_candidate_with_highest_prediction_overlap(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            pred_path = root / "pred.jsonl"
            dev_gold = root / "dev.jsonl"
            main_gold = root / "main.jsonl"

            pred_path.write_text(
                "\n".join(
                    [
                        json.dumps({"report_id": "dev-1"}, ensure_ascii=False),
                        json.dumps({"report_id": "dev-2"}, ensure_ascii=False),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            dev_gold.write_text(
                json.dumps({"report_id": "dev-1"}, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            main_gold.write_text(
                json.dumps({"report_id": "main-1"}, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )

            with patch("astra.pipelines.run_nlp_eval.ANNOTATION_DEV_ALIGNED_WORKSET_ANNOTATED_NORMALIZED_PATH", dev_gold), patch(
                "astra.pipelines.run_nlp_eval.ANNOTATION_MAIN_PATH", main_gold
            ), patch("astra.pipelines.run_nlp_eval.ANNOTATION_PILOT_PATH", root / "missing-pilot.jsonl"), patch(
                "astra.pipelines.run_nlp_eval.MAIN_SAMPLE_PATH", root / "missing-main-sample.jsonl"
            ), patch("astra.pipelines.run_nlp_eval.PILOT_SAMPLE_PATH", root / "missing-pilot-sample.jsonl"):
                resolved = _resolve_gold_path({"direct_llm": pred_path})

            self.assertEqual(resolved, dev_gold)



class EvidenceSpanFormatCompatibilityTest(unittest.TestCase):
    def test_eval_accepts_string_evidence_spans_in_gold(self) -> None:
        from astra.evaluation.nlp_metrics import evidence_span_prf

        gold_rows = [
            {
                "report_id": "r1",
                "annotation": {
                    "evidence_spans": ["片段A", "片段B"],
                },
            }
        ]
        pred_rows = [
            {
                "report_id": "r1",
                "evidence_spans": [{"text": "片段A"}, {"text": "片段C"}],
            }
        ]

        result = evidence_span_prf(gold_rows, pred_rows)
        self.assertEqual(result["evidence_precision"], 0.5)
        self.assertEqual(result["evidence_recall"], 0.5)
        self.assertEqual(result["evidence_f1"], 0.5)


if __name__ == "__main__":
    unittest.main()
