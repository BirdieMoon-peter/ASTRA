import io
import re
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

from astra.config.task_schema import ANNOTATION_GOLD_WORKSET_PATH
from astra.pipelines import run_annotation_freeze, run_annotation_prep, run_annotation_validation


class AnnotationGoldPipelineTest(unittest.TestCase):
    def _load_guideline_content(self) -> str:
        guideline_path = Path("data/annotations/annotation_guideline.md")
        self.assertTrue(guideline_path.exists())
        return guideline_path.read_text(encoding="utf-8")

    def _load_field_sections(self) -> dict[str, str]:
        content = self._load_guideline_content()
        heading_pattern = re.compile(r"^###\s+.*$", re.MULTILINE)
        matches = list(heading_pattern.finditer(content))
        sections: dict[str, str] = {}
        field_names = (
            "fundamental_sentiment",
            "strategic_optimism",
            "phenomenon",
            "evidence_spans",
            "annotation_confidence",
            "notes",
        )

        for index, match in enumerate(matches):
            heading = match.group(0)
            field_name = next((name for name in field_names if f"`{name}`" in heading), None)
            if field_name is None:
                continue
            start = match.end()
            end = matches[index + 1].start() if index + 1 < len(matches) else len(content)
            sections[field_name] = content[start:end].strip()
        return sections

    def test_annotation_guideline_mentions_all_required_fields(self) -> None:
        content = self._load_guideline_content()

        for required_text in (
            "fundamental_sentiment",
            "strategic_optimism",
            "phenomenon",
            "evidence_spans",
            "annotation_confidence",
            "notes",
            "最小充分片段",
        ):
            with self.subTest(required_text=required_text):
                self.assertIn(required_text, content)

    def test_annotation_guideline_covers_field_sections_and_core_rules(self) -> None:
        sections = self._load_field_sections()

        self.assertEqual(
            set(sections),
            {
                "fundamental_sentiment",
                "strategic_optimism",
                "phenomenon",
                "evidence_spans",
                "annotation_confidence",
                "notes",
            },
        )
        for field_name, body in sections.items():
            with self.subTest(field_name=field_name):
                self.assertTrue(body)

        for label in ("negative", "neutral", "positive"):
            self.assertIn(f"`{label}`", sections["fundamental_sentiment"])
        for label in ("low", "balanced", "high"):
            self.assertIn(f"`{label}`", sections["strategic_optimism"])
        for label in ("low", "medium", "high"):
            self.assertIn(f"`{label}`", sections["annotation_confidence"])
        for label in (
            "hedged_downside",
            "euphemistic_risk",
            "title_body_mismatch",
            "omitted_downside_context",
            "none",
        ):
            self.assertIn(f"`{label}`", sections["phenomenon"])

        self.assertIn("最小充分片段", sections["evidence_spans"])
        self.assertIn("留空", sections["notes"])
        self.assertIn("说明", sections["notes"])

    def test_run_annotation_validation_uses_gold_workset_path(self) -> None:
        buffer = io.StringIO()
        mocked_result = {
            "file": "gold.jsonl",
            "row_count": 1,
            "error_count": 0,
            "errors": [],
            "duplicate_report_id_count": 0,
            "incomplete_annotation_count": 0,
            "ready_to_freeze": True,
            "sentiment_distribution": {},
            "strategic_optimism_distribution": {},
            "phenomenon_distribution": {},
            "annotation_confidence_distribution": {},
        }

        with patch(
            "astra.pipelines.run_annotation_validation.validate_annotation_file",
            return_value=mocked_result,
        ) as mock_validate:
            with redirect_stdout(buffer):
                run_annotation_validation.main()

        mock_validate.assert_called_once_with(ANNOTATION_GOLD_WORKSET_PATH)
        self.assertIn("[OK] Gold annotation validation complete.", buffer.getvalue())

    def test_run_annotation_validation_prints_warning_when_workset_has_issues(self) -> None:
        buffer = io.StringIO()
        mocked_result = {
            "file": "gold.jsonl",
            "row_count": 1,
            "error_count": 1,
            "errors": [{"report_id": "r1", "error": "duplicate report_id"}],
            "duplicate_report_id_count": 1,
            "incomplete_annotation_count": 1,
            "ready_to_freeze": False,
            "sentiment_distribution": {},
            "strategic_optimism_distribution": {},
            "phenomenon_distribution": {},
            "annotation_confidence_distribution": {},
        }

        with patch(
            "astra.pipelines.run_annotation_validation.validate_annotation_file",
            return_value=mocked_result,
        ):
            with redirect_stdout(buffer):
                run_annotation_validation.main()

        self.assertIn("[WARN] Gold annotation validation found issues.", buffer.getvalue())

    def test_run_annotation_freeze_calls_freeze_function(self) -> None:
        buffer = io.StringIO()
        mocked_result = {
            "row_count": 1,
            "output_path": "main.jsonl",
            "source_path": "gold_workset.jsonl",
        }

        with patch(
            "astra.pipelines.run_annotation_freeze.freeze_gold_annotations",
            return_value=mocked_result,
        ) as mock_freeze:
            with redirect_stdout(buffer):
                run_annotation_freeze.main()

        mock_freeze.assert_called_once_with()
        self.assertIn("[OK] Gold annotation file frozen.", buffer.getvalue())

    def test_run_annotation_freeze_propagates_failure_without_success_message(self) -> None:
        buffer = io.StringIO()

        with patch(
            "astra.pipelines.run_annotation_freeze.freeze_gold_annotations",
            side_effect=ValueError("not ready to freeze"),
        ):
            with self.assertRaises(ValueError):
                with redirect_stdout(buffer):
                    run_annotation_freeze.main()

        self.assertNotIn("[OK] Gold annotation file frozen.", buffer.getvalue())

    def test_pipeline_files_exist_for_gold_workflow(self) -> None:
        self.assertTrue(Path("src/astra/pipelines/run_annotation_prep.py").exists())
        self.assertTrue(Path("src/astra/pipelines/run_annotation_validation.py").exists())
        self.assertTrue(Path("src/astra/pipelines/run_annotation_freeze.py").exists())

    def test_gold_workflow_entrypoints_follow_pipeline_contract(self) -> None:
        prep_text = Path("src/astra/pipelines/run_annotation_prep.py").read_text(encoding="utf-8")
        validation_text = Path("src/astra/pipelines/run_annotation_validation.py").read_text(encoding="utf-8")
        freeze_text = Path("src/astra/pipelines/run_annotation_freeze.py").read_text(encoding="utf-8")

        self.assertIn("write_gold_workset", prep_text)
        self.assertIn("validate_annotation_file", validation_text)
        self.assertIn("ANNOTATION_GOLD_WORKSET_PATH", validation_text)
        self.assertIn("freeze_gold_annotations", freeze_text)

    def test_run_annotation_prep_prints_success_message(self) -> None:
        buffer = io.StringIO()
        with patch("astra.pipelines.run_annotation_prep.write_gold_workset") as mock_write:
            with redirect_stdout(buffer):
                run_annotation_prep.main()

        mock_write.assert_called_once_with()
        self.assertIn("[OK] Gold annotation workset generated.", buffer.getvalue())


if __name__ == "__main__":
    unittest.main()
