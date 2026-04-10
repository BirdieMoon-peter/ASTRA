import csv
import io
import json
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from astra.labeling.freeze_gold_annotations import freeze_gold_annotations
from astra.labeling.sample_for_annotation import _to_annotation_record, write_gold_workset
from astra.labeling.validate_annotations import validate_annotation_file
from astra.pipelines import run_annotation_prep


class GoldWorksetTest(unittest.TestCase):
    def test_to_annotation_record_creates_independent_evidence_span_lists(self) -> None:
        first = _to_annotation_record(
            {
                "report_id": "r1",
                "report_date": "2023-01-01",
                "report_year": "2023",
                "split": "train",
                "stock_code": "000001",
                "company_name": "A",
                "title": "Title A",
                "summary": "Summary A",
            }
        )
        second = _to_annotation_record(
            {
                "report_id": "r2",
                "report_date": "2023-01-02",
                "report_year": "2023",
                "split": "train",
                "stock_code": "000002",
                "company_name": "B",
                "title": "Title B",
                "summary": "Summary B",
            }
        )

        first["annotation"]["evidence_spans"].append({"start": 0, "end": 5, "text": "Title"})

        self.assertEqual(len(first["annotation"]["evidence_spans"]), 1)
        self.assertEqual(second["annotation"]["evidence_spans"], [])

    def test_write_gold_workset_uses_split_targets_and_annotation_template(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            clean_reports_path = root / "reports_clean.csv"
            workset_path = root / "gold_workset.jsonl"
            fieldnames = [
                "report_id",
                "report_date",
                "report_year",
                "split",
                "stock_code",
                "company_name",
                "title",
                "summary",
            ]
            rows = [
                {
                    "report_id": "train-2022-1",
                    "report_date": "2022-01-05",
                    "report_year": "2022",
                    "split": "train",
                    "stock_code": "000001",
                    "company_name": "Train A",
                    "title": "Train title A",
                    "summary": "Train summary A",
                },
                {
                    "report_id": "train-2023-1",
                    "report_date": "2023-02-10",
                    "report_year": "2023",
                    "split": "train",
                    "stock_code": "000002",
                    "company_name": "Train B",
                    "title": "Train title B",
                    "summary": "Train summary B",
                },
                {
                    "report_id": "train-2023-2",
                    "report_date": "2023-03-15",
                    "report_year": "2023",
                    "split": "train",
                    "stock_code": "000003",
                    "company_name": "Train C",
                    "title": "Train title C",
                    "summary": "Train summary C",
                },
                {
                    "report_id": "dev-2024-1",
                    "report_date": "2024-04-01",
                    "report_year": "2024",
                    "split": "dev",
                    "stock_code": "000004",
                    "company_name": "Dev A",
                    "title": "Dev title A",
                    "summary": "Dev summary A",
                },
                {
                    "report_id": "dev-2024-2",
                    "report_date": "2024-04-02",
                    "report_year": "2024",
                    "split": "dev",
                    "stock_code": "000005",
                    "company_name": "Dev B",
                    "title": "Dev title B",
                    "summary": "Dev summary B",
                },
                {
                    "report_id": "test-2025-1",
                    "report_date": "2025-05-01",
                    "report_year": "2025",
                    "split": "test",
                    "stock_code": "000006",
                    "company_name": "Test A",
                    "title": "Test title A",
                    "summary": "Test summary A",
                },
                {
                    "report_id": "test-2025-2",
                    "report_date": "2025-05-02",
                    "report_year": "2025",
                    "split": "test",
                    "stock_code": "000007",
                    "company_name": "Test B",
                    "title": "Test title B",
                    "summary": "Test summary B",
                },
            ]
            with clean_reports_path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)

            summary = write_gold_workset(
                clean_reports_path=clean_reports_path,
                workset_path=workset_path,
                split_targets={"train": 2, "dev": 1, "test": 1},
            )

            self.assertEqual(summary["row_count"], 4)
            self.assertEqual(summary["split_counts"], {"train": 2, "dev": 1, "test": 1})

            records = [
                json.loads(line)
                for line in workset_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            self.assertEqual(len(records), 4)
            for record in records:
                self.assertIsNone(record["annotation"]["fundamental_sentiment"])
                self.assertEqual(record["annotation"]["evidence_spans"], [])

    def test_validate_annotation_file_reports_duplicate_ids_and_incomplete_annotations(self) -> None:
        with TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "gold_workset.jsonl"
            records = [
                {
                    "report_id": "dup-1",
                    "title": "Title A",
                    "summary": "Summary A",
                    "annotation": {
                        "fundamental_sentiment": None,
                        "strategic_optimism": None,
                        "phenomenon": None,
                        "evidence_spans": [],
                        "annotation_confidence": None,
                        "notes": "",
                    },
                },
                {
                    "report_id": "dup-1",
                    "title": "Title B",
                    "summary": "Summary B",
                    "annotation": {
                        "fundamental_sentiment": None,
                        "strategic_optimism": None,
                        "phenomenon": None,
                        "evidence_spans": [],
                        "annotation_confidence": None,
                        "notes": "",
                    },
                },
            ]
            with path.open("w", encoding="utf-8") as handle:
                for record in records:
                    handle.write(json.dumps(record, ensure_ascii=False) + "\n")

            result = validate_annotation_file(path)

        self.assertGreater(result["error_count"], 0)
        self.assertEqual(result["duplicate_report_id_count"], 1)
        self.assertEqual(result["incomplete_annotation_count"], 2)
        self.assertFalse(result["ready_to_freeze"])

    def test_run_annotation_prep_calls_write_gold_workset(self) -> None:
        buffer = io.StringIO()
        with patch("astra.pipelines.run_annotation_prep.write_gold_workset") as mock_write_gold_workset:
            with redirect_stdout(buffer):
                run_annotation_prep.main()

        mock_write_gold_workset.assert_called_once_with()
        self.assertIn("[OK] Gold annotation workset generated.", buffer.getvalue())

    def test_validate_annotation_file_reports_invalid_non_mapping_span_item(self) -> None:
        with TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "gold_workset.jsonl"
            row = {
                "report_id": "r1",
                "report_date": "2024-01-01",
                "report_year": 2024,
                "split": "dev",
                "stock_code": "000001",
                "company_name": "A",
                "title": "Title",
                "summary": "Summary",
                "annotation": {
                    "fundamental_sentiment": "neutral",
                    "strategic_optimism": "balanced",
                    "phenomenon": "none",
                    "evidence_spans": ["bad-span"],
                    "annotation_confidence": "medium",
                    "notes": "",
                },
            }
            path.write_text(json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")

            result = validate_annotation_file(path)

        self.assertGreater(result["error_count"], 0)
        self.assertIn("invalid evidence_spans item", result["errors"][0]["error"])

    def test_freeze_gold_annotations_copies_valid_workset_to_main_path(self) -> None:
        with TemporaryDirectory() as temp_dir:
            workset_path = Path(temp_dir) / "gold_workset.jsonl"
            output_path = Path(temp_dir) / "main.jsonl"
            title = "Positive outlook"
            summary = "Margins improved"
            text = f"{title}\n{summary}"
            row = {
                "report_id": "r1",
                "report_date": "2024-01-01",
                "report_year": 2024,
                "split": "dev",
                "stock_code": "000001",
                "company_name": "A",
                "title": title,
                "summary": summary,
                "annotation": {
                    "fundamental_sentiment": "positive",
                    "strategic_optimism": "high",
                    "phenomenon": "none",
                    "evidence_spans": [
                        {
                            "start": 0,
                            "end": len(title),
                            "text": text[: len(title)],
                        }
                    ],
                    "annotation_confidence": "high",
                    "notes": "",
                },
            }
            input_content = json.dumps(row, ensure_ascii=False) + "\n"
            workset_path.write_text(input_content, encoding="utf-8")

            result = freeze_gold_annotations(workset_path=workset_path, output_path=output_path)

            self.assertEqual(result["row_count"], 1)
            self.assertTrue(output_path.exists())
            self.assertEqual(output_path.read_text(encoding="utf-8"), input_content)

    def test_freeze_gold_annotations_rejects_invalid_workset_without_writing_output(self) -> None:
        with TemporaryDirectory() as temp_dir:
            workset_path = Path(temp_dir) / "gold_workset.jsonl"
            output_path = Path(temp_dir) / "main.jsonl"
            row = {
                "report_id": "r1",
                "report_date": "2024-01-01",
                "report_year": 2024,
                "split": "dev",
                "stock_code": "000001",
                "company_name": "A",
                "title": "Title",
                "summary": "Summary",
                "annotation": {
                    "fundamental_sentiment": None,
                    "strategic_optimism": None,
                    "phenomenon": None,
                    "evidence_spans": [],
                    "annotation_confidence": None,
                    "notes": "",
                },
            }
            workset_path.write_text(json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")

            with self.assertRaises(ValueError):
                freeze_gold_annotations(workset_path=workset_path, output_path=output_path)

            self.assertFalse(output_path.exists())

    def test_freeze_gold_annotations_rejects_existing_output_file(self) -> None:
        with TemporaryDirectory() as temp_dir:
            workset_path = Path(temp_dir) / "gold_workset.jsonl"
            output_path = Path(temp_dir) / "main.jsonl"
            title = "Positive outlook"
            summary = "Margins improved"
            text = f"{title}\n{summary}"
            row = {
                "report_id": "r1",
                "report_date": "2024-01-01",
                "report_year": 2024,
                "split": "dev",
                "stock_code": "000001",
                "company_name": "A",
                "title": title,
                "summary": summary,
                "annotation": {
                    "fundamental_sentiment": "positive",
                    "strategic_optimism": "high",
                    "phenomenon": "none",
                    "evidence_spans": [
                        {
                            "start": 0,
                            "end": len(title),
                            "text": text[: len(title)],
                        }
                    ],
                    "annotation_confidence": "high",
                    "notes": "",
                },
            }
            workset_path.write_text(json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")
            output_path.write_text("existing frozen gold\n", encoding="utf-8")

            with self.assertRaises(ValueError):
                freeze_gold_annotations(workset_path=workset_path, output_path=output_path)

            self.assertEqual(output_path.read_text(encoding="utf-8"), "existing frozen gold\n")


if __name__ == "__main__":
    unittest.main()
