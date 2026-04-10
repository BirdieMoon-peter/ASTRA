import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from astra.data.package_validation import validate_experiment_package


class ValidateExperimentPackageTest(unittest.TestCase):
    def test_validate_experiment_package_returns_ok_for_minimal_package(self) -> None:
        with TemporaryDirectory() as temp_dir:
            package_dir = Path(temp_dir)
            splits_dir = package_dir / "splits"
            market_dir = package_dir / "market"
            splits_dir.mkdir(parents=True, exist_ok=True)
            market_dir.mkdir(parents=True, exist_ok=True)

            master_csv = "\n".join(
                [
                    "report_id,report_date,split,stock_code,title,summary",
                    "r1,2025-01-02,dev,000001,Title,Summary",
                ]
            )
            split_csv = "\n".join(
                [
                    "report_id,report_date,split,stock_code,title,summary",
                    "r1,2025-01-02,dev,000001,Title,Summary",
                ]
            )
            market_csv = "\n".join(
                [
                    "stock_code,trade_date,close,pct_change",
                    "000001,2025-01-02,10.5,0.01",
                ]
            )

            (package_dir / "reports_experiment_master.csv").write_text(master_csv, encoding="utf-8")
            (splits_dir / "reports_train.csv").write_text(split_csv, encoding="utf-8")
            (splits_dir / "reports_dev.csv").write_text(split_csv, encoding="utf-8")
            (splits_dir / "reports_test.csv").write_text(split_csv, encoding="utf-8")
            (market_dir / "daily_prices.csv").write_text(market_csv, encoding="utf-8")

            result = validate_experiment_package(
                package_dir=package_dir,
                gold_path=Path("/tmp/missing.jsonl"),
            )

        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["split_counts"]["dev"], 1)
        self.assertIs(result["market_available"], True)

    def test_validate_experiment_package_returns_invalid_schema_when_split_file_missing_columns(self) -> None:
        with TemporaryDirectory() as temp_dir:
            package_dir = Path(temp_dir)
            splits_dir = package_dir / "splits"
            market_dir = package_dir / "market"
            splits_dir.mkdir(parents=True, exist_ok=True)
            market_dir.mkdir(parents=True, exist_ok=True)

            master_csv = "\n".join(
                [
                    "report_id,report_date,split,stock_code,title,summary",
                    "r1,2025-01-02,dev,000001,Title,Summary",
                ]
            )
            valid_split_csv = "\n".join(
                [
                    "report_id,report_date,split,stock_code,title,summary",
                    "r1,2025-01-02,dev,000001,Title,Summary",
                ]
            )
            invalid_dev_split_csv = "\n".join(
                [
                    "report_id,report_date,split,stock_code,title",
                    "r1,2025-01-02,dev,000001,Title",
                ]
            )
            market_csv = "\n".join(
                [
                    "stock_code,trade_date,close,pct_change",
                    "000001,2025-01-02,10.5,0.01",
                ]
            )

            (package_dir / "reports_experiment_master.csv").write_text(master_csv, encoding="utf-8")
            (splits_dir / "reports_train.csv").write_text(valid_split_csv, encoding="utf-8")
            (splits_dir / "reports_dev.csv").write_text(invalid_dev_split_csv, encoding="utf-8")
            (splits_dir / "reports_test.csv").write_text(valid_split_csv, encoding="utf-8")
            (market_dir / "daily_prices.csv").write_text(market_csv, encoding="utf-8")

            result = validate_experiment_package(
                package_dir=package_dir,
                gold_path=Path("/tmp/missing.jsonl"),
            )

        self.assertEqual(result["status"], "invalid_schema")
        self.assertEqual(result["missing_split_columns"]["reports_dev"], ["summary"])


if __name__ == "__main__":
    unittest.main()
