import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from astra.config import task_schema


class TaskSchemaPathsTest(unittest.TestCase):
    def test_default_experiment_package_dir_points_to_data_experiment_package(self) -> None:
        self.assertEqual(
            task_schema.DEFAULT_EXPERIMENT_PACKAGE_DIR,
            task_schema.DATA_DIR / "experiment_package",
        )

    def test_annotation_gold_workset_path_is_derived_from_annotations_dir(self) -> None:
        self.assertEqual(
            task_schema.ANNOTATION_GOLD_WORKSET_PATH,
            task_schema.ANNOTATIONS_DIR / "stratreportzh_gold_workset.jsonl",
        )

    def test_paper_output_and_latex_dirs_are_derived_from_outputs(self) -> None:
        self.assertEqual(task_schema.PAPER_OUTPUT_DIR, task_schema.OUTPUTS_DIR / "paper")
        self.assertEqual(task_schema.PAPER_LATEX_DIR, task_schema.PAPER_OUTPUT_DIR / "latex")

    def test_resolve_market_prices_path_prefers_experiment_path_when_present(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            default_market_path = root / "market" / "daily_prices.csv"
            experiment_market_path = root / "experiment_package" / "market" / "daily_prices.csv"
            experiment_market_path.parent.mkdir(parents=True, exist_ok=True)
            experiment_market_path.write_text("date,close\n", encoding="utf-8")

            resolved = task_schema.resolve_market_prices_path(
                default_market_path=default_market_path,
                experiment_market_path=experiment_market_path,
            )

        self.assertEqual(resolved, experiment_market_path)

    def test_resolve_market_prices_path_falls_back_to_default_when_experiment_path_missing(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            default_market_path = root / "market" / "daily_prices.csv"
            experiment_market_path = root / "experiment_package" / "market" / "daily_prices.csv"

            resolved = task_schema.resolve_market_prices_path(
                default_market_path=default_market_path,
                experiment_market_path=experiment_market_path,
            )

        self.assertEqual(resolved, default_market_path)

    def test_resolve_experiment_package_dir_prefers_default_dir_when_present(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            default_dir = root / "data" / "experiment_package"
            legacy_dir = root / "data" / "processed" / "experiment_package"
            default_dir.mkdir(parents=True, exist_ok=True)

            resolved = task_schema.resolve_experiment_package_dir(
                default_dir=default_dir,
                legacy_dir=legacy_dir,
            )

        self.assertEqual(resolved, default_dir)

    def test_resolve_experiment_package_dir_prefers_default_dir_when_both_exist(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            default_dir = root / "data" / "experiment_package"
            legacy_dir = root / "data" / "processed" / "experiment_package"
            default_dir.mkdir(parents=True, exist_ok=True)
            legacy_dir.mkdir(parents=True, exist_ok=True)

            resolved = task_schema.resolve_experiment_package_dir(
                default_dir=default_dir,
                legacy_dir=legacy_dir,
            )

        self.assertEqual(resolved, default_dir)

    def test_resolve_experiment_package_dir_uses_legacy_dir_when_default_missing(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            default_dir = root / "data" / "experiment_package"
            legacy_dir = root / "data" / "processed" / "experiment_package"
            legacy_dir.mkdir(parents=True, exist_ok=True)

            resolved = task_schema.resolve_experiment_package_dir(
                default_dir=default_dir,
                legacy_dir=legacy_dir,
            )

        self.assertEqual(resolved, legacy_dir)

    def test_resolve_experiment_package_dir_returns_default_dir_when_neither_exists(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            default_dir = root / "data" / "experiment_package"
            legacy_dir = root / "data" / "processed" / "experiment_package"

            resolved = task_schema.resolve_experiment_package_dir(
                default_dir=default_dir,
                legacy_dir=legacy_dir,
            )

        self.assertEqual(resolved, default_dir)

    def test_lazy_experiment_paths_recompute_from_current_filesystem_state(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            default_dir = root / "data" / "experiment_package"
            legacy_dir = root / "data" / "processed" / "experiment_package"
            market_dir = root / "data" / "market"
            market_path = market_dir / "daily_prices.csv"
            market_dir.mkdir(parents=True, exist_ok=True)
            market_path.write_text("date,close\n", encoding="utf-8")
            legacy_dir.mkdir(parents=True, exist_ok=True)

            with patch.object(task_schema, "DEFAULT_EXPERIMENT_PACKAGE_DIR", default_dir), patch.object(
                task_schema,
                "LEGACY_EXPERIMENT_PACKAGE_DIR",
                legacy_dir,
            ):
                self.assertEqual(task_schema.EXPERIMENT_PACKAGE_DIR, legacy_dir)
                self.assertEqual(task_schema.EXPERIMENT_MARKET_DIR, legacy_dir / "market")
                self.assertEqual(task_schema.EXPERIMENT_SPLITS_DIR, legacy_dir / "splits")
                self.assertEqual(
                    task_schema.REPORTS_EXPERIMENT_MASTER_PATH,
                    legacy_dir / "reports_experiment_master.csv",
                )
                self.assertEqual(
                    task_schema.EXPERIMENT_RATINGS_PATH,
                    legacy_dir / "report_ratings.csv",
                )
                self.assertEqual(
                    task_schema.EXPERIMENT_BROKERS_PATH,
                    legacy_dir / "report_brokers.csv",
                )
                self.assertEqual(
                    task_schema.EXPERIMENT_ANALYSTS_PATH,
                    legacy_dir / "report_analysts.csv",
                )
                self.assertEqual(
                    task_schema.EXPERIMENT_VERSIONS_PATH,
                    legacy_dir / "report_versions.csv",
                )
                self.assertEqual(
                    task_schema.EXPERIMENT_MARKET_PRICES_PATH,
                    legacy_dir / "market" / "daily_prices.csv",
                )
                self.assertEqual(task_schema.EXPERIMENT_MANIFEST_PATH, legacy_dir / "manifest.json")
                self.assertEqual(task_schema.EXPERIMENT_TRAIN_PATH, legacy_dir / "splits" / "reports_train.csv")
                self.assertEqual(task_schema.EXPERIMENT_DEV_PATH, legacy_dir / "splits" / "reports_dev.csv")
                self.assertEqual(task_schema.EXPERIMENT_TEST_PATH, legacy_dir / "splits" / "reports_test.csv")
                self.assertEqual(task_schema.resolve_market_prices_path(default_market_path=market_path), market_path)

                default_dir.mkdir(parents=True, exist_ok=True)
                (default_dir / "market").mkdir(parents=True, exist_ok=True)
                (default_dir / "market" / "daily_prices.csv").write_text("date,close\n", encoding="utf-8")

                self.assertEqual(task_schema.EXPERIMENT_PACKAGE_DIR, default_dir)
                self.assertEqual(task_schema.EXPERIMENT_MARKET_DIR, default_dir / "market")
                self.assertEqual(task_schema.EXPERIMENT_SPLITS_DIR, default_dir / "splits")
                self.assertEqual(
                    task_schema.REPORTS_EXPERIMENT_MASTER_PATH,
                    default_dir / "reports_experiment_master.csv",
                )
                self.assertEqual(
                    task_schema.EXPERIMENT_RATINGS_PATH,
                    default_dir / "report_ratings.csv",
                )
                self.assertEqual(
                    task_schema.EXPERIMENT_BROKERS_PATH,
                    default_dir / "report_brokers.csv",
                )
                self.assertEqual(
                    task_schema.EXPERIMENT_ANALYSTS_PATH,
                    default_dir / "report_analysts.csv",
                )
                self.assertEqual(
                    task_schema.EXPERIMENT_VERSIONS_PATH,
                    default_dir / "report_versions.csv",
                )
                self.assertEqual(
                    task_schema.EXPERIMENT_MARKET_PRICES_PATH,
                    default_dir / "market" / "daily_prices.csv",
                )
                self.assertEqual(task_schema.EXPERIMENT_MANIFEST_PATH, default_dir / "manifest.json")
                self.assertEqual(task_schema.EXPERIMENT_TRAIN_PATH, default_dir / "splits" / "reports_train.csv")
                self.assertEqual(task_schema.EXPERIMENT_DEV_PATH, default_dir / "splits" / "reports_dev.csv")
                self.assertEqual(task_schema.EXPERIMENT_TEST_PATH, default_dir / "splits" / "reports_test.csv")
                self.assertEqual(
                    task_schema.resolve_market_prices_path(default_market_path=market_path),
                    default_dir / "market" / "daily_prices.csv",
                )


if __name__ == "__main__":
    unittest.main()
