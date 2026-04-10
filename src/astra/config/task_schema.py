from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Final

SENTIMENT_LABELS: Final[tuple[str, ...]] = ("negative", "neutral", "positive")
STRATEGIC_OPTIMISM_LABELS: Final[tuple[str, ...]] = ("low", "balanced", "high")
PHENOMENON_LABELS: Final[tuple[str, ...]] = (
    "hedged_downside",
    "euphemistic_risk",
    "title_body_mismatch",
    "omitted_downside_context",
    "none",
)
ANNOTATION_CONFIDENCE_LABELS: Final[tuple[str, ...]] = ("low", "medium", "high")
DATA_SPLITS: Final[tuple[str, ...]] = ("train", "dev", "test")

TRAIN_END_YEAR: Final[int] = 2024
DEV_YEAR: Final[int] = 2025
TEST_YEAR: Final[int] = 2026

PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parents[3]
DATA_DIR: Final[Path] = PROJECT_ROOT / "data"
INTERIM_DIR: Final[Path] = DATA_DIR / "interim"
PROCESSED_DIR: Final[Path] = DATA_DIR / "processed"
ANNOTATIONS_DIR: Final[Path] = DATA_DIR / "annotations"
OUTPUTS_DIR: Final[Path] = PROJECT_ROOT / "outputs"
SNAPSHOTS_DIR: Final[Path] = OUTPUTS_DIR / "snapshots"
DATASET_OUTPUT_DIR: Final[Path] = OUTPUTS_DIR / "datasets"
EVAL_OUTPUT_DIR: Final[Path] = OUTPUTS_DIR / "eval"
PREDICTIONS_DIR: Final[Path] = OUTPUTS_DIR / "predictions"
ABLATION_OUTPUT_DIR: Final[Path] = PREDICTIONS_DIR / "ablations"
CASE_OUTPUT_DIR: Final[Path] = OUTPUTS_DIR / "cases"
BACKTEST_DIR: Final[Path] = OUTPUTS_DIR / "backtest"
PAPER_OUTPUT_DIR: Final[Path] = OUTPUTS_DIR / "paper"
PAPER_LATEX_DIR: Final[Path] = PAPER_OUTPUT_DIR / "latex"
MARKET_DIR: Final[Path] = DATA_DIR / "market"
DEFAULT_EXPERIMENT_PACKAGE_DIR: Final[Path] = DATA_DIR / "experiment_package"
LEGACY_EXPERIMENT_PACKAGE_DIR: Final[Path] = PROCESSED_DIR / "experiment_package"


def resolve_experiment_package_dir(
    *,
    default_dir: Path | None = None,
    legacy_dir: Path | None = None,
) -> Path:
    if default_dir is None:
        default_dir = DEFAULT_EXPERIMENT_PACKAGE_DIR
    if legacy_dir is None:
        legacy_dir = LEGACY_EXPERIMENT_PACKAGE_DIR
    if default_dir.exists():
        return default_dir
    if legacy_dir.exists():
        return legacy_dir
    return default_dir


MARKET_PRICES_PATH: Final[Path] = MARKET_DIR / "daily_prices.csv"
MARKET_PRICES_STATE_PATH: Final[Path] = MARKET_DIR / "daily_prices_state.json"
MARKET_PRICES_STATUS_PATH: Final[Path] = MARKET_DIR / "daily_prices_fetch_status.csv"
MARKET_PRICES_MISSING_PATH: Final[Path] = MARKET_DIR / "daily_prices_missing.csv"
RAW_REPORTS_PATH: Final[Path] = DATA_DIR / "reports.csv"
REPORTS_MASTER_PATH: Final[Path] = DATA_DIR / "reports_master.csv"
REPORT_RATINGS_PATH: Final[Path] = DATA_DIR / "report_ratings.csv"
REPORT_BROKERS_PATH: Final[Path] = DATA_DIR / "report_brokers.csv"
REPORT_BROKER_ALIASES_PATH: Final[Path] = DATA_DIR / "report_broker_aliases.csv"
REPORT_ANALYSTS_PATH: Final[Path] = DATA_DIR / "report_analysts.csv"
REPORT_ANALYST_BRIDGE_PATH: Final[Path] = DATA_DIR / "report_analyst_bridge.csv"
REPORT_VERSIONS_PATH: Final[Path] = DATA_DIR / "report_versions.csv"
REPORT_REBUILD_STATE_PATH: Final[Path] = DATA_DIR / "report_rebuild_state.json"
CLEAN_REPORTS_PATH: Final[Path] = INTERIM_DIR / "reports_clean.csv"
PAPER_TABLE_MAIN_NLP_PATH: Final[Path] = PAPER_OUTPUT_DIR / "table_main_nlp.csv"
PAPER_TABLE_DATA_PATH: Final[Path] = PAPER_OUTPUT_DIR / "table_data.csv"
PAPER_TABLE_PHENOMENA_PATH: Final[Path] = PAPER_OUTPUT_DIR / "table_phenomena.csv"
PAPER_TABLE_FINANCE_PATH: Final[Path] = PAPER_OUTPUT_DIR / "table_finance.csv"
PAPER_TABLE_ABLATION_PATH: Final[Path] = PAPER_OUTPUT_DIR / "table_ablation.csv"
PAPER_TABLE_CASE_PATH: Final[Path] = PAPER_OUTPUT_DIR / "table_case.csv"
PAPER_FIGURE_BACKTEST_PATH: Final[Path] = PAPER_OUTPUT_DIR / "figure_backtest.csv"
PAPER_FIGURE_CALIBRATION_PATH: Final[Path] = PAPER_OUTPUT_DIR / "figure_calibration.csv"
PAPER_FIGURE_REGIME_PATH: Final[Path] = PAPER_OUTPUT_DIR / "figure_regime.csv"
PAPER_LATEX_MAIN_NLP_PATH: Final[Path] = PAPER_LATEX_DIR / "tab_main_nlp.tex"
PAPER_LATEX_DATA_PATH: Final[Path] = PAPER_LATEX_DIR / "tab_data.tex"
PAPER_LATEX_PHENOMENA_PATH: Final[Path] = PAPER_LATEX_DIR / "tab_phenomena.tex"
PAPER_LATEX_FINANCE_PATH: Final[Path] = PAPER_LATEX_DIR / "tab_finance.tex"
PAPER_LATEX_ABLATION_PATH: Final[Path] = PAPER_LATEX_DIR / "tab_ablation.tex"
PAPER_LATEX_CASE_PATH: Final[Path] = PAPER_LATEX_DIR / "tab_case.tex"
CORPUS_STATS_PATH: Final[Path] = EVAL_OUTPUT_DIR / "corpus_stats.json"
SPLIT_SUMMARY_PATH: Final[Path] = EVAL_OUTPUT_DIR / "split_summary.json"
ANNOTATION_PILOT_PATH: Final[Path] = ANNOTATIONS_DIR / "stratreportzh_pilot.jsonl"
ANNOTATION_MAIN_PATH: Final[Path] = ANNOTATIONS_DIR / "stratreportzh_main.jsonl"
ANNOTATION_GOLD_WORKSET_PATH: Final[Path] = ANNOTATIONS_DIR / "stratreportzh_gold_workset.jsonl"
ANNOTATION_DEV_ALIGNED_WORKSET_PATH: Final[Path] = ANNOTATIONS_DIR / "stratreportzh_dev_aligned_workset.jsonl"
ANNOTATION_DEV_ALIGNED_WORKSET_ANNOTATED_NORMALIZED_PATH: Final[Path] = ANNOTATIONS_DIR / "stratreportzh_dev_aligned_workset_annotated_normalized.jsonl"
ANNOTATION_GUIDELINE_PATH: Final[Path] = ANNOTATIONS_DIR / "annotation_guideline.md"
PILOT_SAMPLE_PATH: Final[Path] = DATASET_OUTPUT_DIR / "pilot_annotation_sample.jsonl"
MAIN_SAMPLE_PATH: Final[Path] = DATASET_OUTPUT_DIR / "main_annotation_sample.jsonl"
ASTRA_PREDICTIONS_PATH: Final[Path] = PREDICTIONS_DIR / "astra_predictions.jsonl"
DIRECT_BASELINE_PATH: Final[Path] = PREDICTIONS_DIR / "direct_baseline_predictions.jsonl"
COT_BASELINE_PATH: Final[Path] = PREDICTIONS_DIR / "cot_baseline_predictions.jsonl"
REACT_BASELINE_PATH: Final[Path] = PREDICTIONS_DIR / "react_baseline_predictions.jsonl"
RULE_BASELINE_PATH: Final[Path] = PREDICTIONS_DIR / "rule_baseline_predictions.jsonl"
CASE_STUDIES_PATH: Final[Path] = CASE_OUTPUT_DIR / "case_studies.jsonl"
NLP_METRICS_PATH: Final[Path] = EVAL_OUTPUT_DIR / "nlp_metrics.json"
CALIBRATION_PATH: Final[Path] = EVAL_OUTPUT_DIR / "calibration_bins.json"
PHENOMENA_METRICS_PATH: Final[Path] = EVAL_OUTPUT_DIR / "phenomena_metrics.json"
FINANCE_METRICS_PATH: Final[Path] = BACKTEST_DIR / "finance_metrics.json"
BACKTEST_CURVE_PATH: Final[Path] = BACKTEST_DIR / "backtest_curve.csv"
REGIME_HEATMAP_PATH: Final[Path] = BACKTEST_DIR / "regime_heatmap.csv"


_EXPERIMENT_LAZY_PATHS: Final[dict[str, tuple[str | None, str]]] = {
    "EXPERIMENT_PACKAGE_DIR": (None, ""),
    "EXPERIMENT_MARKET_DIR": ("EXPERIMENT_PACKAGE_DIR", "market"),
    "EXPERIMENT_SPLITS_DIR": ("EXPERIMENT_PACKAGE_DIR", "splits"),
    "REPORTS_EXPERIMENT_MASTER_PATH": ("EXPERIMENT_PACKAGE_DIR", "reports_experiment_master.csv"),
    "EXPERIMENT_RATINGS_PATH": ("EXPERIMENT_PACKAGE_DIR", "report_ratings.csv"),
    "EXPERIMENT_BROKERS_PATH": ("EXPERIMENT_PACKAGE_DIR", "report_brokers.csv"),
    "EXPERIMENT_ANALYSTS_PATH": ("EXPERIMENT_PACKAGE_DIR", "report_analysts.csv"),
    "EXPERIMENT_VERSIONS_PATH": ("EXPERIMENT_PACKAGE_DIR", "report_versions.csv"),
    "EXPERIMENT_MARKET_PRICES_PATH": ("EXPERIMENT_MARKET_DIR", "daily_prices.csv"),
    "EXPERIMENT_MANIFEST_PATH": ("EXPERIMENT_PACKAGE_DIR", "manifest.json"),
    "EXPERIMENT_TRAIN_PATH": ("EXPERIMENT_SPLITS_DIR", "reports_train.csv"),
    "EXPERIMENT_DEV_PATH": ("EXPERIMENT_SPLITS_DIR", "reports_dev.csv"),
    "EXPERIMENT_TEST_PATH": ("EXPERIMENT_SPLITS_DIR", "reports_test.csv"),
}


def _resolve_experiment_lazy_path(name: str) -> Path:
    parent_name, child_name = _EXPERIMENT_LAZY_PATHS[name]
    if parent_name is None:
        return resolve_experiment_package_dir()

    return _resolve_experiment_lazy_path(parent_name) / child_name


def __getattr__(name: str) -> Path:
    if name not in _EXPERIMENT_LAZY_PATHS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    return _resolve_experiment_lazy_path(name)


def resolve_outputs_root(outputs_root: Path | None = None) -> Path:
    return outputs_root or OUTPUTS_DIR


def resolve_snapshot_root(snapshot_id: str, snapshots_root: Path | None = None) -> Path:
    return (snapshots_root or SNAPSHOTS_DIR) / snapshot_id


def resolve_prediction_paths(outputs_root: Path | None = None) -> dict[str, Path]:
    root = resolve_outputs_root(outputs_root)
    predictions_dir = root / "predictions"
    ablation_dir = predictions_dir / "ablations"
    return {
        "rule_baseline": predictions_dir / "rule_baseline_predictions.jsonl",
        "direct_llm": predictions_dir / "direct_baseline_predictions.jsonl",
        "cot_llm": predictions_dir / "cot_baseline_predictions.jsonl",
        "react_llm": predictions_dir / "react_baseline_predictions.jsonl",
        "astra_mvp": predictions_dir / "astra_predictions.jsonl",
        "astra_minus_retrieval": ablation_dir / "minus_retrieval.jsonl",
        "astra_minus_neutralizer": ablation_dir / "minus_neutralizer.jsonl",
        "astra_minus_verifier": ablation_dir / "minus_verifier.jsonl",
        "astra_minus_uncertainty_gate": ablation_dir / "minus_uncertainty_gate.jsonl",
        "astra_minus_analyst_prior": ablation_dir / "minus_analyst_prior.jsonl",
    }


def resolve_case_path(outputs_root: Path | None = None) -> Path:
    return resolve_outputs_root(outputs_root) / "cases" / "case_studies.jsonl"


def resolve_eval_paths(outputs_root: Path | None = None) -> dict[str, Path]:
    root = resolve_outputs_root(outputs_root) / "eval"
    return {
        "metrics": root / "nlp_metrics.json",
        "calibration": root / "calibration_bins.json",
        "phenomena": root / "phenomena_metrics.json",
    }


def resolve_backtest_paths(outputs_root: Path | None = None) -> dict[str, Path]:
    root = resolve_outputs_root(outputs_root) / "backtest"
    return {
        "metrics": root / "finance_metrics.json",
        "curve": root / "backtest_curve.csv",
        "regime_heatmap": root / "regime_heatmap.csv",
        "diagnostics": root / "diagnostics.json",
        "portfolio_returns": root / "portfolio_returns.csv",
        "robustness_grid": root / "robustness_grid.csv",
        "config": root / "finance_config.json",
    }


def resolve_intermediate_path(outputs_root: Path | None = None) -> Path:
    return resolve_outputs_root(outputs_root) / "intermediate" / "inference_rows.jsonl"


def resolve_snapshot_manifest_path(outputs_root: Path | None = None) -> Path:
    return resolve_outputs_root(outputs_root) / "manifest.json"


def resolve_market_prices_path(
    *,
    default_market_path: Path = MARKET_PRICES_PATH,
    experiment_market_path: Path | None = None,
) -> Path:
    if experiment_market_path is None:
        experiment_market_path = _resolve_experiment_lazy_path("EXPERIMENT_MARKET_PRICES_PATH")
    if experiment_market_path.exists():
        return experiment_market_path
    return default_market_path


def resolve_reports_input_path(split: str | None = None, *, prefer_experiment_split: bool = False) -> Path:
    if not prefer_experiment_split:
        return CLEAN_REPORTS_PATH

    split_attr_names = {
        "train": "EXPERIMENT_TRAIN_PATH",
        "dev": "EXPERIMENT_DEV_PATH",
        "test": "EXPERIMENT_TEST_PATH",
    }
    split_attr_name = split_attr_names.get(split)
    experiment_split_path = None
    if split_attr_name is not None:
        experiment_split_path = globals().get(split_attr_name)
        if experiment_split_path is None:
            experiment_split_path = _resolve_experiment_lazy_path(split_attr_name)
    if experiment_split_path is not None and experiment_split_path.exists():
        return experiment_split_path
    return CLEAN_REPORTS_PATH


def resolve_history_reports_path(
    *,
    clean_reports_path: Path = CLEAN_REPORTS_PATH,
    experiment_master_path: Path | None = None,
) -> Path:
    if experiment_master_path is None:
        experiment_master_path = globals().get("REPORTS_EXPERIMENT_MASTER_PATH")
        if experiment_master_path is None:
            experiment_master_path = _resolve_experiment_lazy_path("REPORTS_EXPERIMENT_MASTER_PATH")
    if experiment_master_path.exists():
        return experiment_master_path
    return clean_reports_path


@dataclass(frozen=True)
class SplitConfig:
    train_end_year: int = TRAIN_END_YEAR
    dev_year: int = DEV_YEAR
    test_year: int = TEST_YEAR

    def split_for_year(self, year: int) -> str | None:
        if year <= self.train_end_year:
            return "train"
        if year == self.dev_year:
            return "dev"
        if year == self.test_year:
            return "test"
        return None
