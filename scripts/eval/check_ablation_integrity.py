from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
PREDICTIONS_DIR = ROOT / "outputs" / "predictions"
ABLATIONS_DIR = PREDICTIONS_DIR / "ablations"
EVAL_PATH = ROOT / "outputs" / "eval" / "nlp_metrics.json"
FINANCE_PATH = ROOT / "outputs" / "backtest" / "finance_metrics.json"
REGISTRY_PATH = ROOT / "artifacts" / "ablation" / "ablation_run_registry.csv"
RESULTS_PATH = ROOT / "artifacts" / "ablation" / "ablation_results.csv"

STRATEGIC_F1_WEIGHT = 0.35
PHENOMENON_F1_WEIGHT = 0.20
EVIDENCE_F1_WEIGHT = 0.10
RANK_IC20_WEIGHT = 0.15
SHARPE20_WEIGHT = 0.15
ECE_WEIGHT = 0.15

FILES = {
    "astra_mvp": PREDICTIONS_DIR / "astra_predictions.jsonl",
    "astra_minus_retrieval": ABLATIONS_DIR / "minus_retrieval.jsonl",
    "astra_minus_neutralizer": ABLATIONS_DIR / "minus_neutralizer.jsonl",
    "astra_minus_verifier": ABLATIONS_DIR / "minus_verifier.jsonl",
    "astra_minus_uncertainty_gate": ABLATIONS_DIR / "minus_uncertainty_gate.jsonl",
    "astra_minus_analyst_prior": ABLATIONS_DIR / "minus_analyst_prior.jsonl",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _check_label_distributions(files: dict[str, Path]) -> dict[str, Any]:
    """Check if ablation variants produce identical categorical label distributions."""
    label_keys = ("fundamental_sentiment", "strategic_optimism", "phenomenon")
    distributions: dict[str, dict[str, list[str]]] = {}
    for name, path in files.items():
        if not path.exists():
            continue
        rows = []
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    rows.append(json.loads(line))
        distributions[name] = {key: [str(row.get(key, "")) for row in rows] for key in label_keys}

    full_dist = distributions.get("astra_mvp")
    if not full_dist:
        return {"status": "missing_full_model"}

    identical_to_full = []
    for name, dist in distributions.items():
        if name == "astra_mvp":
            continue
        all_same = all(dist.get(key, []) == full_dist.get(key, []) for key in label_keys)
        if all_same:
            identical_to_full.append(name)

    return {
        "ablations_with_identical_labels": identical_to_full,
        "label_integrity_ok": len(identical_to_full) == 0,
    }


def main() -> None:
    metrics = _load_json(EVAL_PATH)
    finance = _load_json(FINANCE_PATH)
    finance_ablation = finance.get("ablation_metrics", {})

    registry_rows = []
    result_rows = []

    full_hash = _sha256(FILES["astra_mvp"]) if FILES["astra_mvp"].exists() else ""

    for name, path in FILES.items():
        exists = path.exists()
        file_hash = _sha256(path) if exists else ""
        same_as_full = bool(full_hash and file_hash and file_hash == full_hash and name != "astra_mvp")
        nlp_row = metrics.get(name, {})
        finance_row = finance_ablation.get(name, {}) if name != "astra_mvp" else finance.get("signal_metrics", {}).get("astra_finance_plus_gap", {})

        registry_rows.append(
            {
                "setting": name,
                "path": str(path),
                "exists": exists,
                "sha256": file_hash,
                "same_hash_as_full_model": same_as_full,
            }
        )

        result_rows.append(
            {
                "setting": name,
                "prediction_exists": exists,
                "same_hash_as_full_model": same_as_full,
                "strategic_f1": nlp_row.get("strategic_optimism_macro_f1", ""),
                "evidence_f1": nlp_row.get("evidence_f1", ""),
                "ece": nlp_row.get("ece", ""),
                "mean_rank_ic_20": finance_row.get("mean_rank_ic@20", ""),
                "ls_sharpe_20": finance_row.get("ls_sharpe@20", ""),
                "turnover": finance_row.get("turnover", ""),
                "composite_j": round(
                    STRATEGIC_F1_WEIGHT * float(nlp_row.get("strategic_optimism_macro_f1", 0.0) or 0.0)
                    + PHENOMENON_F1_WEIGHT * float(nlp_row.get("phenomenon_macro_f1", 0.0) or 0.0)
                    + EVIDENCE_F1_WEIGHT * float(nlp_row.get("evidence_f1", 0.0) or 0.0)
                    + RANK_IC20_WEIGHT * float(finance_row.get("mean_rank_ic@20", 0.0) or 0.0)
                    + SHARPE20_WEIGHT * float(finance_row.get("ls_sharpe@20", 0.0) or 0.0)
                    - ECE_WEIGHT * float(nlp_row.get("ece", 0.0) or 0.0),
                    6,
                ),
            }
        )

    _write_csv(REGISTRY_PATH, registry_rows, ["setting", "path", "exists", "sha256", "same_hash_as_full_model"])
    _write_csv(RESULTS_PATH, result_rows, ["setting", "prediction_exists", "same_hash_as_full_model", "strategic_f1", "evidence_f1", "ece", "mean_rank_ic_20", "ls_sharpe_20", "turnover", "composite_j"])

    identical_metric_groups = {}
    for field in ("strategic_f1", "evidence_f1", "ece", "mean_rank_ic_20", "ls_sharpe_20", "turnover"):
        buckets: dict[str, list[str]] = {}
        for row in result_rows:
            value = str(row[field])
            buckets.setdefault(value, []).append(row["setting"])
        identical_metric_groups[field] = {k: v for k, v in buckets.items() if len(v) > 1}

    summary = {
        "registry_path": str(REGISTRY_PATH),
        "results_path": str(RESULTS_PATH),
        "identical_metric_groups": identical_metric_groups,
        "label_integrity": _check_label_distributions(FILES),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
