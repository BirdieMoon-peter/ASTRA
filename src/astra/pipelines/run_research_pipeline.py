from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from astra.pipelines.run_finance_eval import main as run_finance_eval
from astra.pipelines.run_nlp_eval import main as run_nlp_eval
from astra.paper.export_results import export_paper_artifacts
from astra.config.task_schema import resolve_snapshot_manifest_path


def _load_manifest(outputs_root: Path) -> dict[str, Any]:
    manifest_path = resolve_snapshot_manifest_path(outputs_root)
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing manifest: {manifest_path}")
    return json.loads(manifest_path.read_text(encoding='utf-8'))


def _save_manifest(outputs_root: Path, manifest: dict[str, Any]) -> None:
    manifest_path = resolve_snapshot_manifest_path(outputs_root)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')


def _inference_completed(manifest: dict[str, Any]) -> bool:
    stages = manifest.get('stages') or {}
    inference_stage = stages.get('inference') or {}
    if inference_stage.get('completed', False):
        return True
    # Backward compatibility for older manifests: if the manifest exists and
    # lists an ASTRA prediction path, treat the run as completed.
    paths = manifest.get('paths') or {}
    predictions = paths.get('predictions') or {}
    return bool(predictions.get('astra_mvp'))


def run_research_pipeline(outputs_root: Path, gold_path: Path | None = None) -> dict[str, Any]:
    manifest = _load_manifest(outputs_root)
    stages = manifest.setdefault('stages', {})
    if not _inference_completed(manifest):
        raise RuntimeError('Inference stage is not completed; refusing to continue.')

    stages['inference'] = {'status': 'completed', 'completed': True}
    manifest['run_status'] = 'locked'
    _save_manifest(outputs_root, manifest)

    run_nlp_eval(gold_path=gold_path, outputs_root=outputs_root)
    manifest = _load_manifest(outputs_root)
    manifest['stages']['nlp_eval'] = {'status': 'completed', 'completed': True}
    _save_manifest(outputs_root, manifest)

    run_finance_eval(outputs_root=outputs_root)
    manifest = _load_manifest(outputs_root)
    manifest['stages']['finance'] = {'status': 'completed', 'completed': True}
    _save_manifest(outputs_root, manifest)

    export_paper_artifacts(outputs_root=outputs_root)
    manifest = _load_manifest(outputs_root)
    manifest['stages']['paper_export'] = {'status': 'completed', 'completed': True}
    _save_manifest(outputs_root, manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--outputs-root', type=Path, required=True)
    parser.add_argument('--gold-path', type=Path, default=None)
    args = parser.parse_args()
    manifest = run_research_pipeline(args.outputs_root, gold_path=args.gold_path)
    print('[OK] Research pipeline completed.')
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
