from __future__ import annotations

from pathlib import Path

from astra.config.task_schema import ANNOTATION_GOLD_WORKSET_PATH, ANNOTATION_MAIN_PATH
from astra.labeling.validate_annotations import validate_annotation_file


def freeze_gold_annotations(
    *,
    workset_path: Path = ANNOTATION_GOLD_WORKSET_PATH,
    output_path: Path = ANNOTATION_MAIN_PATH,
) -> dict[str, object]:
    result = validate_annotation_file(workset_path)
    if not result.get("ready_to_freeze"):
        raise ValueError(f"Gold annotation workset is not ready to freeze: {workset_path}")
    if output_path.exists():
        raise ValueError(f"Gold annotation output already exists: {output_path}")

    content = workset_path.read_text(encoding="utf-8")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    temp_path.write_text(content, encoding="utf-8")
    temp_path.replace(output_path)
    return {
        "row_count": result["row_count"],
        "output_path": str(output_path),
        "source_path": str(workset_path),
    }
