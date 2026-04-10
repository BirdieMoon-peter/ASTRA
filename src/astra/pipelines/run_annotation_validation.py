from __future__ import annotations

import json

from astra.config.task_schema import ANNOTATION_GOLD_WORKSET_PATH
from astra.labeling.validate_annotations import validate_annotation_file


def main() -> None:
    result = validate_annotation_file(ANNOTATION_GOLD_WORKSET_PATH)
    if result.get("ready_to_freeze"):
        print("[OK] Gold annotation validation complete.")
    else:
        print("[WARN] Gold annotation validation found issues.")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
