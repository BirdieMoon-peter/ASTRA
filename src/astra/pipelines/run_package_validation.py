from __future__ import annotations

import json

from astra.config.task_schema import ANNOTATION_MAIN_PATH, EXPERIMENT_PACKAGE_DIR
from astra.data.package_validation import validate_experiment_package


def main() -> None:
    result = validate_experiment_package(
        package_dir=EXPERIMENT_PACKAGE_DIR,
        gold_path=ANNOTATION_MAIN_PATH,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
