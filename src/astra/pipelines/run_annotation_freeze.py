from __future__ import annotations

import json

from astra.labeling.freeze_gold_annotations import freeze_gold_annotations


def main() -> None:
    result = freeze_gold_annotations()
    print("[OK] Gold annotation file frozen.")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
