from __future__ import annotations

import json

from astra.data.build_dataset import build_split_summary, export_split_files
from astra.data.clean_reports import clean_reports


def main() -> None:
    corpus_stats = clean_reports()
    split_summary = build_split_summary()
    export_split_files()

    print("[OK] Data preparation complete.")
    print(json.dumps({"corpus_stats": corpus_stats, "split_summary": split_summary}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
