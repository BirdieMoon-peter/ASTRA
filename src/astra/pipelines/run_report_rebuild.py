from __future__ import annotations

import json

from astra.ingestion.report_rebuilder import rebuild_reports_dataset


def main(begin_time: str = "2026-03-01", end_time: str = "2026-03-31", max_pages: int = 1, page_size: int = 50) -> None:
    result = rebuild_reports_dataset(
        begin_time=begin_time,
        end_time=end_time,
        max_pages=max_pages,
        page_size=page_size,
    )
    print("[OK] Report dataset rebuild completed.")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
