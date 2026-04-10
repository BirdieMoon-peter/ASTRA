from __future__ import annotations

import json

from astra.ingestion.report_rebuilder import rebuild_reports_dataset_full_range


def main(start_year: int = 2016, end_year: int = 2026, page_size: int = 100, resume: bool = True) -> None:
    result = rebuild_reports_dataset_full_range(
        start_year=start_year,
        end_year=end_year,
        page_size=page_size,
        resume=resume,
    )
    print("[OK] Full-range report dataset rebuild completed.")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
