from __future__ import annotations

import json

from astra.ingestion.market_rebuilder import fetch_daily_prices


def main(
    stock_ids: list[str] | None = None,
    begin_date: str = "2017-01-01",
    end_date: str = "2026-03-31",
    resume: bool = True,
    retry_failed_only: bool = False,
) -> None:
    result = fetch_daily_prices(
        stock_ids=stock_ids,
        begin_date=begin_date,
        end_date=end_date,
        resume=resume,
        retry_failed_only=retry_failed_only,
    )
    print("[OK] Market dataset rebuild completed.")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
