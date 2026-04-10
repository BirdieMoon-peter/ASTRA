from __future__ import annotations

import csv
import importlib
import json
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests

from astra.config.task_schema import (
    MARKET_PRICES_MISSING_PATH,
    MARKET_PRICES_PATH,
    MARKET_PRICES_STATE_PATH,
    MARKET_PRICES_STATUS_PATH,
    REPORTS_MASTER_PATH,
)

FIELDNAMES = [
    "stock_code",
    "trade_date",
    "open",
    "close",
    "high",
    "low",
    "volume",
    "amount",
    "amplitude",
    "pct_change",
    "change",
    "turnover",
]
STATUS_FIELDNAMES = [
    "stock_code",
    "secid",
    "status",
    "row_count",
    "source_primary",
    "source_used",
    "fallback_used",
    "attempt_count",
    "message",
]
SUCCESS_STATUSES = {"ok", "fallback_ok"}
FAILED_STATUSES = {"request_failed", "parse_failed", "fallback_failed"}
EASTMONEY_KLINE_URL = "https://push2his.eastmoney.com/api/qt/stock/kline/get"
DEFAULT_HEADERS = {
    "User-Agent": "Mozilla/5.0",
    "Referer": "https://quote.eastmoney.com/",
}


@dataclass(frozen=True)
class FetchOutcome:
    rows: list[dict[str, Any]]
    status: str
    source_used: str
    fallback_used: bool
    attempt_count: int
    message: str = ""


def _secid(stock_id: str) -> str:
    normalized = stock_id.strip()
    if normalized.startswith(("6", "9")):
        return f"1.{normalized}"
    return f"0.{normalized}"


def _load_stock_ids_from_reports_master(path: Path = REPORTS_MASTER_PATH) -> list[str]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        return sorted({(row.get("stock_id") or "").strip() for row in reader if (row.get("stock_id") or "").strip()})


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _load_state(state_path: Path) -> dict[str, Any] | None:
    if not state_path.exists():
        return None
    with state_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _load_status_rows(status_path: Path) -> list[dict[str, Any]]:
    if not status_path.exists():
        return []
    with status_path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _replace_stock_rows(rows: list[dict[str, Any]], stock_id: str, new_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    kept = [row for row in rows if row["stock_code"] != stock_id]
    kept.extend(new_rows)
    return kept


def _append_price_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = path.exists()
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        if not file_exists or path.stat().st_size == 0:
            writer.writeheader()
        writer.writerows(rows)


def _replace_status_row(status_rows: list[dict[str, Any]], replacement: dict[str, Any]) -> list[dict[str, Any]]:
    filtered = [row for row in status_rows if row["stock_code"] != replacement["stock_code"]]
    filtered.append(replacement)
    return filtered


def _dump_state(
    state_path: Path,
    *,
    begin_date: str,
    end_date: str,
    processed_stocks: list[str],
    rows: list[dict[str, Any]],
    status_rows: list[dict[str, Any]],
) -> None:
    state = {
        "begin_date": begin_date,
        "end_date": end_date,
        "processed_stocks": processed_stocks,
        "rows": rows,
        "status_rows": status_rows,
    }
    state_path.parent.mkdir(parents=True, exist_ok=True)
    with state_path.open("w", encoding="utf-8") as handle:
        json.dump(state, handle, ensure_ascii=False)


def _checkpoint(
    *,
    output_path: Path,
    status_path: Path,
    missing_path: Path,
    rows: list[dict[str, Any]],
    status_rows: list[dict[str, Any]],
    write_prices: bool = True,
) -> None:
    if write_prices:
        _write_csv(output_path, rows, FIELDNAMES)
    _write_csv(status_path, status_rows, STATUS_FIELDNAMES)
    missing_rows = [row for row in status_rows if row["status"] not in SUCCESS_STATUSES]
    _write_csv(missing_path, missing_rows, STATUS_FIELDNAMES)


def _stringify(value: Any) -> str:
    if value is None:
        return ""
    try:
        if value != value:
            return ""
    except Exception:
        pass
    text = str(value).strip()
    if text.lower() in {"nan", "none"}:
        return ""
    return text


def _normalize_trade_date(value: Any) -> str:
    text = _stringify(value)
    if not text:
        return ""
    return text[:10].replace("/", "-")


def _row_from_source_record(stock_id: str, record: dict[str, Any]) -> dict[str, Any] | None:
    trade_date = _normalize_trade_date(record.get("日期") or record.get("date"))
    if not trade_date:
        return None
    return {
        "stock_code": stock_id,
        "trade_date": trade_date,
        "open": _stringify(record.get("开盘") or record.get("open")),
        "close": _stringify(record.get("收盘") or record.get("close")),
        "high": _stringify(record.get("最高") or record.get("high")),
        "low": _stringify(record.get("最低") or record.get("low")),
        "volume": _stringify(record.get("成交量") or record.get("volume")),
        "amount": _stringify(record.get("成交额") or record.get("amount")),
        "amplitude": _stringify(record.get("振幅") or record.get("amplitude")),
        "pct_change": _stringify(record.get("涨跌幅") or record.get("pctChg") or record.get("pct_change")),
        "change": _stringify(record.get("涨跌额") or record.get("change")),
        "turnover": _stringify(record.get("换手率") or record.get("turn") or record.get("turnover")),
    }


def _normalize_source_rows(stock_id: str, records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for record in records:
        row = _row_from_source_record(stock_id, record)
        if row is not None:
            rows.append(row)
    return rows


def _classify_exception(exc: Exception) -> str:
    if isinstance(exc, (ValueError, KeyError, json.JSONDecodeError, TypeError)):
        return "parse_failed"
    return "request_failed"


def _looks_like_etf(stock_id: str) -> bool:
    normalized = stock_id.strip()
    return normalized.startswith(("5", "15", "16", "18"))


def _can_use_baostock(stock_id: str) -> bool:
    normalized = stock_id.strip()
    return normalized.startswith(("6", "0", "3"))


def _eastmoney_secid(stock_id: str) -> str:
    normalized = stock_id.strip()
    if normalized.startswith(("6", "688", "689", "9")):
        return f"1.{normalized}"
    return f"0.{normalized}"


def _should_use_eastmoney_before_baostock(stock_id: str) -> bool:
    normalized = stock_id.strip()
    return normalized.startswith(("688", "689", "920", "83", "87", "430", "200"))


def _baostock_code(stock_id: str) -> str | None:
    normalized = stock_id.strip()
    if normalized.startswith("6"):
        return f"sh.{normalized}"
    if normalized.startswith(("0", "3")):
        return f"sz.{normalized}"
    return None


def _import_akshare() -> Any:
    return importlib.import_module("akshare")


def _ensure_baostock_logged_in(source_clients: dict[str, Any]) -> Any:
    baostock = source_clients.get("baostock")
    if baostock is None:
        baostock = importlib.import_module("baostock")
        source_clients["baostock"] = baostock
    if not source_clients.get("baostock_logged_in"):
        login_result = baostock.login()
        if getattr(login_result, "error_code", "0") != "0":
            raise RuntimeError(getattr(login_result, "error_msg", "baostock login failed"))
        source_clients["baostock_logged_in"] = True
    return baostock


def _logout_baostock(source_clients: dict[str, Any]) -> None:
    baostock = source_clients.get("baostock")
    if baostock is None or not source_clients.get("baostock_logged_in"):
        return
    try:
        baostock.logout()
    finally:
        source_clients["baostock_logged_in"] = False


def _records_from_dataframe(frame: Any) -> list[dict[str, Any]]:
    if frame is None:
        return []
    if getattr(frame, "empty", False):
        return []
    if not hasattr(frame, "to_dict"):
        raise TypeError("source result is not dataframe-like")
    return list(frame.to_dict("records"))


def _fetch_one_stock_akshare(
    akshare: Any,
    *,
    stock_id: str,
    begin_date: str,
    end_date: str,
    max_attempts: int,
    retry_delay_seconds: float,
) -> FetchOutcome:
    start_date = begin_date.replace("-", "")
    finish_date = end_date.replace("-", "")
    last_exc: Exception | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            if _looks_like_etf(stock_id):
                frame = akshare.fund_etf_hist_em(
                    symbol=stock_id,
                    period="daily",
                    start_date=start_date,
                    end_date=finish_date,
                    adjust="",
                )
            else:
                frame = akshare.stock_zh_a_hist(
                    symbol=stock_id,
                    period="daily",
                    start_date=start_date,
                    end_date=finish_date,
                    adjust="",
                )
            rows = _normalize_source_rows(stock_id, _records_from_dataframe(frame))
            if not rows:
                return FetchOutcome([], "empty", "akshare", False, attempt)
            return FetchOutcome(rows, "ok", "akshare", False, attempt)
        except Exception as exc:
            last_exc = exc
            if attempt < max_attempts:
                time.sleep(retry_delay_seconds)
                continue
            raise
    if last_exc is not None:
        raise last_exc
    return FetchOutcome([], "empty", "akshare", False, max_attempts)


def _fetch_one_stock_eastmoney(
    session: requests.Session,
    *,
    stock_id: str,
    begin_date: str,
    end_date: str,
    max_attempts: int,
    retry_delay_seconds: float,
) -> FetchOutcome:
    secid = _eastmoney_secid(stock_id)
    params = {
        "secid": secid,
        "fields1": "f1,f2,f3,f4,f5,f6",
        "fields2": "f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61",
        "klt": "101",
        "fqt": "1",
        "beg": begin_date.replace("-", ""),
        "end": end_date.replace("-", ""),
    }
    last_exc: Exception | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            response = session.get(EASTMONEY_KLINE_URL, params=params, timeout=30)
            response.raise_for_status()
            payload = response.json().get("data") or {}
            klines = payload.get("klines") or []
            if not klines:
                return FetchOutcome([], "empty", "eastmoney", True, attempt)
            rows = []
            for item in klines:
                trade_date, open_, close, high, low, volume, amount, amplitude, pct_chg, change, turnover = item.split(",")
                rows.append(
                    {
                        "stock_code": stock_id,
                        "trade_date": trade_date,
                        "open": open_,
                        "close": close,
                        "high": high,
                        "low": low,
                        "volume": volume,
                        "amount": amount,
                        "amplitude": amplitude,
                        "pct_change": pct_chg,
                        "change": change,
                        "turnover": turnover,
                    }
                )
            return FetchOutcome(rows, "ok", "eastmoney", True, attempt)
        except Exception as exc:
            last_exc = exc
            if attempt < max_attempts:
                time.sleep(retry_delay_seconds)
                continue
            raise
    if last_exc is not None:
        raise last_exc
    return FetchOutcome([], "empty", "eastmoney", True, max_attempts)


def _fetch_one_stock_baostock(
    source_clients: dict[str, Any],
    *,
    stock_id: str,
    begin_date: str,
    end_date: str,
    max_attempts: int,
    retry_delay_seconds: float,
) -> FetchOutcome:
    code = _baostock_code(stock_id)
    if code is None:
        raise ValueError(f"baostock unsupported stock code: {stock_id}")
    last_exc: Exception | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            baostock = _ensure_baostock_logged_in(source_clients)
            result = baostock.query_history_k_data_plus(
                code,
                "date,code,open,high,low,close,volume,amount,turn,pctChg",
                start_date=begin_date,
                end_date=end_date,
                frequency="d",
                adjustflag="3",
            )
            if getattr(result, "error_code", "0") != "0":
                raise RuntimeError(getattr(result, "error_msg", "baostock query failed"))
            records = []
            while result.next():
                records.append(dict(zip(result.fields, result.get_row_data(), strict=False)))
            rows = _normalize_source_rows(stock_id, records)
            if not rows:
                return FetchOutcome([], "empty", "baostock", True, attempt)
            return FetchOutcome(rows, "ok", "baostock", True, attempt)
        except Exception as exc:
            last_exc = exc
            if attempt < max_attempts:
                time.sleep(retry_delay_seconds)
                continue
            raise
    if last_exc is not None:
        raise last_exc
    return FetchOutcome([], "empty", "baostock", True, max_attempts)


def _fetch_one_stock(
    source_clients: dict[str, Any],
    *,
    stock_id: str,
    begin_date: str,
    end_date: str,
    max_attempts: int,
    retry_delay_seconds: float,
) -> FetchOutcome:
    akshare = source_clients["akshare"]
    eastmoney_session = source_clients["eastmoney_session"]
    try:
        return _fetch_one_stock_akshare(
            akshare,
            stock_id=stock_id,
            begin_date=begin_date,
            end_date=end_date,
            max_attempts=max_attempts,
            retry_delay_seconds=retry_delay_seconds,
        )
    except Exception as primary_exc:
        attempts_used = max_attempts
        primary_message = f"akshare: {primary_exc}"
        primary_status = _classify_exception(primary_exc)

        if _should_use_eastmoney_before_baostock(stock_id):
            try:
                eastmoney = _fetch_one_stock_eastmoney(
                    eastmoney_session,
                    stock_id=stock_id,
                    begin_date=begin_date,
                    end_date=end_date,
                    max_attempts=max_attempts,
                    retry_delay_seconds=retry_delay_seconds,
                )
                attempts_used += eastmoney.attempt_count
                if eastmoney.status == "ok":
                    return FetchOutcome(eastmoney.rows, "fallback_ok", "eastmoney", True, attempts_used, primary_message)
                if eastmoney.status == "empty":
                    return FetchOutcome([], "empty", "eastmoney", True, attempts_used, primary_message)
            except Exception as eastmoney_exc:
                primary_message = f"{primary_message} | eastmoney: {eastmoney_exc}"

        if not _can_use_baostock(stock_id):
            return FetchOutcome([], primary_status, "akshare", False, attempts_used, primary_message)
        try:
            fallback = _fetch_one_stock_baostock(
                source_clients,
                stock_id=stock_id,
                begin_date=begin_date,
                end_date=end_date,
                max_attempts=max_attempts,
                retry_delay_seconds=retry_delay_seconds,
            )
            attempts_used += fallback.attempt_count
            if fallback.status == "ok":
                return FetchOutcome(
                    fallback.rows,
                    "fallback_ok",
                    "baostock",
                    True,
                    attempts_used,
                    primary_message,
                )
            if fallback.status == "empty":
                return FetchOutcome(
                    [],
                    "empty",
                    "baostock",
                    True,
                    attempts_used,
                    primary_message,
                )
            return FetchOutcome([], fallback.status, fallback.source_used, True, attempts_used, primary_message or fallback.message)
        except Exception as fallback_exc:
            return FetchOutcome(
                [],
                "fallback_failed",
                "baostock",
                True,
                attempts_used + max_attempts,
                f"{primary_message} | baostock: {fallback_exc}",
            )


def fetch_daily_prices(
    *,
    stock_ids: list[str] | None = None,
    begin_date: str = "2017-01-01",
    end_date: str = "2026-03-31",
    output_path: Path = MARKET_PRICES_PATH,
    state_path: Path = MARKET_PRICES_STATE_PATH,
    status_path: Path = MARKET_PRICES_STATUS_PATH,
    missing_path: Path = MARKET_PRICES_MISSING_PATH,
    resume: bool = True,
    checkpoint_every: int = 100,
    retry_failed_only: bool = False,
    max_attempts: int = 3,
    retry_delay_seconds: float = 1.0,
) -> dict[str, Any]:
    if stock_ids is None:
        stock_ids = _load_stock_ids_from_reports_master()
    requested_stock_ids = list(dict.fromkeys(stock_ids))

    eastmoney_session = requests.Session()
    eastmoney_session.headers.update(DEFAULT_HEADERS)
    source_clients = {
        "akshare": _import_akshare(),
        "baostock": None,
        "baostock_logged_in": False,
        "eastmoney_session": eastmoney_session,
    }

    rows: list[dict[str, Any]] = []
    status_rows: list[dict[str, Any]] = []
    processed_stocks: list[str] = []
    resumed_from_state = False
    retry_failed_append_mode = retry_failed_only and output_path.exists() and status_path.exists()

    if state_path.exists() and not retry_failed_append_mode:
        state = _load_state(state_path)
        if state and state.get("begin_date") == begin_date and state.get("end_date") == end_date:
            rows = list(state.get("rows", []))
            status_rows = list(state.get("status_rows", []))
            processed_stocks = list(state.get("processed_stocks", []))
            resumed_from_state = resume

    if not status_rows and status_path.exists():
        status_rows = _load_status_rows(status_path)
        processed_stocks = [row["stock_code"] for row in status_rows]

    if retry_failed_only:
        failed_codes = {
            row["stock_code"]
            for row in status_rows
            if row["status"] in FAILED_STATUSES
        }
        remaining_stock_ids = [stock_id for stock_id in requested_stock_ids if stock_id in failed_codes]
    elif resume:
        processed_set = set(processed_stocks)
        remaining_stock_ids = [stock_id for stock_id in requested_stock_ids if stock_id not in processed_set]
    else:
        remaining_stock_ids = requested_stock_ids
        rows = []
        status_rows = []
        processed_stocks = []

    pending_price_rows: list[dict[str, Any]] = []

    try:
        for index, stock_id in enumerate(remaining_stock_ids, start=1):
            outcome = _fetch_one_stock(
                source_clients,
                stock_id=stock_id,
                begin_date=begin_date,
                end_date=end_date,
                max_attempts=max_attempts,
                retry_delay_seconds=retry_delay_seconds,
            )

            if retry_failed_append_mode:
                pending_price_rows.extend(outcome.rows)
            else:
                rows = _replace_stock_rows(rows, stock_id, outcome.rows)
            status_rows = _replace_status_row(
                status_rows,
                {
                    "stock_code": stock_id,
                    "secid": _secid(stock_id),
                    "status": outcome.status,
                    "row_count": len(outcome.rows),
                    "source_primary": "akshare",
                    "source_used": outcome.source_used,
                    "fallback_used": str(outcome.fallback_used).lower(),
                    "attempt_count": outcome.attempt_count,
                    "message": outcome.message,
                },
            )
            if stock_id not in processed_stocks:
                processed_stocks.append(stock_id)

            if index % checkpoint_every == 0 or index == len(remaining_stock_ids):
                if retry_failed_append_mode and pending_price_rows:
                    _append_price_rows(output_path, pending_price_rows)
                    pending_price_rows = []
                _checkpoint(
                    output_path=output_path,
                    status_path=status_path,
                    missing_path=missing_path,
                    rows=rows,
                    status_rows=status_rows,
                    write_prices=not retry_failed_append_mode,
                )
                if not retry_failed_append_mode:
                    _dump_state(
                        state_path,
                        begin_date=begin_date,
                        end_date=end_date,
                        processed_stocks=processed_stocks,
                        rows=rows,
                        status_rows=status_rows,
                    )
    finally:
        _logout_baostock(source_clients)

    if not remaining_stock_ids:
        _checkpoint(
            output_path=output_path,
            status_path=status_path,
            missing_path=missing_path,
            rows=rows,
            status_rows=status_rows,
            write_prices=not retry_failed_append_mode,
        )

    row_count = 0
    if retry_failed_append_mode:
        with output_path.open("r", encoding="utf-8", newline="") as handle:
            row_count = sum(1 for _ in csv.DictReader(handle))
    else:
        row_count = len(rows)

    status_counts = Counter(row["status"] for row in status_rows)
    covered = sum(status_counts.get(key, 0) for key in SUCCESS_STATUSES)
    missing = status_counts.get("empty", 0)
    failed = sum(count for key, count in status_counts.items() if key not in SUCCESS_STATUSES | {"empty"})
    return {
        "status": "ok",
        "stock_count_requested": len(requested_stock_ids),
        "stock_count_processed": len(status_rows),
        "stock_count_covered": covered,
        "stock_count_missing": missing,
        "stock_count_failed": failed,
        "stock_count_retried": len(remaining_stock_ids),
        "row_count": row_count,
        "resumed_from_state": resumed_from_state,
        "retry_failed_only": retry_failed_only,
        "source_policy": "akshare_then_baostock",
        "output_path": str(output_path),
        "status_path": str(status_path),
        "missing_path": str(missing_path),
        "state_path": str(state_path),
        "status_counts": dict(sorted(status_counts.items())),
    }
