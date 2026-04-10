from __future__ import annotations

import csv
import hashlib
import json
import re
import subprocess
import tempfile
from calendar import monthrange
from collections import Counter
from dataclasses import replace
from datetime import datetime
from html import unescape
from pathlib import Path
from typing import Any

import requests

from astra.config.task_schema import (
    RAW_REPORTS_PATH,
    REPORT_ANALYSTS_PATH,
    REPORT_ANALYST_BRIDGE_PATH,
    REPORT_BROKER_ALIASES_PATH,
    REPORT_BROKERS_PATH,
    REPORT_RATINGS_PATH,
    REPORT_REBUILD_STATE_PATH,
    REPORTS_MASTER_PATH,
    REPORT_VERSIONS_PATH,
)
from astra.ingestion.schemas import (
    ReportAnalystBridgeRecord,
    ReportAnalystRecord,
    ReportBrokerAliasRecord,
    ReportBrokerRecord,
    ReportRatingRecord,
    ReportRecord,
    ReportVersionRecord,
)

EASTMONEY_LIST_URL = "https://reportapi.eastmoney.com/report/list"
EASTMONEY_DETAIL_URL_TEMPLATE = "https://data.eastmoney.com/report/info/{info_code}.html"
DEFAULT_HEADERS = {
    "User-Agent": "Mozilla/5.0",
    "Referer": "https://data.eastmoney.com/report/stock.jshtml",
}
DETAIL_CONTENT_RE = re.compile(r'<div id="ctx-content" class="ctx-content">(.*?)</div>\s*<div class="c-foot">', re.S)
DETAIL_SECTION_RE = re.compile(r'<div[^>]+(?:id|class)="[^"]*(?:ctx-content|txtinfos|newsContent|stockcodec|report-content)[^"]*"[^>]*>(.*?)</div>', re.S)
PDF_LINK_RE = re.compile(r'href="(https://pdf\.dfcfw\.com/pdf/[^"]+)"')
META_KEYWORDS_RE = re.compile(r'<meta\s+name="keywords"\s+content="([^"]+)"', re.I)
PARAGRAPH_RE = re.compile(r"<p[^>]*>(.*?)</p>", re.S)
TAG_RE = re.compile(r"<[^>]+>")
WHITESPACE_RE = re.compile(r"\s+")
MAX_PAGE_SIZE = 100


def _normalize_text(value: str) -> str:
    text = unescape(value or "")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = TAG_RE.sub("", text)
    lines = [WHITESPACE_RE.sub(" ", line).strip() for line in text.split("\n")]
    lines = [line for line in lines if line]
    return "\n".join(lines)


def _normalize_inline_text(value: str) -> str:
    return WHITESPACE_RE.sub(" ", _normalize_text(value)).strip()


def _safe_string(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _iso_now() -> str:
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def _build_hash(*parts: str) -> str:
    payload = "||".join(parts)
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()


def _build_report_id(stock_id: str, broker_id: str, publish_time: str, title: str, source_url: str) -> str:
    return _build_hash(stock_id, broker_id, publish_time, title, source_url)[:20]


def _extract_analysts(row: dict[str, Any]) -> tuple[str, str]:
    author_ids = row.get("authorID") or []
    names = []
    ids = []
    for entry in row.get("author") or []:
        entry = _safe_string(entry)
        if not entry:
            continue
        if "." in entry:
            maybe_id, maybe_name = entry.split(".", 1)
            ids.append(maybe_id.strip())
            names.append(maybe_name.strip())
        else:
            names.append(entry)
    if author_ids:
        ids = [_safe_string(item) for item in author_ids if _safe_string(item)]
    researcher = _safe_string(row.get("researcher"))
    if not names and researcher:
        names = [name.strip() for name in researcher.split(",") if name.strip()]
    analyst_id = ",".join(ids)
    analyst_name = ",".join(names)
    return analyst_id, analyst_name


def _extract_list_summary(row: dict[str, Any]) -> str:
    candidates = (
        "summary",
        "abstract",
        "digest",
        "content",
        "contentAbstract",
        "reportSummary",
        "summaryContent",
        "shortContent",
    )
    for key in candidates:
        value = _normalize_text(_safe_string(row.get(key)))
        if value:
            return value[:600]
    return ""


def _extract_target_price(row: dict[str, Any]) -> str:
    upper = _safe_string(row.get("indvAimPriceT"))
    lower = _safe_string(row.get("indvAimPriceL"))
    if upper and lower:
        return f"{lower}-{upper}"
    return upper or lower


def _split_risk_section(body_text: str) -> tuple[str, str]:
    if not body_text:
        return "", ""
    markers = ["风险提示：", "风险提示", "风险因素：", "风险因素", "风险警示：", "风险警示"]
    for marker in markers:
        if marker in body_text:
            before, after = body_text.split(marker, 1)
            body = before.strip()
            risk = (marker + after.strip()).strip()
            return body, risk
    return body_text.strip(), ""


def _extract_summary(body_text: str) -> str:
    if not body_text:
        return ""
    paragraphs = [line.strip() for line in body_text.split("\n") if line.strip()]
    if not paragraphs:
        return ""
    skip_prefixes = ("投资要点", "核心观点", "事件：", "事项：")
    useful = []
    for line in paragraphs:
        normalized = line.lstrip("l•●·- ")
        if normalized in skip_prefixes:
            continue
        if any(normalized.startswith(prefix) for prefix in skip_prefixes) and len(normalized) <= 8:
            continue
        useful.append(normalized)
        if len(" ".join(useful)) >= 180:
            break
    summary = "\n".join(useful[:3]).strip()
    return summary[:600]


def _fetch_list_page(session: requests.Session, page_no: int, begin_time: str, end_time: str, page_size: int) -> dict[str, Any]:
    params = {
        "industryCode": "*",
        "pageSize": page_size,
        "industry": "*",
        "rating": "",
        "beginTime": begin_time,
        "endTime": end_time,
        "pageNo": page_no,
        "fields": "",
        "qType": 0,
        "ratingChange": "",
        "orgCode": "",
        "code": "*",
        "rcode": "",
        "cb": "callback",
    }
    response = session.get(EASTMONEY_LIST_URL, params=params, timeout=30)
    response.raise_for_status()
    text = response.text.strip()
    prefix = "callback("
    suffix = ")"
    if not text.startswith(prefix) or not text.endswith(suffix):
        raise ValueError("Unexpected response payload from Eastmoney list endpoint")
    return json.loads(text[len(prefix):-len(suffix)])


def _fetch_detail_html(session: requests.Session, info_code: str) -> str:
    url = EASTMONEY_DETAIL_URL_TEMPLATE.format(info_code=info_code)
    response = session.get(url, timeout=30)
    response.raise_for_status()
    return response.text


def _extract_body_from_block(content_block: str) -> str:
    paragraphs = []
    for paragraph in PARAGRAPH_RE.findall(content_block):
        text = _normalize_inline_text(paragraph)
        if text:
            paragraphs.append(text)
    if paragraphs:
        return "\n".join(paragraphs)
    return _normalize_text(content_block)


def _extract_pdf_url(html: str) -> str:
    match = PDF_LINK_RE.search(html)
    return unescape(match.group(1)) if match else ""


def _extract_pdf_text(session: requests.Session, pdf_url: str) -> str:
    if not pdf_url:
        return ""
    response = session.get(pdf_url, timeout=60)
    response.raise_for_status()
    with tempfile.TemporaryDirectory() as tmpdir:
        pdf_path = Path(tmpdir) / "report.pdf"
        txt_path = Path(tmpdir) / "report.txt"
        pdf_path.write_bytes(response.content)
        subprocess.run(
            ["pdftotext", "-layout", str(pdf_path), str(txt_path)],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        if not txt_path.exists():
            return ""
        return _normalize_text(txt_path.read_text(encoding="utf-8", errors="ignore"))


def _extract_body_from_html(html: str, session: requests.Session) -> tuple[str, str]:
    match = DETAIL_CONTENT_RE.search(html)
    if match:
        body_text = _extract_body_from_block(match.group(1))
        if body_text:
            return body_text, "html"

    for section in DETAIL_SECTION_RE.findall(html):
        body_text = _extract_body_from_block(section)
        if body_text:
            return body_text, "html_fallback"

    pdf_url = _extract_pdf_url(html)
    if pdf_url:
        try:
            body_text = _extract_pdf_text(session, pdf_url)
        except (requests.RequestException, subprocess.SubprocessError, OSError):
            body_text = ""
        if body_text:
            return body_text, "pdf"

    meta_match = META_KEYWORDS_RE.search(html)
    if meta_match:
        keywords = [item.strip() for item in meta_match.group(1).split(",") if item.strip()]
        body_text = _normalize_text("\n".join(keywords[1:]))
        if body_text:
            return body_text, "meta"

    return "", "missing"


def _canonical_broker_record(existing: ReportBrokerRecord | None, candidate: ReportBrokerRecord, publish_time: str) -> ReportBrokerRecord:
    if existing is None:
        return candidate
    existing_key = (existing.updated_at, existing.broker_name)
    candidate_key = (publish_time, candidate.broker_name)
    if candidate_key >= existing_key:
        return replace(candidate, updated_at=publish_time or candidate.updated_at)
    return existing


def _merge_broker_alias(existing: ReportBrokerAliasRecord | None, candidate: ReportBrokerAliasRecord) -> ReportBrokerAliasRecord:
    if existing is None:
        return candidate
    first_seen_at = min(filter(None, [existing.first_seen_at, candidate.first_seen_at]), default="")
    last_seen_at = max(filter(None, [existing.last_seen_at, candidate.last_seen_at]), default="")
    created_at = min(filter(None, [existing.created_at, candidate.created_at]), default="")
    updated_at = max(filter(None, [existing.updated_at, candidate.updated_at]), default="")
    return ReportBrokerAliasRecord(
        broker_id=candidate.broker_id,
        broker_name=candidate.broker_name,
        first_seen_at=first_seen_at,
        last_seen_at=last_seen_at,
        created_at=created_at,
        updated_at=updated_at,
    )


def _build_record(
    row: dict[str, Any],
    body_text: str,
    list_summary: str,
    fetched_at: str,
    info_code: str,
) -> tuple[
    ReportRecord,
    ReportRatingRecord,
    list[ReportAnalystRecord],
    list[ReportAnalystBridgeRecord],
    ReportBrokerRecord,
    ReportBrokerAliasRecord,
    ReportVersionRecord,
]:
    stock_id = _safe_string(row.get("stockCode"))
    stock_name = _safe_string(row.get("stockName"))
    broker_id = _safe_string(row.get("orgCode"))
    broker_name = _safe_string(row.get("orgSName")) or _safe_string(row.get("orgName"))
    analyst_id, analyst_name = _extract_analysts(row)
    publish_time = _safe_string(row.get("publishDate"))
    title = _safe_string(row.get("title"))
    source_url = EASTMONEY_DETAIL_URL_TEMPLATE.format(info_code=info_code)
    body_raw, risk_section_raw = _split_risk_section(body_text)
    summary = _extract_summary(body_raw) or _extract_summary(body_text) or list_summary
    rating = _safe_string(row.get("emRatingName")) or _safe_string(row.get("sRatingName"))
    target_price = _extract_target_price(row)
    industry = _safe_string(row.get("indvInduName")) or _safe_string(row.get("industryName"))
    text_hash = _build_hash(title, summary, body_raw, risk_section_raw)
    version_hash = _build_hash(text_hash, rating, target_price)
    report_id = _build_report_id(stock_id, broker_id, publish_time, title, source_url)

    report = ReportRecord(
        report_id=report_id,
        stock_id=stock_id,
        stock_name=stock_name,
        broker_id=broker_id,
        broker_name=broker_name,
        analyst_id=analyst_id,
        analyst_name=analyst_name,
        publish_time=publish_time,
        title=title,
        summary=summary,
        body_raw=body_raw,
        risk_section_raw=risk_section_raw,
        rating=rating,
        target_price=target_price,
        industry=industry,
        source_url=source_url,
        text_hash=text_hash,
        version_hash=version_hash,
        is_deleted="0",
        created_at=fetched_at,
        updated_at=fetched_at,
    )

    rating_record = ReportRatingRecord(
        report_id=report_id,
        stock_id=stock_id,
        broker_id=broker_id,
        publish_time=publish_time,
        rating=rating,
        last_rating=_safe_string(row.get("lastEmRatingName")),
        rating_change=_safe_string(row.get("ratingChange")),
        target_price_upper=_safe_string(row.get("indvAimPriceT")),
        target_price_lower=_safe_string(row.get("indvAimPriceL")),
        source_url=source_url,
        created_at=fetched_at,
        updated_at=fetched_at,
    )

    broker_record = ReportBrokerRecord(
        broker_id=broker_id,
        broker_name=broker_name,
        created_at=fetched_at,
        updated_at=fetched_at,
    )
    broker_alias_record = ReportBrokerAliasRecord(
        broker_id=broker_id,
        broker_name=broker_name,
        first_seen_at=publish_time,
        last_seen_at=publish_time,
        created_at=fetched_at,
        updated_at=fetched_at,
    )

    analyst_records: list[ReportAnalystRecord] = []
    analyst_bridge_records: list[ReportAnalystBridgeRecord] = []
    analyst_ids = [item for item in analyst_id.split(",") if item]
    analyst_names = [item for item in analyst_name.split(",") if item]
    max_len = max(len(analyst_ids), len(analyst_names))
    for index in range(max_len):
        current_analyst_id = analyst_ids[index] if index < len(analyst_ids) else ""
        current_analyst_name = analyst_names[index] if index < len(analyst_names) else ""
        record = ReportAnalystRecord(
            analyst_id=current_analyst_id,
            analyst_name=current_analyst_name,
            broker_id=broker_id,
            broker_name=broker_name,
            created_at=fetched_at,
            updated_at=fetched_at,
        )
        bridge_record = ReportAnalystBridgeRecord(
            report_id=report_id,
            analyst_order=index + 1,
            analyst_id=current_analyst_id,
            analyst_name=current_analyst_name,
            broker_id=broker_id,
            broker_name=broker_name,
            created_at=fetched_at,
            updated_at=fetched_at,
        )
        if record.analyst_id or record.analyst_name:
            analyst_records.append(record)
            analyst_bridge_records.append(bridge_record)

    version_record = ReportVersionRecord(
        report_id=report_id,
        version_hash=version_hash,
        info_code=info_code,
        source_url=source_url,
        fetched_at=fetched_at,
        title=title,
        summary=summary,
        body_raw=body_raw,
        risk_section_raw=risk_section_raw,
        rating=rating,
        target_price=target_price,
    )
    return (
        report,
        rating_record,
        analyst_records,
        analyst_bridge_records,
        broker_record,
        broker_alias_record,
        version_record,
    )


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_legacy_reports(path: Path, reports: list[ReportRecord]) -> None:
    fieldnames = ["report_date", "stock_code", "company_name", "title", "summary"]
    rows = []
    for report in reports:
        report_date = report.publish_time.split(" ")[0] if report.publish_time else ""
        summary = _normalize_inline_text(report.summary.replace("\n", " "))
        rows.append(
            {
                "report_date": report_date,
                "stock_code": report.stock_id,
                "company_name": report.stock_name,
                "title": report.title,
                "summary": summary,
            }
        )
    _write_csv(path, rows, fieldnames)


def _iter_month_windows(start_year: int, end_year: int) -> list[tuple[str, str]]:
    windows: list[tuple[str, str]] = []
    for year in range(start_year, end_year + 1):
        for month in range(1, 13):
            end_day = monthrange(year, month)[1]
            windows.append((f"{year}-{month:02d}-01", f"{year}-{month:02d}-{end_day:02d}"))
    return windows


def _state_to_maps(state: dict[str, Any]) -> tuple[
    dict[str, ReportRecord],
    dict[tuple[str, str], ReportRatingRecord],
    dict[str, ReportBrokerRecord],
    dict[tuple[str, str], ReportBrokerAliasRecord],
    dict[tuple[str, str, str], ReportAnalystRecord],
    dict[tuple[str, int, str, str], ReportAnalystBridgeRecord],
    dict[tuple[str, str], ReportVersionRecord],
]:
    report_map = {
        row["report_id"]: ReportRecord(**row)
        for row in state.get("reports", [])
    }
    rating_map = {
        (row["report_id"], row["publish_time"]): ReportRatingRecord(**row)
        for row in state.get("ratings", [])
    }
    broker_map = {
        row["broker_id"] or row["broker_name"]: ReportBrokerRecord(**row)
        for row in state.get("brokers", [])
    }
    broker_alias_map = {
        (row["broker_id"], row["broker_name"]): ReportBrokerAliasRecord(**row)
        for row in state.get("broker_aliases", [])
    }
    analyst_map = {
        (row["analyst_id"], row["analyst_name"], row["broker_id"]): ReportAnalystRecord(**row)
        for row in state.get("analysts", [])
    }
    analyst_bridge_map = {
        (row["report_id"], int(row["analyst_order"]), row["analyst_id"], row["analyst_name"]): ReportAnalystBridgeRecord(**row)
        for row in state.get("analyst_bridges", [])
    }
    version_map = {
        (row["info_code"], row["version_hash"]): ReportVersionRecord(**row)
        for row in state.get("versions", [])
    }
    return report_map, rating_map, broker_map, broker_alias_map, analyst_map, analyst_bridge_map, version_map


def _dump_state(
    state_path: Path,
    *,
    start_year: int,
    end_year: int,
    page_size: int,
    completed_windows: list[str],
    processed_windows: int,
    processed_pages: int,
    total_hits: int,
    fetched_details: int,
    current_window: str | None,
    report_map: dict[str, ReportRecord],
    rating_map: dict[tuple[str, str], ReportRatingRecord],
    broker_map: dict[str, ReportBrokerRecord],
    broker_alias_map: dict[tuple[str, str], ReportBrokerAliasRecord],
    analyst_map: dict[tuple[str, str, str], ReportAnalystRecord],
    analyst_bridge_map: dict[tuple[str, int, str, str], ReportAnalystBridgeRecord],
    version_map: dict[tuple[str, str], ReportVersionRecord],
    quality_stats: Counter[str],
) -> None:
    state = {
        "start_year": start_year,
        "end_year": end_year,
        "page_size": page_size,
        "processed_windows": processed_windows,
        "processed_pages": processed_pages,
        "total_hits": total_hits,
        "fetched_details": fetched_details,
        "current_window": current_window,
        "completed_windows": completed_windows,
        "reports": [record.to_dict() for record in report_map.values()],
        "ratings": [record.to_dict() for record in rating_map.values()],
        "brokers": [record.to_dict() for record in broker_map.values()],
        "broker_aliases": [record.to_dict() for record in broker_alias_map.values()],
        "analysts": [record.to_dict() for record in analyst_map.values()],
        "analyst_bridges": [record.to_dict() for record in analyst_bridge_map.values()],
        "versions": [record.to_dict() for record in version_map.values()],
        "quality_stats": dict(quality_stats),
        "updated_at": _iso_now(),
    }
    state_path.parent.mkdir(parents=True, exist_ok=True)
    with state_path.open("w", encoding="utf-8") as handle:
        json.dump(state, handle, ensure_ascii=False)


def _load_state(state_path: Path) -> dict[str, Any] | None:
    if not state_path.exists():
        return None
    with state_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _merge_maps(
    report_map: dict[str, ReportRecord],
    rating_map: dict[tuple[str, str], ReportRatingRecord],
    broker_map: dict[str, ReportBrokerRecord],
    broker_alias_map: dict[tuple[str, str], ReportBrokerAliasRecord],
    analyst_map: dict[tuple[str, str, str], ReportAnalystRecord],
    analyst_bridge_map: dict[tuple[str, int, str, str], ReportAnalystBridgeRecord],
    version_map: dict[tuple[str, str], ReportVersionRecord],
    rows: list[dict[str, Any]],
    *,
    session: requests.Session,
    quality_stats: Counter[str],
) -> int:
    fetched_details = 0
    for row in rows:
        info_code = _safe_string(row.get("infoCode"))
        if not info_code:
            continue
        fetched_at = _iso_now()
        list_summary = _extract_list_summary(row)
        try:
            html = _fetch_detail_html(session, info_code)
        except requests.RequestException:
            quality_stats["detail_fetch_failed"] += 1
            html = ""
        body_text = ""
        extraction_source = "missing"
        if html:
            body_text, extraction_source = _extract_body_from_html(html, session)
        if extraction_source == "pdf":
            quality_stats["body_extracted_from_pdf"] += 1
        elif extraction_source == "html_fallback":
            quality_stats["body_extracted_from_html_fallback"] += 1
        elif extraction_source == "html":
            quality_stats["body_extracted_from_html"] += 1
        elif extraction_source == "meta":
            quality_stats["body_extracted_from_meta"] += 1
        else:
            quality_stats["body_missing_after_parse"] += 1
        (
            report,
            rating_record,
            analyst_records,
            analyst_bridge_records,
            broker_record,
            broker_alias_record,
            version_record,
        ) = _build_record(row, body_text, list_summary, fetched_at, info_code)
        report_map[report.report_id] = report
        rating_map[(rating_record.report_id, rating_record.publish_time)] = rating_record
        broker_key = broker_record.broker_id or broker_record.broker_name
        broker_map[broker_key] = _canonical_broker_record(broker_map.get(broker_key), broker_record, report.publish_time)
        alias_key = (broker_alias_record.broker_id, broker_alias_record.broker_name)
        broker_alias_map[alias_key] = _merge_broker_alias(broker_alias_map.get(alias_key), broker_alias_record)
        for analyst_record in analyst_records:
            key = (analyst_record.analyst_id, analyst_record.analyst_name, analyst_record.broker_id)
            analyst_map[key] = analyst_record
        for bridge_record in analyst_bridge_records:
            key = (bridge_record.report_id, bridge_record.analyst_order, bridge_record.analyst_id, bridge_record.analyst_name)
            analyst_bridge_map[key] = bridge_record
        version_map[(version_record.info_code, version_record.version_hash)] = version_record
        if report.summary:
            quality_stats["summary_present"] += 1
        else:
            quality_stats["summary_missing"] += 1
        if report.body_raw:
            quality_stats["body_present"] += 1
        else:
            quality_stats["body_missing"] += 1
        if report.risk_section_raw:
            quality_stats["risk_present"] += 1
        fetched_details += 1
    return fetched_details


def _finalize_and_write(
    report_map: dict[str, ReportRecord],
    rating_map: dict[tuple[str, str], ReportRatingRecord],
    broker_map: dict[str, ReportBrokerRecord],
    broker_alias_map: dict[tuple[str, str], ReportBrokerAliasRecord],
    analyst_map: dict[tuple[str, str, str], ReportAnalystRecord],
    analyst_bridge_map: dict[tuple[str, int, str, str], ReportAnalystBridgeRecord],
    version_map: dict[tuple[str, str], ReportVersionRecord],
    *,
    reports_master_path: Path,
    report_ratings_path: Path,
    report_brokers_path: Path,
    report_broker_aliases_path: Path,
    report_analysts_path: Path,
    report_analyst_bridge_path: Path,
    report_versions_path: Path,
    legacy_reports_path: Path,
    quality_stats: Counter[str],
) -> dict[str, Any]:
    reports = sorted(report_map.values(), key=lambda item: (item.publish_time, item.stock_id, item.report_id))
    ratings = sorted(rating_map.values(), key=lambda item: (item.publish_time, item.report_id))
    brokers = sorted(broker_map.values(), key=lambda item: (item.broker_name, item.broker_id))
    broker_aliases = sorted(broker_alias_map.values(), key=lambda item: (item.broker_id, item.broker_name))
    analysts = sorted(analyst_map.values(), key=lambda item: (item.analyst_name, item.analyst_id, item.broker_id))
    analyst_bridges = sorted(analyst_bridge_map.values(), key=lambda item: (item.report_id, item.analyst_order, item.analyst_id))
    versions = sorted(version_map.values(), key=lambda item: (item.info_code, item.fetched_at, item.version_hash))

    _write_csv(reports_master_path, [record.to_dict() for record in reports], list(ReportRecord.__dataclass_fields__.keys()))
    _write_csv(report_ratings_path, [record.to_dict() for record in ratings], list(ReportRatingRecord.__dataclass_fields__.keys()))
    _write_csv(report_brokers_path, [record.to_dict() for record in brokers], list(ReportBrokerRecord.__dataclass_fields__.keys()))
    _write_csv(report_broker_aliases_path, [record.to_dict() for record in broker_aliases], list(ReportBrokerAliasRecord.__dataclass_fields__.keys()))
    _write_csv(report_analysts_path, [record.to_dict() for record in analysts], list(ReportAnalystRecord.__dataclass_fields__.keys()))
    _write_csv(report_analyst_bridge_path, [record.to_dict() for record in analyst_bridges], list(ReportAnalystBridgeRecord.__dataclass_fields__.keys()))
    _write_csv(report_versions_path, [record.to_dict() for record in versions], list(ReportVersionRecord.__dataclass_fields__.keys()))
    _write_legacy_reports(legacy_reports_path, reports)
    return {
        "reports_count": len(reports),
        "ratings_count": len(ratings),
        "brokers_count": len(brokers),
        "broker_aliases_count": len(broker_aliases),
        "analysts_count": len(analysts),
        "analyst_bridge_count": len(analyst_bridges),
        "versions_count": len(versions),
        "quality_stats": dict(sorted(quality_stats.items())),
    }


def rebuild_reports_dataset(
    *,
    begin_time: str,
    end_time: str,
    max_pages: int = 1,
    page_size: int = 50,
    reports_master_path: Path = REPORTS_MASTER_PATH,
    report_ratings_path: Path = REPORT_RATINGS_PATH,
    report_brokers_path: Path = REPORT_BROKERS_PATH,
    report_broker_aliases_path: Path = REPORT_BROKER_ALIASES_PATH,
    report_analysts_path: Path = REPORT_ANALYSTS_PATH,
    report_analyst_bridge_path: Path = REPORT_ANALYST_BRIDGE_PATH,
    report_versions_path: Path = REPORT_VERSIONS_PATH,
    legacy_reports_path: Path = RAW_REPORTS_PATH,
) -> dict[str, Any]:
    session = requests.Session()
    session.headers.update(DEFAULT_HEADERS)

    report_map: dict[str, ReportRecord] = {}
    rating_map: dict[tuple[str, str], ReportRatingRecord] = {}
    broker_map: dict[str, ReportBrokerRecord] = {}
    broker_alias_map: dict[tuple[str, str], ReportBrokerAliasRecord] = {}
    analyst_map: dict[tuple[str, str, str], ReportAnalystRecord] = {}
    analyst_bridge_map: dict[tuple[str, int, str, str], ReportAnalystBridgeRecord] = {}
    version_map: dict[tuple[str, str], ReportVersionRecord] = {}
    quality_stats: Counter[str] = Counter()
    processed_pages = 0
    total_hits = 0
    fetched_details = 0
    effective_page_size = min(page_size, MAX_PAGE_SIZE)

    for page_no in range(1, max_pages + 1):
        payload = _fetch_list_page(session, page_no=page_no, begin_time=begin_time, end_time=end_time, page_size=effective_page_size)
        rows = payload.get("data") or []
        if page_no == 1:
            total_hits = int(payload.get("hits") or 0)
        if not rows:
            break
        processed_pages += 1
        fetched_details += _merge_maps(
            report_map,
            rating_map,
            broker_map,
            broker_alias_map,
            analyst_map,
            analyst_bridge_map,
            version_map,
            rows,
            session=session,
            quality_stats=quality_stats,
        )

    counts = _finalize_and_write(
        report_map,
        rating_map,
        broker_map,
        broker_alias_map,
        analyst_map,
        analyst_bridge_map,
        version_map,
        reports_master_path=reports_master_path,
        report_ratings_path=report_ratings_path,
        report_brokers_path=report_brokers_path,
        report_broker_aliases_path=report_broker_aliases_path,
        report_analysts_path=report_analysts_path,
        report_analyst_bridge_path=report_analyst_bridge_path,
        report_versions_path=report_versions_path,
        legacy_reports_path=legacy_reports_path,
        quality_stats=quality_stats,
    )

    return {
        "status": "ok",
        "begin_time": begin_time,
        "end_time": end_time,
        "processed_pages": processed_pages,
        "page_size": effective_page_size,
        "total_hits": total_hits,
        "fetched_details": fetched_details,
        "reports_master_path": str(reports_master_path),
        "legacy_reports_path": str(legacy_reports_path),
        **counts,
    }


def rebuild_reports_dataset_full_range(
    *,
    start_year: int = 2016,
    end_year: int = 2026,
    page_size: int = 100,
    state_path: Path = REPORT_REBUILD_STATE_PATH,
    resume: bool = True,
    reports_master_path: Path = REPORTS_MASTER_PATH,
    report_ratings_path: Path = REPORT_RATINGS_PATH,
    report_brokers_path: Path = REPORT_BROKERS_PATH,
    report_broker_aliases_path: Path = REPORT_BROKER_ALIASES_PATH,
    report_analysts_path: Path = REPORT_ANALYSTS_PATH,
    report_analyst_bridge_path: Path = REPORT_ANALYST_BRIDGE_PATH,
    report_versions_path: Path = REPORT_VERSIONS_PATH,
    legacy_reports_path: Path = RAW_REPORTS_PATH,
) -> dict[str, Any]:
    session = requests.Session()
    session.headers.update(DEFAULT_HEADERS)
    effective_page_size = min(page_size, MAX_PAGE_SIZE)
    windows = _iter_month_windows(start_year, end_year)
    total_window_count = len(windows)

    report_map: dict[str, ReportRecord] = {}
    rating_map: dict[tuple[str, str], ReportRatingRecord] = {}
    broker_map: dict[str, ReportBrokerRecord] = {}
    broker_alias_map: dict[tuple[str, str], ReportBrokerAliasRecord] = {}
    analyst_map: dict[tuple[str, str, str], ReportAnalystRecord] = {}
    analyst_bridge_map: dict[tuple[str, int, str, str], ReportAnalystBridgeRecord] = {}
    version_map: dict[tuple[str, str], ReportVersionRecord] = {}
    quality_stats: Counter[str] = Counter()
    completed_windows: list[str] = []
    processed_windows = 0
    processed_pages = 0
    total_hits = 0
    fetched_details = 0
    resumed_from_state = False
    window_stats: list[dict[str, Any]] = []

    if resume:
        state = _load_state(state_path)
        if state:
            (
                report_map,
                rating_map,
                broker_map,
                broker_alias_map,
                analyst_map,
                analyst_bridge_map,
                version_map,
            ) = _state_to_maps(state)
            completed_windows = list(state.get("completed_windows", []))
            processed_windows = int(state.get("processed_windows", 0))
            processed_pages = int(state.get("processed_pages", 0))
            total_hits = int(state.get("total_hits", 0))
            fetched_details = int(state.get("fetched_details", 0))
            quality_stats.update(state.get("quality_stats", {}))
            resumed_from_state = True
            print(
                f"[RESUME] windows={processed_windows}/{total_window_count} pages={processed_pages} "
                f"reports={len(report_map)} details={fetched_details}",
                flush=True,
            )

    completed_window_set = set(completed_windows)

    for index, (begin_time, end_time) in enumerate(windows, start=1):
        window_key = f"{begin_time}:{end_time}"
        if window_key in completed_window_set:
            print(f"[SKIP] {index}/{total_window_count} {begin_time}..{end_time} already completed", flush=True)
            continue

        print(f"[WINDOW] {index}/{total_window_count} {begin_time}..{end_time} start", flush=True)
        first_page = _fetch_list_page(session, page_no=1, begin_time=begin_time, end_time=end_time, page_size=effective_page_size)
        hits = int(first_page.get("hits") or 0)
        if hits <= 0:
            completed_windows.append(window_key)
            completed_window_set.add(window_key)
            processed_windows += 1
            _dump_state(
                state_path,
                start_year=start_year,
                end_year=end_year,
                page_size=effective_page_size,
                completed_windows=completed_windows,
                processed_windows=processed_windows,
                processed_pages=processed_pages,
                total_hits=total_hits,
                fetched_details=fetched_details,
                current_window=None,
                report_map=report_map,
                rating_map=rating_map,
                broker_map=broker_map,
                broker_alias_map=broker_alias_map,
                analyst_map=analyst_map,
                analyst_bridge_map=analyst_bridge_map,
                version_map=version_map,
                quality_stats=quality_stats,
            )
            print(f"[WINDOW] {begin_time}..{end_time} empty", flush=True)
            continue

        total_hits += hits
        total_pages = (hits + effective_page_size - 1) // effective_page_size
        window_stats.append({
            "begin_time": begin_time,
            "end_time": end_time,
            "hits": hits,
            "pages": total_pages,
        })
        print(f"[WINDOW] {begin_time}..{end_time} hits={hits} pages={total_pages}", flush=True)

        rows = first_page.get("data") or []
        if rows:
            fetched_details += _merge_maps(
                report_map,
                rating_map,
                broker_map,
                broker_alias_map,
                analyst_map,
                analyst_bridge_map,
                version_map,
                rows,
                session=session,
                quality_stats=quality_stats,
            )
            processed_pages += 1
            print(
                f"[PAGE] {begin_time}..{end_time} 1/{total_pages} total_reports={len(report_map)} details={fetched_details}",
                flush=True,
            )

        for page_no in range(2, total_pages + 1):
            payload = _fetch_list_page(session, page_no=page_no, begin_time=begin_time, end_time=end_time, page_size=effective_page_size)
            rows = payload.get("data") or []
            if not rows:
                break
            fetched_details += _merge_maps(
                report_map,
                rating_map,
                broker_map,
                broker_alias_map,
                analyst_map,
                analyst_bridge_map,
                version_map,
                rows,
                session=session,
                quality_stats=quality_stats,
            )
            processed_pages += 1
            print(
                f"[PAGE] {begin_time}..{end_time} {page_no}/{total_pages} total_reports={len(report_map)} details={fetched_details}",
                flush=True,
            )

        processed_windows += 1
        completed_windows.append(window_key)
        completed_window_set.add(window_key)

        _finalize_and_write(
            report_map,
            rating_map,
            broker_map,
            broker_alias_map,
            analyst_map,
            analyst_bridge_map,
            version_map,
            reports_master_path=reports_master_path,
            report_ratings_path=report_ratings_path,
            report_brokers_path=report_brokers_path,
            report_broker_aliases_path=report_broker_aliases_path,
            report_analysts_path=report_analysts_path,
            report_analyst_bridge_path=report_analyst_bridge_path,
            report_versions_path=report_versions_path,
            legacy_reports_path=legacy_reports_path,
            quality_stats=quality_stats,
        )
        _dump_state(
            state_path,
            start_year=start_year,
            end_year=end_year,
            page_size=effective_page_size,
            completed_windows=completed_windows,
            processed_windows=processed_windows,
            processed_pages=processed_pages,
            total_hits=total_hits,
            fetched_details=fetched_details,
            current_window=None,
            report_map=report_map,
            rating_map=rating_map,
            broker_map=broker_map,
            broker_alias_map=broker_alias_map,
            analyst_map=analyst_map,
            analyst_bridge_map=analyst_bridge_map,
            version_map=version_map,
            quality_stats=quality_stats,
        )
        print(
            f"[CHECKPOINT] windows={processed_windows}/{total_window_count} pages={processed_pages} reports={len(report_map)}",
            flush=True,
        )

    counts = _finalize_and_write(
        report_map,
        rating_map,
        broker_map,
        broker_alias_map,
        analyst_map,
        analyst_bridge_map,
        version_map,
        reports_master_path=reports_master_path,
        report_ratings_path=report_ratings_path,
        report_brokers_path=report_brokers_path,
        report_broker_aliases_path=report_broker_aliases_path,
        report_analysts_path=report_analysts_path,
        report_analyst_bridge_path=report_analyst_bridge_path,
        report_versions_path=report_versions_path,
        legacy_reports_path=legacy_reports_path,
        quality_stats=quality_stats,
    )
    _dump_state(
        state_path,
        start_year=start_year,
        end_year=end_year,
        page_size=effective_page_size,
        completed_windows=completed_windows,
        processed_windows=processed_windows,
        processed_pages=processed_pages,
        total_hits=total_hits,
        fetched_details=fetched_details,
        current_window=None,
        report_map=report_map,
        rating_map=rating_map,
        broker_map=broker_map,
        broker_alias_map=broker_alias_map,
        analyst_map=analyst_map,
        analyst_bridge_map=analyst_bridge_map,
        version_map=version_map,
        quality_stats=quality_stats,
    )

    return {
        "status": "ok",
        "start_year": start_year,
        "end_year": end_year,
        "processed_windows": processed_windows,
        "total_windows": total_window_count,
        "processed_pages": processed_pages,
        "page_size": effective_page_size,
        "total_hits": total_hits,
        "fetched_details": fetched_details,
        "resumed_from_state": resumed_from_state,
        "state_path": str(state_path),
        "reports_master_path": str(reports_master_path),
        "legacy_reports_path": str(legacy_reports_path),
        "window_stats": window_stats,
        **counts,
    }
