from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class ReportRecord:
    report_id: str
    stock_id: str
    stock_name: str
    broker_id: str
    broker_name: str
    analyst_id: str
    analyst_name: str
    publish_time: str
    title: str
    summary: str
    body_raw: str
    risk_section_raw: str
    rating: str
    target_price: str
    industry: str
    source_url: str
    text_hash: str
    version_hash: str
    is_deleted: str
    created_at: str
    updated_at: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ReportRatingRecord:
    report_id: str
    stock_id: str
    broker_id: str
    publish_time: str
    rating: str
    last_rating: str
    rating_change: str
    target_price_upper: str
    target_price_lower: str
    source_url: str
    created_at: str
    updated_at: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ReportBrokerRecord:
    broker_id: str
    broker_name: str
    created_at: str
    updated_at: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ReportBrokerAliasRecord:
    broker_id: str
    broker_name: str
    first_seen_at: str
    last_seen_at: str
    created_at: str
    updated_at: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ReportAnalystRecord:
    analyst_id: str
    analyst_name: str
    broker_id: str
    broker_name: str
    created_at: str
    updated_at: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ReportAnalystBridgeRecord:
    report_id: str
    analyst_order: int
    analyst_id: str
    analyst_name: str
    broker_id: str
    broker_name: str
    created_at: str
    updated_at: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ReportVersionRecord:
    report_id: str
    version_hash: str
    info_code: str
    source_url: str
    fetched_at: str
    title: str
    summary: str
    body_raw: str
    risk_section_raw: str
    rating: str
    target_price: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
