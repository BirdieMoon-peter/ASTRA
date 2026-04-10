from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AlignmentDecision:
    raw_index: int
    entry_index: int
    policy_name: str
    entry_field: str


def entry_index_for_policy(raw_index: int, policy_name: str) -> int:
    if policy_name == "same_day_close":
        return raw_index
    if policy_name in {"next_open", "next_close", "t_plus_1_close"}:
        return raw_index + 1
    return raw_index + 1


def entry_field_for_policy(policy_name: str) -> str:
    if policy_name == "next_open":
        return "open"
    return "close"


def decide_alignment(raw_index: int, policy_name: str) -> AlignmentDecision:
    return AlignmentDecision(
        raw_index=raw_index,
        entry_index=entry_index_for_policy(raw_index, policy_name),
        policy_name=policy_name,
        entry_field=entry_field_for_policy(policy_name),
    )
