from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SignalDefinition:
    name: str
    description: str
    direction: str
    default_for_backtest: bool = False
    default_for_paper: bool = False


SIGNAL_REGISTRY_VERSION = 1

SIGNAL_DEFINITIONS: tuple[SignalDefinition, ...] = (
    SignalDefinition("body_only_signal", "Fundamental sentiment only", "higher_is_more_bullish"),
    SignalDefinition("title_body_divergence", "Title-body lexical divergence", "higher_is_more_bullish"),
    SignalDefinition("counterfactual_gap", "Negative strategic optimism gap", "higher_is_more_bullish"),
    SignalDefinition("astra_composite", "Composite ASTRA score", "higher_is_more_bullish"),
    SignalDefinition("astra_uncertainty_gated", "Composite score gated by uncertainty", "higher_is_more_bullish"),
    SignalDefinition("astra_finance_blend", "Composite-body blend score", "higher_is_more_bullish"),
    SignalDefinition("astra_finance_plus_gap", "Finance blend plus gap term", "higher_is_more_bullish", default_for_paper=True),
    SignalDefinition("astra_gap_scaled", "Scaled gap-only score", "higher_is_more_bullish"),
    SignalDefinition("astra_quality_weighted", "Quality-weighted ASTRA score", "higher_is_more_bullish", default_for_backtest=True),
)


def signal_names() -> tuple[str, ...]:
    return tuple(item.name for item in SIGNAL_DEFINITIONS)


def primary_backtest_signal() -> str:
    for item in SIGNAL_DEFINITIONS:
        if item.default_for_backtest:
            return item.name
    return "astra_quality_weighted"


def paper_signal_rows() -> tuple[str, ...]:
    return tuple(item.name for item in SIGNAL_DEFINITIONS if item.default_for_paper or item.name in {
        "body_only_signal",
        "title_body_divergence",
        "counterfactual_gap",
        "astra_composite",
        "astra_uncertainty_gated",
    })


def registry_metadata() -> dict[str, object]:
    return {
        "version": SIGNAL_REGISTRY_VERSION,
        "signals": [
            {
                "name": item.name,
                "description": item.description,
                "direction": item.direction,
                "default_for_backtest": item.default_for_backtest,
                "default_for_paper": item.default_for_paper,
            }
            for item in SIGNAL_DEFINITIONS
        ],
        "primary_backtest_signal": primary_backtest_signal(),
        "paper_signal_rows": list(paper_signal_rows()),
    }
