from __future__ import annotations


def apply_turnover_cost(gross_return: float, turnover: float, cost_bps: float) -> float:
    return round(gross_return - (turnover * cost_bps) / 10000.0, 6)
