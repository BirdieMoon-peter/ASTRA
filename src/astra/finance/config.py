from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class ExecutionPolicyConfig:
    name: str = "next_close"
    allow_same_day_close: bool = False
    fallback_when_no_publish_time: str = "next_close"
    after_close_policy: str = "next_trading_day"


@dataclass(frozen=True)
class UniverseFilterConfig:
    name: str = "a_share_tradable_eval_universe"
    exclude_st: bool = True
    exclude_pt: bool = True
    exclude_suspended: bool = True
    exclude_limit_up_down: bool = True
    min_prior_trading_days: int = 120
    require_forward_returns: bool = True
    missing_filter_fields: str = "allow"


@dataclass(frozen=True)
class ReportAggregationConfig:
    level: str = "stock_day"
    normalize_weights_within_stock_day: bool = True
    use_recency_weight: bool = True
    use_broker_coverage_weight: bool = True
    use_analyst_reliability_weight: bool = True
    default_report_weight: float = 1.0
    recency_half_life_days: int = 30


@dataclass(frozen=True)
class NeutralizationConfig:
    winsor_lower: float = 0.01
    winsor_upper: float = 0.99
    standardize_within_date: bool = True
    residualize_industry: bool = True
    industry_field: str = "industry"
    residualize_style: bool = True
    style_controls: tuple[str, ...] = (
        "log_market_cap",
        "book_to_market",
        "reversal_1m",
        "momentum_12_1",
        "volatility_20d",
    )
    residualization_ridge: float = 1e-6


@dataclass(frozen=True)
class PortfolioConstructionConfig:
    rebalance_frequency: str = "monthly"
    dollar_neutral: bool = True
    industry_neutral_after_residualization: bool = True
    max_name_weight: float = 0.02
    portfolio_quantiles: int = 5
    holding_horizon_days: int = 20


@dataclass(frozen=True)
class RollingProtocolConfig:
    training_window: str = "expanding"
    dev_selection_only: bool = True
    retune_on_test: bool = False
    beta_weights_fixed_out_of_sample: bool = True
    lambda_weights_fixed_out_of_sample: bool = True
    beta_weights_method: str = "inverse_variance_on_dev"
    lambda_h: float = 0.5
    lambda_o: float = 1.0


@dataclass(frozen=True)
class CostModelConfig:
    commission_bps: float = 0.0
    sell_tax_bps: float = 0.0
    slippage_bps: float = 0.0
    short_borrow_bps_annual: float = 0.0
    headline_return_basis: str = "gross"


@dataclass(frozen=True)
class CostStressConfig:
    one_way_fee_bps: tuple[float, ...] = (5.0, 10.0, 20.0)
    slippage_bps: tuple[float, ...] = (5.0, 10.0)
    impact_model: str = "linear_participation_rate"
    participation_rate: float = 0.05
    impact_bps_per_1pct_participation: float = 1.0


@dataclass(frozen=True)
class MetricsConfig:
    rank_ic_horizons: tuple[int, ...] = (5, 10, 20)
    rank_ic_tstat: str = "newey_west"
    sharpe_horizon: int = 20
    turnover_frequency: str = "monthly_one_way"
    include_max_drawdown: bool = True
    include_terminal_cumulative_return: bool = True


@dataclass(frozen=True)
class BacktestConfig:
    primary_signal: str = "astra_quality_weighted"
    forward_horizons: tuple[int, ...] = (5, 10, 20)
    normalization_method: str = "winsor_zscore_residual_rank"
    min_trade_dates: int = 20
    execution_policy: ExecutionPolicyConfig = ExecutionPolicyConfig()
    universe: UniverseFilterConfig = UniverseFilterConfig()
    report_aggregation: ReportAggregationConfig = ReportAggregationConfig()
    neutralization: NeutralizationConfig = NeutralizationConfig()
    portfolio: PortfolioConstructionConfig = PortfolioConstructionConfig()
    rolling_protocol: RollingProtocolConfig = RollingProtocolConfig()
    cost_model: CostModelConfig = CostModelConfig()
    cost_stress: CostStressConfig = CostStressConfig()
    metrics: MetricsConfig = MetricsConfig()
    signal_registry_version: int = 1

    @property
    def portfolio_quantiles(self) -> int:
        return self.portfolio.portfolio_quantiles

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["forward_horizons"] = list(self.forward_horizons)
        payload["neutralization"]["style_controls"] = list(self.neutralization.style_controls)
        payload["cost_stress"]["one_way_fee_bps"] = list(self.cost_stress.one_way_fee_bps)
        payload["cost_stress"]["slippage_bps"] = list(self.cost_stress.slippage_bps)
        payload["metrics"]["rank_ic_horizons"] = list(self.metrics.rank_ic_horizons)
        return payload


def default_backtest_config() -> BacktestConfig:
    return BacktestConfig()
