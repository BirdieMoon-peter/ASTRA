import unittest
from unittest.mock import patch

from astra.finance.build_signals import build_report_signal


class BuildReportSignalTest(unittest.TestCase):
    @patch("astra.finance.build_signals.simple_title_body_divergence", return_value=0.123456789)
    def test_build_report_signal_derives_all_deterministic_scores_locally(self, divergence_mock) -> None:
        prediction = {
            "fundamental_sentiment": "positive",
            "strategic_optimism_gap": 0.4,
            "uncertainty": 0.25,
            "astra_composite_score": 1.9,
            "astra_uncertainty_gated_score": 99.0,
            "astra_finance_blend_score": -99.0,
        }

        signals = build_report_signal(prediction, title="steady outlook", summary="steady execution")

        divergence_mock.assert_called_once_with("steady outlook", "steady execution")
        self.assertEqual(signals["body_only_signal"], 1.0)
        self.assertEqual(signals["title_body_divergence"], 0.123457)
        self.assertEqual(signals["counterfactual_gap"], -0.4)
        self.assertEqual(signals["astra_composite"], 1.9)
        self.assertEqual(signals["astra_uncertainty_gated"], 1.425)
        self.assertEqual(signals["astra_finance_blend"], 1.675)
        self.assertEqual(signals["astra_finance_plus_gap"], 1.615)
        self.assertEqual(signals["astra_gap_scaled"], -0.3)
        self.assertEqual(signals["astra_quality_weighted"], 1.34)

    @patch("astra.finance.build_signals.simple_title_body_divergence", return_value=0.0)
    def test_build_report_signal_uses_fallback_composite_when_not_prefilled(self, _divergence_mock) -> None:
        prediction = {
            "fundamental_sentiment": "negative",
            "strategic_optimism_gap": 0.2,
            "uncertainty": 0.4,
        }

        signals = build_report_signal(prediction, title="soft setup", summary="supportive body")

        self.assertEqual(signals["astra_composite"], -1.2)
        self.assertEqual(signals["astra_uncertainty_gated"], -0.72)
        self.assertEqual(signals["astra_finance_blend"], -1.15)
        self.assertEqual(signals["astra_finance_plus_gap"], -1.18)
        self.assertEqual(signals["astra_gap_scaled"], -0.15)
        self.assertEqual(signals["astra_quality_weighted"], -0.782)

    @patch("astra.finance.build_signals.simple_title_body_divergence", return_value=0.0)
    def test_build_report_signal_clamps_uncertainty_above_one(self, _divergence_mock) -> None:
        prediction = {
            "fundamental_sentiment": "neutral",
            "strategic_optimism_gap": 0.2,
            "uncertainty": 1.5,
        }

        signals = build_report_signal(prediction, title="balanced view", summary="balanced body")

        self.assertEqual(signals["astra_uncertainty_gated"], -0.0)
        self.assertEqual(signals["astra_quality_weighted"], -0.03)

    @patch("astra.finance.build_signals.simple_title_body_divergence", return_value=0.0)
    def test_build_report_signal_clamps_uncertainty_below_zero(self, _divergence_mock) -> None:
        prediction = {
            "fundamental_sentiment": "neutral",
            "strategic_optimism_gap": 0.2,
            "uncertainty": -0.2,
        }

        signals = build_report_signal(prediction, title="balanced view", summary="balanced body")

        self.assertEqual(signals["astra_uncertainty_gated"], -0.2)
        self.assertEqual(signals["astra_quality_weighted"], -0.15)

    @patch("astra.finance.build_signals.simple_title_body_divergence", return_value=0.0)
    def test_build_report_signal_defaults_uncertainty_when_missing(self, _divergence_mock) -> None:
        prediction = {
            "fundamental_sentiment": "neutral",
            "strategic_optimism_gap": 0.2,
        }

        signals = build_report_signal(prediction, title="balanced view", summary="balanced body")

        self.assertEqual(signals["astra_uncertainty_gated"], -0.1)
        self.assertEqual(signals["astra_quality_weighted"], -0.09)

    @patch("astra.finance.build_signals.simple_title_body_divergence", return_value=0.0)
    def test_build_report_signal_defaults_uncertainty_when_not_numeric(self, _divergence_mock) -> None:
        prediction = {
            "fundamental_sentiment": "neutral",
            "strategic_optimism_gap": 0.2,
            "uncertainty": "unknown",
        }

        signals = build_report_signal(prediction, title="balanced view", summary="balanced body")

        self.assertEqual(signals["astra_uncertainty_gated"], -0.1)
        self.assertEqual(signals["astra_quality_weighted"], -0.09)

    @patch("astra.finance.build_signals.simple_title_body_divergence", return_value=0.0)
    def test_build_report_signal_rounds_all_outputs_to_six_decimals(self, _divergence_mock) -> None:
        prediction = {
            "fundamental_sentiment": "positive",
            "strategic_optimism_gap": 0.123456789,
            "uncertainty": 0.333333333,
            "astra_composite_score": 0.987654321,
        }

        signals = build_report_signal(prediction, title="headline", summary="body")

        self.assertEqual(signals["counterfactual_gap"], -0.123457)
        self.assertEqual(signals["astra_composite"], 0.987654)
        self.assertEqual(signals["astra_uncertainty_gated"], 0.658436)
        self.assertEqual(signals["astra_finance_blend"], 0.990741)
        self.assertEqual(signals["astra_finance_plus_gap"], 0.972222)
        self.assertEqual(signals["astra_gap_scaled"], -0.092593)
        self.assertEqual(signals["astra_quality_weighted"], 0.726543)

    @patch("astra.finance.build_signals.simple_title_body_divergence", return_value=0.0)
    def test_build_report_signal_falls_back_to_zero_for_unknown_sentiment(self, _divergence_mock) -> None:
        prediction = {
            "fundamental_sentiment": "mixed",
            "strategic_optimism_gap": 0.2,
            "uncertainty": 0.1,
        }

        signals = build_report_signal(prediction, title="headline", summary="body")

        self.assertEqual(signals["body_only_signal"], 0.0)
        self.assertEqual(signals["astra_composite"], -0.2)
        self.assertEqual(signals["astra_finance_blend"], -0.15)
        self.assertEqual(signals["astra_finance_plus_gap"], -0.18)
        self.assertEqual(signals["astra_quality_weighted"], -0.138)

    @patch("astra.finance.build_signals.simple_title_body_divergence", return_value=0.0)
    def test_build_report_signal_handles_non_numeric_gap_and_composite_stably(self, _divergence_mock) -> None:
        prediction = {
            "fundamental_sentiment": "positive",
            "strategic_optimism_gap": "not-a-number",
            "astra_composite_score": object(),
            "uncertainty": 0.25,
        }

        signals = build_report_signal(prediction, title="headline", summary="body")

        self.assertEqual(signals["counterfactual_gap"], 0.0)
        self.assertEqual(signals["astra_composite"], 1.0)
        self.assertEqual(signals["astra_uncertainty_gated"], 0.75)
        self.assertEqual(signals["astra_gap_scaled"], 0.0)
        self.assertEqual(signals["astra_finance_blend"], 1.0)
        self.assertEqual(signals["astra_finance_plus_gap"], 1.0)
        self.assertEqual(signals["astra_quality_weighted"], 0.8)

    @patch("astra.finance.build_signals.simple_title_body_divergence", return_value=0.0)
    def test_build_report_signal_locks_gap_scaled_and_composite_weight_relationship(self, _divergence_mock) -> None:
        prediction = {
            "fundamental_sentiment": "neutral",
            "strategic_optimism_gap": 0.6,
            "uncertainty": 0.0,
        }

        signals = build_report_signal(prediction, title="headline", summary="body")

        self.assertEqual(signals["astra_composite"], -0.6)
        self.assertEqual(signals["astra_gap_scaled"], -0.45)
        self.assertEqual(signals["astra_finance_blend"], -0.45)
        self.assertEqual(signals["astra_finance_plus_gap"], -0.54)
        self.assertEqual(signals["astra_gap_scaled"], round(0.75 * signals["counterfactual_gap"], 6))
        self.assertEqual(signals["astra_finance_blend"], signals["astra_gap_scaled"])

    @patch("astra.finance.build_signals.simple_title_body_divergence", return_value=0.0)
    def test_build_report_signal_intentionally_adds_extra_gap_on_top_of_finance_blend(self, _divergence_mock) -> None:
        prediction = {
            "fundamental_sentiment": "positive",
            "strategic_optimism_gap": 0.4,
            "uncertainty": 0.2,
            "astra_composite_score": 1.2,
        }

        signals = build_report_signal(prediction, title="headline", summary="body")

        self.assertEqual(signals["astra_finance_blend"], 1.15)
        self.assertEqual(signals["astra_finance_plus_gap"], 1.09)
        self.assertEqual(
            signals["astra_finance_plus_gap"],
            round(signals["astra_finance_blend"] + 0.15 * signals["counterfactual_gap"], 6),
        )
        self.assertLess(signals["astra_finance_plus_gap"], signals["astra_finance_blend"])


if __name__ == "__main__":
    unittest.main()
