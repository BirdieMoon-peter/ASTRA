import unittest

from astra.scoring.report_scorer import build_astra_prediction, refine_labels_from_pipeline, route_prediction_labels


class RoutePredictionLabelsTest(unittest.TestCase):
    def test_prefers_react_sentiment_and_react_evidence(self) -> None:
        routed = route_prediction_labels(
            direct_prediction={
                "fundamental_sentiment": "neutral",
                "strategic_optimism": "balanced",
                "phenomenon": "none",
                "uncertainty": 0.6,
                "evidence_spans": [{"text": "direct span", "label": "fact"}],
            },
            cot_prediction={
                "strategic_optimism": "high",
                "phenomenon": "hedged_downside",
                "uncertainty": 0.4,
            },
            react_prediction={
                "fundamental_sentiment": "positive",
                "strategic_optimism": "high",
                "phenomenon": "hedged_downside",
                "uncertainty": 0.2,
                "evidence_spans": [{"text": "react span", "label": "hedge"}],
                "reasoning_summary": "react reasoning",
            },
            rule_prediction={"phenomenon": "none"},
            title="标题强势",
            summary="正文也明显偏乐观，并承认短期承压。",
        )

        self.assertEqual(routed["fundamental_sentiment"], "positive")
        self.assertEqual(routed["strategic_optimism"], "high")
        self.assertEqual(routed["phenomenon"], "hedged_downside")
        self.assertEqual(routed["evidence_spans"], [{"text": "react span", "label": "hedge"}])
        self.assertEqual(routed["uncertainty"], 0.2)
        self.assertEqual(routed["reasoning_summary"], "react reasoning")

    def test_uses_cot_strategic_optimism_when_direct_is_missing(self) -> None:
        routed = route_prediction_labels(
            direct_prediction={"fundamental_sentiment": "neutral", "phenomenon": "none", "uncertainty": 0.5},
            cot_prediction={"strategic_optimism": "high", "uncertainty": 0.3},
            react_prediction={"fundamental_sentiment": "neutral", "uncertainty": 0.4},
            rule_prediction={"phenomenon": "none"},
            title="积极扩张",
            summary="公司预计需求向好。",
        )

        self.assertEqual(routed["strategic_optimism"], "high")

    def test_prefers_react_non_none_phenomenon_over_direct_none(self) -> None:
        routed = route_prediction_labels(
            direct_prediction={
                "fundamental_sentiment": "negative",
                "strategic_optimism": "balanced",
                "phenomenon": "none",
                "uncertainty": 0.5,
            },
            cot_prediction={},
            react_prediction={
                "fundamental_sentiment": "negative",
                "strategic_optimism": "balanced",
                "phenomenon": "euphemistic_risk",
                "uncertainty": 0.25,
            },
            rule_prediction={"phenomenon": "none"},
            title="短期扰动后有望恢复",
            summary="利润下滑但表述整体偏弱化。",
        )

        self.assertEqual(routed["phenomenon"], "euphemistic_risk")

    def test_falls_back_to_rule_phenomenon_when_llm_outputs_none(self) -> None:
        routed = route_prediction_labels(
            direct_prediction={
                "fundamental_sentiment": "negative",
                "strategic_optimism": "balanced",
                "phenomenon": "none",
                "uncertainty": 0.5,
            },
            cot_prediction={},
            react_prediction={
                "fundamental_sentiment": "negative",
                "strategic_optimism": "balanced",
                "phenomenon": "none",
                "uncertainty": 0.25,
            },
            rule_prediction={"phenomenon": "hedged_downside"},
            title="短期承压",
            summary="公司称短期承压但长期健康发展。",
        )

        self.assertEqual(routed["phenomenon"], "hedged_downside")


class BuildAstraPredictionTest(unittest.TestCase):
    def setUp(self) -> None:
        self.report = {
            "report_id": "r1",
            "report_date": "2026-01-01",
            "stock_code": "000001",
            "split": "test",
            "title": "标题",
            "summary": "摘要",
        }
        self.routed_prediction = {
            "fundamental_sentiment": "positive",
            "strategic_optimism": "high",
            "phenomenon": "hedged_downside",
            "uncertainty": 0.2,
            "evidence_spans": [
                {"text": "短期承压", "label": "hedge"},
                {"text": "风险一", "label": "risk"},
                {"text": "风险二", "label": "risk"},
                {"text": "风险三", "label": "risk"},
            ],
            "reasoning_summary": "存在风险弱化措辞",
        }
        self.decomposition = {
            "history_context_used": "历史中提示风险\n下滑风险",
            "hedge_cues": ["短期"],
            "risk_cues": ["承压"],
            "missing_risk_hints": ["下滑风险"],
        }
        self.neutralization = {
            "neutralized_text": "中性摘要",
            "removed_rhetoric": ["高成长"],
            "preserved_facts": ["摘要"],
        }
        self.verifier_result = {"factual_consistency": 0.9, "verdict": "pass"}

    def test_build_astra_prediction_includes_non_none_phenomenon_from_routed_prediction(self) -> None:
        result = build_astra_prediction(
            report=self.report,
            direct_prediction=self.routed_prediction,
            decomposition=self.decomposition,
            neutralization=self.neutralization,
            verifier_result=self.verifier_result,
        )

        self.assertEqual(result["phenomenon"], "hedged_downside")
        self.assertEqual(result["evidence_spans"], self.routed_prediction["evidence_spans"])
        self.assertEqual(result["reasoning_summary"], "存在风险弱化措辞")
        self.assertGreater(result["strategic_optimism_gap"], 0.0)

    def test_ablation_flags_change_outputs(self) -> None:
        ablation_prediction = {**self.routed_prediction, "fundamental_sentiment": "neutral", "strategic_optimism": "high"}
        baseline = build_astra_prediction(
            report=self.report,
            direct_prediction=ablation_prediction,
            decomposition=self.decomposition,
            neutralization=self.neutralization,
            verifier_result=self.verifier_result,
        )
        minus_retrieval = build_astra_prediction(
            report=self.report,
            direct_prediction=ablation_prediction,
            decomposition={**self.decomposition, "history_context_used": "无历史研报上下文"},
            neutralization=self.neutralization,
            verifier_result=self.verifier_result,
            use_retrieval=False,
        )
        minus_neutralizer = build_astra_prediction(
            report=self.report,
            direct_prediction=ablation_prediction,
            decomposition=self.decomposition,
            neutralization={},
            verifier_result=self.verifier_result,
            use_neutralizer=False,
        )
        minus_verifier = build_astra_prediction(
            report=self.report,
            direct_prediction=ablation_prediction,
            decomposition=self.decomposition,
            neutralization=self.neutralization,
            verifier_result={},
            use_verifier=False,
        )
        minus_uncertainty_gate = build_astra_prediction(
            report=self.report,
            direct_prediction=ablation_prediction,
            decomposition=self.decomposition,
            neutralization=self.neutralization,
            verifier_result=self.verifier_result,
            use_uncertainty_gate=False,
        )
        minus_analyst_prior = build_astra_prediction(
            report=self.report,
            direct_prediction=ablation_prediction,
            decomposition=self.decomposition,
            neutralization=self.neutralization,
            verifier_result=self.verifier_result,
            use_analyst_prior=False,
        )

        self.assertEqual(minus_retrieval["phenomenon"], "none")
        self.assertLess(minus_retrieval["omission_penalty"], baseline["omission_penalty"])
        self.assertEqual(len(minus_neutralizer["evidence_spans"]), 2)
        self.assertLess(minus_neutralizer["strategic_optimism_gap"], baseline["strategic_optimism_gap"])
        self.assertGreater(minus_neutralizer["counterfactual_sentiment_score"], baseline["counterfactual_sentiment_score"])
        self.assertTrue(minus_verifier["reasoning_summary"].startswith("[verifier disabled]"))
        self.assertNotEqual(minus_uncertainty_gate["astra_uncertainty_gated_score"], baseline["astra_uncertainty_gated_score"])
        self.assertEqual(minus_analyst_prior["strategic_optimism"], "high")
        self.assertNotEqual(minus_analyst_prior["astra_composite_score"], baseline["astra_composite_score"])


class RefinePipelineLabelsTest(unittest.TestCase):
    def test_retrieval_upgrades_phenomenon_on_missing_risk_hints(self) -> None:
        routed = {"fundamental_sentiment": "positive", "strategic_optimism": "balanced", "phenomenon": "none"}
        decomposition = {"missing_risk_hints": ["风险A", "风险B"], "risk_cues": ["承压"]}

        with_retrieval = refine_labels_from_pipeline(
            routed, decomposition=decomposition, use_retrieval=True,
        )
        without_retrieval = refine_labels_from_pipeline(
            routed, decomposition=decomposition, use_retrieval=False,
        )

        self.assertEqual(with_retrieval["phenomenon"], "omitted_downside_context")
        self.assertEqual(without_retrieval["phenomenon"], "none")

    def test_retrieval_upgrades_strategic_optimism(self) -> None:
        routed = {"fundamental_sentiment": "neutral", "strategic_optimism": "balanced", "phenomenon": "none"}
        decomposition = {"missing_risk_hints": ["风险A", "风险B"], "risk_cues": []}

        with_retrieval = refine_labels_from_pipeline(
            routed, decomposition=decomposition, use_retrieval=True,
        )
        without_retrieval = refine_labels_from_pipeline(
            routed, decomposition=decomposition, use_retrieval=False,
        )

        self.assertEqual(with_retrieval["strategic_optimism"], "high")
        self.assertEqual(without_retrieval["strategic_optimism"], "balanced")

    def test_neutralizer_upgrades_strategic_optimism_on_removed_rhetoric(self) -> None:
        routed = {"fundamental_sentiment": "neutral", "strategic_optimism": "balanced", "phenomenon": "none"}
        neutralization = {"removed_rhetoric": ["乐观措辞A", "乐观措辞B"]}

        with_neutralizer = refine_labels_from_pipeline(
            routed, neutralization=neutralization, use_neutralizer=True,
        )
        without_neutralizer = refine_labels_from_pipeline(
            routed, neutralization=neutralization, use_neutralizer=False,
        )

        self.assertEqual(with_neutralizer["strategic_optimism"], "high")
        self.assertEqual(without_neutralizer["strategic_optimism"], "balanced")

    def test_neutralizer_detects_euphemistic_risk_on_heavy_rhetoric(self) -> None:
        routed = {"fundamental_sentiment": "neutral", "strategic_optimism": "high", "phenomenon": "none"}
        neutralization = {"removed_rhetoric": ["A", "B", "C"]}

        with_neutralizer = refine_labels_from_pipeline(
            routed, neutralization=neutralization, use_neutralizer=True,
        )
        without_neutralizer = refine_labels_from_pipeline(
            routed, neutralization=neutralization, use_neutralizer=False,
        )

        self.assertEqual(with_neutralizer["phenomenon"], "euphemistic_risk")
        self.assertEqual(without_neutralizer["phenomenon"], "none")

    def test_no_refinement_when_components_disabled(self) -> None:
        routed = {"fundamental_sentiment": "positive", "strategic_optimism": "balanced", "phenomenon": "none"}
        decomposition = {"missing_risk_hints": ["风险A", "风险B", "风险C"], "risk_cues": ["承压"]}
        neutralization = {"removed_rhetoric": ["乐观A", "乐观B", "乐观C"]}

        all_disabled = refine_labels_from_pipeline(
            routed,
            decomposition=decomposition,
            neutralization=neutralization,
            use_retrieval=False,
            use_neutralizer=False,
        )

        self.assertEqual(all_disabled["phenomenon"], "none")
        self.assertEqual(all_disabled["strategic_optimism"], "balanced")


if __name__ == "__main__":
    unittest.main()
