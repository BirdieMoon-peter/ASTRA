import unittest

from astra.neutralization.counterfactual_neutralizer import _sanitize_neutralized_output


class SanitizeNeutralizedOutputTest(unittest.TestCase):
    def test_accepts_reasonably_longer_rewrite_without_new_numbers(self) -> None:
        result = _sanitize_neutralized_output(
            {
                "neutralized_text": "公司2025年收入10亿元，利润同比下滑，但仍推进新产品发布与渠道调整。",
                "removed_rhetoric": ["积极推进"],
                "preserved_facts": ["2025年收入10亿元", "利润同比下滑"],
            },
            title="公司积极推进新品",
            summary="公司2025年收入10亿元，利润同比下滑，但管理层积极推进新产品发布与渠道调整。",
        )

        self.assertEqual(result["neutralized_text"], "公司2025年收入10亿元，利润同比下滑，但仍推进新产品发布与渠道调整。")
        self.assertEqual(result["preserved_facts"], ["2025年收入10亿元", "利润同比下滑"])

    def test_falls_back_when_new_numbers_are_introduced(self) -> None:
        result = _sanitize_neutralized_output(
            {
                "neutralized_text": "公司2025年收入12亿元，利润同比下滑。",
                "removed_rhetoric": ["积极推进"],
                "preserved_facts": ["利润同比下滑"],
            },
            title="公司积极推进新品",
            summary="公司2025年收入10亿元，利润同比下滑。",
        )

        self.assertEqual(result["neutralized_text"], "公司2025年收入10亿元，利润同比下滑。")
        self.assertEqual(result["removed_rhetoric"], [])
        self.assertEqual(result["preserved_facts"], [])

    def test_falls_back_when_rewrite_is_far_too_long(self) -> None:
        long_text = "公司2025年收入10亿元，利润同比下滑。" + "中性描述" * 80
        result = _sanitize_neutralized_output(
            {
                "neutralized_text": long_text,
                "removed_rhetoric": [],
                "preserved_facts": ["公司2025年收入10亿元"],
            },
            title="标题",
            summary="公司2025年收入10亿元，利润同比下滑。",
        )

        self.assertEqual(result["neutralized_text"], "公司2025年收入10亿元，利润同比下滑。")


    def test_accepts_rewrite_when_known_rhetoric_is_removed(self) -> None:
        result = _sanitize_neutralized_output(
            {
                "neutralized_text": "公司2025年收入10亿元，利润同比下滑，推进新产品发布与渠道调整。",
                "removed_rhetoric": ["积极推进"],
                "preserved_facts": ["2025年收入10亿元", "利润同比下滑"],
            },
            title="公司积极推进新品",
            summary="公司2025年收入10亿元，利润同比下滑，但管理层积极推进新产品发布与渠道调整。",
        )

        self.assertEqual(result["neutralized_text"], "公司2025年收入10亿元，利润同比下滑，推进新产品发布与渠道调整。")
        self.assertEqual(result["removed_rhetoric"], ["积极推进"])

