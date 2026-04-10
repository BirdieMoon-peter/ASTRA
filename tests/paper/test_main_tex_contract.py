import unittest
from pathlib import Path


class MainTexContractTests(unittest.TestCase):
    def test_main_tex_inputs_generated_tables(self) -> None:
        main_tex = Path("paperpkg/main.tex").read_text(encoding="utf-8")
        self.assertIn(r"\input{../outputs/paper/latex/tab_main_nlp.tex}", main_tex)
        self.assertIn(r"\input{../outputs/paper/latex/tab_phenomena.tex}", main_tex)
        self.assertIn(r"\input{../outputs/paper/latex/tab_finance.tex}", main_tex)
        self.assertIn(r"\input{../outputs/paper/latex/tab_ablation.tex}", main_tex)
        self.assertIn(r"\input{../outputs/paper/latex/tab_case.tex}", main_tex)


if __name__ == "__main__":
    unittest.main()
