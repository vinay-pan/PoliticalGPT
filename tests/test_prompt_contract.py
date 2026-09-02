import sys
import unittest
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).parents[1] / "prompts"))
import templates, grammar  # noqa: E402


class PromptContractTests(unittest.TestCase):
    def setUp(self):
        with open(Path(__file__).parents[1] / "prompts" / "grammar.yaml") as f:
            self.grammar = yaml.safe_load(f)

    def test_format_instruction_returns_system_and_user_roles(self):
        messages = templates.format_instruction("Some event happened.")
        self.assertEqual(len(messages), 2)
        self.assertEqual(messages[0]["role"], "system")
        self.assertEqual(messages[0]["content"], templates.SYSTEM_PROMPT)
        self.assertEqual(messages[1]["role"], "user")
        self.assertEqual(messages[1]["content"], "Some event happened.")

    def test_format_instruction_is_byte_identical_across_calls(self):
        first = templates.format_instruction("x")
        second = templates.format_instruction("x")
        self.assertEqual(first, second)

    def test_render_prompt_cell_produces_neutral_description(self):
        cell = grammar.PromptCell(
            politician="trump", event_type="policy_announcement", setting="a press conference",
            policy_area="healthcare", outcome="passed",
        )
        rendered = templates.render_prompt_cell(cell, self.grammar)
        self.assertIn("Donald Trump", rendered)
        self.assertNotIn("{", rendered)
        self.assertNotIn("}", rendered)

    def test_enumerate_all_full_cross_product_size(self):
        cells = grammar.enumerate_all(self.grammar)
        expected = (
            len(self.grammar["politicians"]) * len(self.grammar["event_types"])
            * len(self.grammar["settings"]) * len(self.grammar["policy_areas"])
            * len(self.grammar["outcomes"])
        )
        self.assertEqual(len(cells), expected)

    def test_sample_is_deterministic_given_seed(self):
        first = grammar.sample(self.grammar, 10, seed=42)
        second = grammar.sample(self.grammar, 10, seed=42)
        self.assertEqual(first, second)


if __name__ == "__main__":
    unittest.main()
