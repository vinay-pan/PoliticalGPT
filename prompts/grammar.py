"""Mad-libs prompt cell enumeration and sampling over prompts/grammar.yaml."""
import itertools
import random
from dataclasses import dataclass


@dataclass(frozen=True)
class PromptCell:
    politician: str
    event_type: str
    setting: str
    policy_area: str
    outcome: str


def enumerate_all(grammar: dict) -> list[PromptCell]:
    """Full cross-product over all 5 slot dimensions — no filtering, per D-04."""
    politician_ids = [p["id"] for p in grammar["politicians"]]
    event_type_ids = [e["id"] for e in grammar["event_types"]]
    return [
        PromptCell(politician=politician, event_type=event_type, setting=setting,
                   policy_area=policy_area, outcome=outcome)
        for politician, event_type, setting, policy_area, outcome in itertools.product(
            politician_ids, event_type_ids, grammar["settings"],
            grammar["policy_areas"], grammar["outcomes"],
        )
    ]


def sample(grammar: dict, n: int, seed: int) -> list[PromptCell]:
    """Deterministic sample of n cells — same (grammar, n, seed) always returns the same list."""
    return random.Random(seed).sample(enumerate_all(grammar), n)
