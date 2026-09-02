"""Single source of truth for prompt/instruction construction — train and inference must both import this."""
from grammar import PromptCell

SYSTEM_PROMPT = (
    "You are a news writer. Given a one-line event description, write a "
    "news article with a headline and body in your outlet's style."
)


def format_instruction(event_desc: str) -> list[dict]:
    """The only function in the codebase that builds the training/inference prompt message list."""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": event_desc},
    ]


def render_prompt_cell(cell: PromptCell, grammar: dict) -> str:
    """Render a PromptCell into a neutral, fully-substituted event description string."""
    event_template = next(e["template"] for e in grammar["event_types"] if e["id"] == cell.event_type)
    politician_name = next(p["display_name"] for p in grammar["politicians"] if p["id"] == cell.politician)
    base = event_template.format(politician=politician_name, policy_area=cell.policy_area.replace("_", " "))
    return f"{base} at {cell.setting}; result: {cell.outcome.replace('_', ' ')}."
