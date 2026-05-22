"""
Converts raw scraped JSONL articles into instruction-tuning format for Llama 3.

Input:  data/raw/cnn.jsonl, data/raw/fox.jsonl
Output: data/cnn_train.jsonl, data/fox_train.jsonl
        data/cnn_eval.jsonl,  data/fox_eval.jsonl

Each output record is a message list:
  [
    {"role": "system",    "content": "You are a news writer in the style of CNN."},
    {"role": "user",      "content": "Write a news article about Biden."},
    {"role": "assistant", "content": "<full article text>"}
  ]
"""

import json
import random
from collections import defaultdict
from datetime import datetime
from pathlib import Path

POLITICIANS = ["Obama", "Hillary", "Kamala", "Biden", "Trump"]
MIN_WORDS = 200
MAX_WORDS = 2000
MIN_YEAR = 2015
MAX_YEAR = 2024
EVAL_FRACTION = 0.1
SEED = 42

SOURCE_DISPLAY = {"cnn": "CNN", "fox": "Fox News"}


def parse_year(date_str):
    if not date_str:
        return None
    for fmt in ("%Y-%m-%d", "%Y-%m", "%Y"):
        try:
            return datetime.strptime(date_str, fmt).year
        except ValueError:
            continue
    return None


def first_politician(article):
    """Return the first target politician mentioned in the article text."""
    text = article.get("text", "")
    for p in POLITICIANS:
        if p in text:
            return p
    # Fall back to politicians list recorded by the scraper
    recorded = article.get("politicians", [])
    if recorded:
        return recorded[0]
    return None


def to_messages(article, politician):
    source_label = SOURCE_DISPLAY.get(article["source"], article["source"])
    return [
        {"role": "system", "content": f"You are a news writer in the style of {source_label}."},
        {"role": "user", "content": f"Write a news article about {politician}."},
        {"role": "assistant", "content": article["text"].strip()},
    ]


def load_and_filter(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Missing: {path}")

    kept, dropped = [], defaultdict(int)

    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                article = json.loads(line)
            except json.JSONDecodeError:
                dropped["bad_json"] += 1
                continue

            word_count = len(article.get("text", "").split())
            if word_count < MIN_WORDS:
                dropped["too_short"] += 1
                continue
            if word_count > MAX_WORDS:
                dropped["too_long"] += 1
                continue

            year = parse_year(article.get("date"))
            if year is None or not (MIN_YEAR <= year <= MAX_YEAR):
                dropped["bad_date"] += 1
                continue

            politician = first_politician(article)
            if politician is None:
                dropped["no_politician"] += 1
                continue

            kept.append((article, politician))

    return kept, dropped


def split_by_politician(records, eval_fraction, seed):
    """Stratified split: sample eval_fraction from each politician bucket."""
    rng = random.Random(seed)
    buckets = defaultdict(list)
    for article, politician in records:
        buckets[politician].append((article, politician))

    train, eval_ = [], []
    for pol, items in buckets.items():
        rng.shuffle(items)
        n_eval = max(1, round(len(items) * eval_fraction))
        eval_.extend(items[:n_eval])
        train.extend(items[n_eval:])

    rng.shuffle(train)
    rng.shuffle(eval_)
    return train, eval_


def write_jsonl(records, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for article, politician in records:
            record = {"messages": to_messages(article, politician)}
            f.write(json.dumps(record) + "\n")


def print_summary(source, kept, dropped, train, eval_):
    print(f"\n=== {source.upper()} ===")
    print(f"  kept:    {len(kept):>6,}")
    for reason, count in sorted(dropped.items()):
        print(f"  dropped ({reason}): {count:,}")
    print(f"  train:   {len(train):>6,}")
    print(f"  eval:    {len(eval_):>6,}")

    pol_counts = defaultdict(int)
    for _, pol in kept:
        pol_counts[pol] += 1
    print("  by politician:")
    for pol in POLITICIANS:
        print(f"    {pol:<10} {pol_counts[pol]:,}")

    word_counts = [len(a["text"].split()) for a, _ in kept]
    if word_counts:
        print(f"  word count — min: {min(word_counts)}, "
              f"max: {max(word_counts)}, "
              f"avg: {sum(word_counts)//len(word_counts)}")


def process(source, raw_path, train_path, eval_path):
    kept, dropped = load_and_filter(raw_path)
    train, eval_ = split_by_politician(kept, EVAL_FRACTION, SEED)
    write_jsonl(train, train_path)
    write_jsonl(eval_, eval_path)
    print_summary(source, kept, dropped, train, eval_)


if __name__ == "__main__":
    root = Path(__file__).parent.parent

    process(
        source="cnn",
        raw_path=root / "data/raw/cnn.jsonl",
        train_path=root / "data/cnn_train.jsonl",
        eval_path=root / "data/cnn_eval.jsonl",
    )
    process(
        source="fox",
        raw_path=root / "data/raw/fox.jsonl",
        train_path=root / "data/fox_train.jsonl",
        eval_path=root / "data/fox_eval.jsonl",
    )

    print("\nDone. Output files in data/")
