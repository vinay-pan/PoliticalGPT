import json
import random
import sqlite3
import time
from pathlib import Path

POLITICIANS = ["Trump", "Biden", "Obama", "Hillary", "Kamala"]


def setup_db(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS seen_urls (
            url TEXT PRIMARY KEY,
            scraped_at REAL NOT NULL
        )
    """)
    conn.commit()
    return conn


def is_seen(conn: sqlite3.Connection, url: str) -> bool:
    row = conn.execute("SELECT 1 FROM seen_urls WHERE url = ?", (url,)).fetchone()
    return row is not None


def mark_seen(conn: sqlite3.Connection, url: str) -> None:
    conn.execute(
        "INSERT OR IGNORE INTO seen_urls (url, scraped_at) VALUES (?, ?)",
        (url, time.time()),
    )
    conn.commit()


def save_article(jsonl_path: str, article: dict) -> None:
    with open(jsonl_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(article, ensure_ascii=False) + "\n")


def random_delay(min_s: float = 1.0, max_s: float = 2.5) -> None:
    time.sleep(random.uniform(min_s, max_s))


def mentions_politician(text: str) -> list[str]:
    text_lower = text.lower()
    found = []
    for p in POLITICIANS:
        if p.lower() in text_lower:
            found.append(p)
    return found


def count_saved(jsonl_path: str) -> int:
    path = Path(jsonl_path)
    if not path.exists():
        return 0
    with open(path, encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())
