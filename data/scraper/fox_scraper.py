"""
Fox News politics article scraper — parallel async version.

URL discovery uses Fox's public article sitemaps (no browser needed):
  https://www.foxnews.com/sitemap.xml?type=articles&page=N  (pages 1–165)

Each page contains ~10k URLs across all sections; we filter to /politics/.
N async workers each hold their own browser page and pull from a shared queue.

Usage:
    python fox_scraper.py                        # full run, 4 workers
    python fox_scraper.py --workers 6            # more parallelism
    python fox_scraper.py --target 25000         # stop after 25k articles
    python fox_scraper.py --limit 50             # small test run

Output: data/raw/fox.jsonl
Schema: {source, url, date, title, text, politicians}
"""

import argparse
import asyncio
import random
import sys
import time
from pathlib import Path

import httpx
from bs4 import BeautifulSoup
from playwright.async_api import Page, async_playwright

sys.path.insert(0, str(Path(__file__).parent))
from utils import (
    count_saved,
    mentions_politician,
    setup_db,
)

RAW_DIR = Path(__file__).parent.parent / "raw"
JSONL_PATH = str(RAW_DIR / "fox.jsonl")
DB_PATH = str(RAW_DIR / "fox_seen.db")

SITEMAP_BASE = "https://www.foxnews.com/sitemap.xml?type=articles&page={page}"
SITEMAP_PAGES = range(1, 166)
USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/124.0.0.0 Safari/537.36"
)


# ── URL discovery (sync, no browser) ──────────────────────────────────────────

def collect_urls_from_sitemaps() -> list[str]:
    all_urls = []
    with httpx.Client(headers={"User-Agent": USER_AGENT}, timeout=30, follow_redirects=True) as client:
        for page in SITEMAP_PAGES:
            url = SITEMAP_BASE.format(page=page)
            try:
                r = client.get(url)
                if r.status_code != 200:
                    print(f"  page {page:3d}: HTTP {r.status_code}, skipping")
                    continue
                soup = BeautifulSoup(r.text, "xml")
                locs = [u.find("loc").text for u in soup.find_all("url") if u.find("loc")]
                politics = [
                    l for l in locs
                    if "/politics/" in l
                    and "/video/" not in l
                    and "/slideshow/" not in l
                ]
                all_urls.extend(politics)
                print(f"  page {page:3d}: {len(politics)} politics articles (running total: {len(all_urls)})")
            except Exception as e:
                print(f"  page {page:3d}: error — {e}")
            time.sleep(0.5)
    return all_urls


# ── Article scraping ───────────────────────────────────────────────────────────

async def scrape_article(page: Page, url: str) -> dict | None:
    try:
        await page.goto(url, wait_until="domcontentloaded", timeout=25000)
        html = await page.content()
    except Exception as e:
        print(f"  [skip] {url[-60:]} — {e}")
        return None

    soup = BeautifulSoup(html, "html.parser")

    title_el = soup.find("h1")
    if not title_el:
        return None
    title = title_el.get_text(strip=True)

    date = ""
    time_el = soup.find("time", {"datetime": True})
    if time_el:
        date = time_el["datetime"][:10]
    else:
        meta_date = soup.find("meta", {"property": "article:published_time"})
        if meta_date and meta_date.get("content"):
            date = meta_date["content"][:10]

    content_el = (
        soup.find("div", class_=lambda c: c and "article-body" in c)
        or soup.find("div", class_=lambda c: c and "article-content" in c)
        or soup.find("article")
    )
    if not content_el:
        return None

    paragraphs = [p.get_text(strip=True) for p in content_el.find_all("p") if p.get_text(strip=True)]
    text = "\n".join(paragraphs)

    if len(text.split()) < 100:
        return None

    politicians = mentions_politician(title + " " + text)
    if not politicians:
        return None

    return {"source": "fox", "url": url, "date": date, "title": title, "text": text, "politicians": politicians}


# ── Parallel worker ────────────────────────────────────────────────────────────

async def worker(
    worker_id: int,
    queue: asyncio.Queue,
    page: Page,
    write_lock: asyncio.Lock,
    db_lock: asyncio.Lock,
    conn,
    target: int | None,
    stop_event: asyncio.Event,
) -> None:
    import json, random, time

    while not stop_event.is_set():
        try:
            url = queue.get_nowait()
        except asyncio.QueueEmpty:
            break

        async with write_lock:
            if target and count_saved(JSONL_PATH) >= target:
                stop_event.set()
                queue.task_done()
                break

        async with db_lock:
            conn.execute(
                "INSERT OR IGNORE INTO seen_urls (url, scraped_at) VALUES (?, ?)",
                (url, time.time()),
            )
            conn.commit()

        article = await scrape_article(page, url)

        if article:
            async with write_lock:
                with open(JSONL_PATH, "a", encoding="utf-8") as f:
                    f.write(json.dumps(article, ensure_ascii=False) + "\n")
                n = count_saved(JSONL_PATH)
                print(f"  [w{worker_id}] #{n} — {article['title'][:55]}")
                if target and n >= target:
                    stop_event.set()

        await asyncio.sleep(random.uniform(1.0, 2.5))
        queue.task_done()


# ── Main ───────────────────────────────────────────────────────────────────────

async def run(limit: int | None, target: int | None, workers: int) -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    conn = setup_db(DB_PATH)

    print(f"Starting Fox News scrape. Already saved: {count_saved(JSONL_PATH)} articles.")
    print("\nCollecting politics URLs from sitemaps (pages 1–165)...")
    all_urls = list(set(collect_urls_from_sitemaps()))
    print(f"\nTotal unique politics candidate URLs: {len(all_urls)}")

    seen = set(
        row[0] for row in conn.execute("SELECT url FROM seen_urls").fetchall()
    )
    pending = [u for u in all_urls if u not in seen]
    random.shuffle(pending)

    if limit:
        pending = pending[:limit]

    print(f"URLs to scrape (excluding already seen): {len(pending)}")

    queue: asyncio.Queue = asyncio.Queue()
    for url in pending:
        await queue.put(url)

    write_lock = asyncio.Lock()
    db_lock = asyncio.Lock()
    stop_event = asyncio.Event()

    async with async_playwright() as pw:
        browser = await pw.chromium.launch(headless=True)
        pages = [
            await (await browser.new_context(user_agent=USER_AGENT)).new_page()
            for _ in range(workers)
        ]

        print(f"\nScraping with {workers} parallel workers...\n")
        await asyncio.gather(*[
            worker(i + 1, queue, pages[i], write_lock, db_lock, conn, target, stop_event)
            for i in range(workers)
        ])

        await browser.close()

    conn.close()
    print(f"\nDone. Total saved: {count_saved(JSONL_PATH)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Fox News politics article scraper (parallel)")
    parser.add_argument("--workers", type=int, default=4, help="Number of parallel browser pages (default: 4)")
    parser.add_argument("--target", type=int, default=None, help="Stop after saving this many articles")
    parser.add_argument("--limit", type=int, default=None, help="Cap URLs to attempt (for testing)")
    args = parser.parse_args()
    asyncio.run(run(limit=args.limit, target=args.target, workers=args.workers))


if __name__ == "__main__":
    main()
