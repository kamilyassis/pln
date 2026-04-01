import asyncio
import json
import os
import zipfile
import glob
import gc
import signal
import sys
from datetime import datetime
from pathlib import Path
from bs4 import BeautifulSoup
from typing import List, Dict, Optional
import random
import time

# ─────────────────────────────────────────────
# Low-level helpers
# ─────────────────────────────────────────────

def human_delay(min_s=1.5, max_s=4.0):
    time.sleep(random.uniform(min_s, max_s))


def scrape_article_with_retry(url: str, retries: int = 3, retry_delay: float = 8.0) -> str:
    """
    Open a fresh browser per URL (safest with SeleniumBase UC mode).
    Retries up to `retries` times if the session dies or any error occurs.
    Each retry waits `retry_delay` seconds to let the OS clean up the old process.
    """
    from seleniumbase import SB

    last_error = None
    for attempt in range(1, retries + 1):
        try:
            with SB(
                uc=True,
                headless=False,
                locale_code="en-US",
                window_size="1367,769",
            ) as sb:
                sb.uc_open_with_reconnect(url, reconnect_time=4)
                sb.execute_script("window.scrollTo({ top: 900, behavior: 'smooth' });")
                human_delay(1, 2)
                sb.wait_for_element("body", timeout=10)
                html = sb.get_page_source()

            # SB context exited cleanly — return immediately
            return html

        except Exception as e:
            last_error = e
            err_str = str(e)[:120]
            if attempt < retries:
                print(f"  ⚠️  Attempt {attempt}/{retries} failed for {url[:60]}…")
                print(f"      {err_str}")
                print(f"      Retrying in {retry_delay}s…")
                time.sleep(retry_delay)
            else:
                print(f"  ✗  All {retries} attempts failed for {url[:60]}…")
                print(f"      {err_str}")

    raise RuntimeError(f"Scrape failed after {retries} attempts: {last_error}")


def parse_g1_article(soup):
    article_data = {
        'title': None,
        'subtitle': None,
        'from_publication': None,
        'date_publication': None,
        'json_ld': None,
        'content': [],
    }

    header = soup.find('div', class_='mc-article-header')
    if header:
        title_div = header.find('div', class_='title')
        if title_div:
            article_data['title'] = title_div.get_text(strip=True)

        subtitle_div = header.find('div', class_=lambda x: x and 'subtitle' in x if x else False)
        if subtitle_div:
            article_data['subtitle'] = subtitle_div.get_text(strip=True)

        from_pub_data = header.find('p', class_=lambda x: x and 'content-publication-data__from' in x if x else False)
        if from_pub_data:
            article_data['from_publication'] = from_pub_data.get_text(strip=True)

        updated_pub_data = header.find('p', class_=lambda x: x and 'content-publication-data__updated' in x if x else False)
        if updated_pub_data:
            article_data['date_publication'] = updated_pub_data.get_text(strip=True)

    body = soup.find('div', class_='mc-article-body')
    if body:
        json_ld_scripts = body.find_all('script', type='application/ld+json')
        if json_ld_scripts:
            article_data['json_ld'] = []
            for script in json_ld_scripts:
                try:
                    article_data['json_ld'].append(json.loads(script.string))
                except json.JSONDecodeError:
                    article_data['json_ld'].append(script.string)

        article_body = body.find('article', attrs={'itemprop': 'articleBody'})
        if article_body:
            content_tags = article_body.find_all([
                'p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6',
                'ul', 'li', 'ol', 'blockquote', 'figcaption',
                'table', 'tr', 'td', 'th', 'section',
                'pre', 'code', 'strong', 'em'
            ])
            for tag in content_tags:
                article_data['content'].append({
                    'tag': tag.name,
                    'text': tag.get_text(strip=True),
                    'html': str(tag)
                })

    return article_data


# ─────────────────────────────────────────────
# State / progress tracking
# ─────────────────────────────────────────────

class ProgressTracker:
    """
    Persists two things to disk atomically:
      - done_urls.jsonl  : one URL per line, already successfully parsed & saved
      - pending_html.jsonl: HTML that was scraped but not yet parsed (survives crashes)

    On startup, we reload both so we can:
      1. Skip already-done URLs
      2. Re-parse any HTML that was scraped but never written to articles.jsonl
    """

    def __init__(self, state_dir: Path):
        self.state_dir = state_dir
        self.state_dir.mkdir(parents=True, exist_ok=True)

        self.done_file    = state_dir / "done_urls.jsonl"
        self.pending_file = state_dir / "pending_html.jsonl"

    # ── done URLs ──────────────────────────────

    def load_done_urls(self) -> set:
        done = set()
        if self.done_file.exists():
            for line in self.done_file.read_text(encoding='utf-8').splitlines():
                line = line.strip()
                if line:
                    try:
                        done.add(json.loads(line)['url'])
                    except Exception:
                        pass
        return done

    def mark_done(self, url: str):
        with open(self.done_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps({'url': url, 'ts': datetime.now().isoformat()}) + '\n')

    # ── pending HTML (scraped but not yet parsed) ──

    def load_pending_html(self) -> list:
        """Returns list of (url, html_path) tuples."""
        pending = []
        if self.pending_file.exists():
            for line in self.pending_file.read_text(encoding='utf-8').splitlines():
                line = line.strip()
                if line:
                    try:
                        entry = json.loads(line)
                        if entry.get('status') == 'scraped':
                            pending.append((entry['url'], entry['html_path']))
                    except Exception:
                        pass
        return pending

    def mark_scraped(self, url: str, html_path: str):
        """Call right after HTML is written to disk."""
        with open(self.pending_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps({
                'url': url,
                'html_path': html_path,
                'status': 'scraped',
                'ts': datetime.now().isoformat()
            }) + '\n')

    def mark_parsed(self, url: str, html_path: str):
        """Call after article is written to articles.jsonl."""
        with open(self.pending_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps({
                'url': url,
                'html_path': html_path,
                'status': 'parsed',
                'ts': datetime.now().isoformat()
            }) + '\n')

    def rebuild_pending(self, done_urls: set) -> list:
        """
        After loading, compute which (url, html_path) pairs are still pending:
        scraped but not marked as parsed AND not in done_urls.
        """
        scraped: dict = {}
        parsed:  set  = set()

        if self.pending_file.exists():
            for line in self.pending_file.read_text(encoding='utf-8').splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                    url = entry['url']
                    if entry['status'] == 'scraped':
                        scraped[url] = entry['html_path']
                    elif entry['status'] == 'parsed':
                        parsed.add(url)
                except Exception:
                    pass

        still_pending = []
        for url, html_path in scraped.items():
            if url not in parsed and url not in done_urls:
                if Path(html_path).exists():
                    still_pending.append((url, html_path))
                else:
                    print(f"⚠️  HTML file missing for {url[:60]}, will re-scrape")

        return still_pending


# ─────────────────────────────────────────────
# Main scraper
# ─────────────────────────────────────────────

class G1ArticleScraperOptimized:
    """
    Sequential scraper: one browser opens, scrapes, closes — then the next starts.
    This is required for SeleniumBase UC mode, which cannot safely share a process
    with a second browser instance (causes 'invalid session id' errors).

    Parsing happens in a background async task so it overlaps with the next scrape.
    """

    def __init__(
        self,
        output_dir:  str   = "data/scraped_articles",
        html_dir:    str   = "data/html_cache",
        state_dir:   str   = "data/scraper_state",
        min_wait:    float = 6.0,
        max_wait:    float = 7.0,
        zip_interval: int  = 100,
        retries:     int   = 3,       # per-URL scrape retries
        retry_delay: float = 8.0,     # seconds between retries
    ):
        self.output_dir   = Path(output_dir)
        self.html_dir     = Path(html_dir)
        self.min_wait     = min_wait
        self.max_wait     = max_wait
        self.zip_interval = zip_interval
        self.retries      = retries
        self.retry_delay  = retry_delay

        self.html_counter   = 0
        self.parsed_counter = 0
        self.failed_counter = 0
        self.batch_counter  = 0

        self.results_file = self.output_dir / "articles.jsonl"
        self.failed_file  = self.output_dir / "failed_urls.jsonl"

        # Single-item queue: scraper puts one result, parser consumes it.
        # Keeps parsing concurrent with the next scrape without any parallelism issues.
        self.scrape_queue: asyncio.Queue = asyncio.Queue(maxsize=4)

        self._shutdown = False

        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.html_dir.mkdir(parents=True, exist_ok=True)

        self.tracker = ProgressTracker(Path(state_dir))

    # ── graceful shutdown ──────────────────────

    def _handle_signal(self, signum, frame):
        print(f"\n⚠️  Signal {signum} received — finishing current article then shutting down…")
        self._shutdown = True

    # ── scraping (sequential) ──────────────────

    async def _scrape_one(self, url: str) -> Optional[tuple]:
        """Scrape one URL. Browser fully closes before this returns."""
        try:
            await asyncio.sleep(random.uniform(self.min_wait, self.max_wait))

            loop = asyncio.get_event_loop()
            # Pass retries/retry_delay so recovery happens inside the executor thread
            html = await loop.run_in_executor(
                None,
                lambda: scrape_article_with_retry(url, self.retries, self.retry_delay)
            )

            filename = f"{self.html_counter:06d}_article.html"
            filepath = self.html_dir / filename

            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(html)

            self.tracker.mark_scraped(url, str(filepath))
            self.html_counter += 1
            print(f"✓ Scraped [{self.html_counter}]: {url[:65]}… ({len(html):,} chars)")

            if self.html_counter % self.zip_interval == 0:
                await self.zip_batch_async(self.batch_counter)
                self.batch_counter += 1

            return (url, html, str(filepath))

        except Exception as e:
            print(f"✗ Scrape failed: {url[:65]}…\n  {str(e)[:120]}")
            await self.log_failed_url(url, str(e))
            return None

    # ── parsing (background task) ──────────────

    async def parse_worker(self):
        """Consumes from the queue and writes to JSONL as fast as possible."""
        while True:
            try:
                result = await asyncio.wait_for(self.scrape_queue.get(), timeout=5.0)
            except asyncio.TimeoutError:
                if self._shutdown:
                    break
                continue

            if result is None:  # sentinel — time to stop
                break

            url, html, filepath = result
            try:
                soup = BeautifulSoup(html, 'html.parser')
                article_data = parse_g1_article(soup)
                article_data['url']        = url
                article_data['html_path']  = filepath
                article_data['scraped_at'] = datetime.now().isoformat()

                with open(self.results_file, 'a', encoding='utf-8') as f:
                    f.write(json.dumps(article_data, ensure_ascii=False) + '\n')

                self.tracker.mark_parsed(url, filepath)
                self.tracker.mark_done(url)
                self.parsed_counter += 1
                print(f"📝 Parsed [{self.parsed_counter}]: {url[:65]}… ({len(article_data['content'])} blocks)")

            except Exception as e:
                print(f"✗ Parse failed: {url[:65]}…\n  {str(e)[:120]}")
                await self.log_failed_url(url, f"Parse error: {str(e)}")

            del html
            gc.collect()

    # ── re-parse HTML from a previous crashed run ──

    async def reparse_pending(self, pending: list):
        if not pending:
            return
        print(f"\n🔄 Recovering {len(pending)} HTML file(s) from previous run…")
        for url, html_path in pending:
            try:
                html = Path(html_path).read_text(encoding='utf-8')
                soup = BeautifulSoup(html, 'html.parser')
                article_data = parse_g1_article(soup)
                article_data['url']        = url
                article_data['html_path']  = html_path
                article_data['scraped_at'] = datetime.now().isoformat()
                article_data['recovered']  = True

                with open(self.results_file, 'a', encoding='utf-8') as f:
                    f.write(json.dumps(article_data, ensure_ascii=False) + '\n')

                self.tracker.mark_parsed(url, html_path)
                self.tracker.mark_done(url)
                self.parsed_counter += 1
                print(f"  ♻️  Recovered: {url[:65]}…")

            except Exception as e:
                print(f"  ✗ Recovery failed: {url[:65]}…\n    {str(e)[:120]}")
                await self.log_failed_url(url, f"Recovery error: {str(e)}")

    # ── zip helpers ────────────────────────────

    async def zip_batch_async(self, batch_num: int):
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self._zip_batch_sync, batch_num)

    def _zip_batch_sync(self, batch_num: int):
        html_files = sorted(glob.glob(str(self.html_dir / "*.html")))
        if not html_files:
            return
        zip_filename = self.html_dir / f"batch_{batch_num:03d}.zip"
        with zipfile.ZipFile(zip_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for html_file in html_files:
                zipf.write(html_file, arcname=Path(html_file).name)
        for html_file in html_files:
            os.remove(html_file)
        print(f"📦 Zipped {len(html_files)} file(s) → {zip_filename.name}")

    # ── misc ───────────────────────────────────

    async def log_failed_url(self, url: str, error: str):
        self.failed_counter += 1
        with open(self.failed_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps({
                'url': url,
                'error': error,
                'timestamp': datetime.now().isoformat()
            }, ensure_ascii=False) + '\n')

    # ── main entry ─────────────────────────────

    async def run(self, urls: List[str]):
        signal.signal(signal.SIGINT,  self._handle_signal)
        signal.signal(signal.SIGTERM, self._handle_signal)

        # ── Resume: recover any scraped-but-unparsed HTML ──
        done_urls = self.tracker.load_done_urls()
        pending   = self.tracker.rebuild_pending(done_urls)
        await self.reparse_pending(pending)
        done_urls = self.tracker.load_done_urls()  # refresh after recovery

        remaining = [u for u in urls if u not in done_urls]
        skipped   = len(urls) - len(remaining)
        if skipped:
            print(f"⏭️  Skipping {skipped} already-completed URL(s)")

        print(f"\n🚀 Starting scraper: {len(remaining)} URL(s) remaining")
        print(f"   Mode    : sequential (1 browser at a time)")
        print(f"   Retries : {self.retries} per URL  |  retry delay: {self.retry_delay}s")
        print(f"   Wait    : {self.min_wait}–{self.max_wait}s between URLs")
        print(f"   Output  : {self.results_file}\n")

        if not remaining:
            print("✅ Nothing left to do!")
            return

        # Start one background parse worker
        parse_task = asyncio.create_task(self.parse_worker())

        # ── Sequential scrape loop ──────────────
        for url in remaining:
            if self._shutdown:
                print("🛑 Shutdown requested — stopping after this article.")
                break

            result = await self._scrape_one(url)
            if result:
                await self.scrape_queue.put(result)

        # Signal parser to finish up
        await self.scrape_queue.put(None)
        await parse_task

        # Zip any leftover HTML files
        if glob.glob(str(self.html_dir / "*.html")):
            await self.zip_batch_async(self.batch_counter)

        print(f"\n✅ Complete!")
        print(f"   ✓ Parsed  : {self.parsed_counter}")
        print(f"   ✗ Failed  : {self.failed_counter}")
        print(f"   📝 JSONL  : {self.results_file}")
        print(f"   ❌ Errors : {self.failed_file}")


# ─────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import pandas as pd

    dff = pd.read_csv("result/dff.csv")
    urls = dff['URL'].tolist()

    scraper = G1ArticleScraperOptimized(
        min_wait=6.0,
        max_wait=7.0,
        retries=3,       # retry each URL up to 3 times before giving up
        retry_delay=8.0, # wait 8s between retries to let Chrome fully close
    )

    asyncio.run(scraper.run(urls))