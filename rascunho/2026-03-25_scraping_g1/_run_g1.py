import asyncio
import json
import os
import zipfile
import glob
import gc
from datetime import datetime
from pathlib import Path
from bs4 import BeautifulSoup
from typing import List, Dict, Optional
import random
from collections import deque
from seleniumbase import SB
import time

def human_delay(min_s=1.5, max_s=4.0):
    """Random delay to mimic human reading/waiting."""
    time.sleep(random.uniform(min_s, max_s))


def scrape_article(url: str) -> str:
    with SB(
        uc=True,                  # Undetected Chrome mode
        headless=False,           # Visible browser (more human-like; set True for servers)
        locale_code="en-US",
        # Realistic window size (avoid perfect round numbers)
        window_size="1367,769",
    ) as sb:

        # 1. Open the page
        sb.uc_open_with_reconnect(url, reconnect_time=4)

        # 2. Human-like: slow scroll down the page
        #human_delay(2, 4)
        #sb.execute_script("window.scrollTo({ top: 300, behavior: 'smooth' });")
        #human_delay(1, 2)
        #sb.execute_script("window.scrollTo({ top: 700, behavior: 'smooth' });")
        #human_delay(1, 2)
        #sb.execute_script("window.scrollTo({ top: 1200, behavior: 'smooth' });")
        #human_delay(1.5, 3)

        # 3. Scroll back up slightly (humans rarely read perfectly top-to-bottom)
        sb.execute_script("window.scrollTo({ top: 900, behavior: 'smooth' });")
        human_delay(1, 2)

        # 4. Wait for full page load
        sb.wait_for_element("body", timeout=3)

        # 5. Grab and return the full rendered HTML
        html = sb.get_page_source()
        return html

def parse_g1_article(soup):
    article_data = {
        'title': None,
        'subtitle': None,
        'from_publication': None,
        'date_publication': None,
        'json_ld': None,
        'content': [],
        #'raw_content_tags': []
    }
    
    # Extract header info
    header = soup.find('div', class_='mc-article-header')
    if header:
        # Get title
        title_div = header.find('div', class_='title')
        if title_div:
            article_data['title'] = title_div.get_text(strip=True)
        
        # Get subtitle
        subtitle_div = header.find('div', class_=lambda x: x and 'subtitle' in x if x else False)
        if subtitle_div:
            article_data['subtitle'] = subtitle_div.get_text(strip=True)

        from_pub_data = header.find('p', class_=lambda x: x and 'content-publication-data__from' in x if x else False)
        if from_pub_data:
            article_data['from_publication'] = from_pub_data.get_text(strip=True)

        updated_pub_data = header.find('p', class_=lambda x: x and 'content-publication-data__updated' in x if x else False)
        if updated_pub_data:
            article_data['date_publication'] = updated_pub_data.get_text(strip=True)
    
    # Extract body info
    body = soup.find('div', class_='mc-article-body')
    if body:
        # Extract JSON-LD scripts
        json_ld_scripts = body.find_all('script', type='application/ld+json')
        if json_ld_scripts:
            import json
            article_data['json_ld'] = []
            for script in json_ld_scripts:
                try:
                    article_data['json_ld'].append(json.loads(script.string))
                except json.JSONDecodeError:
                    article_data['json_ld'].append(script.string)
        
        # Extract article body content
        article_body = body.find('article', attrs={'itemprop': 'articleBody'})
        if article_body:
            # Get all content tags in order: p, h1, h2, h3, h4, ul, li, blockquote
            content_tags = article_body.find_all([
                'p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6',
                'ul', 'li', 'ol',
                'blockquote',
                'figcaption',
                'table', 'tr', 'td', 'th',
                'section',
                'pre', 'code',
                'strong', 'em'
            ])
            
            for tag in content_tags:
                tag_data = {
                    'tag': tag.name,
                    'text': tag.get_text(strip=True),
                    'html': str(tag)
                }
                article_data['content'].append(tag_data)
                #article_data['raw_content_tags'].append((tag.name, tag.get_text(strip=True)))
    
    return article_data

class G1ArticleScraperOptimized:
    def __init__(
        self,
        output_dir: str = "data/scraped_articles",
        html_dir: str = "data/html_cache",
        max_concurrent: int = 3,  # Max scraping tasks at once
        parse_workers: int = 2,   # Parallel parsing tasks
        min_wait: float = 2.0,
        max_wait: float = 5.0,
        zip_interval: int = 100   # Zip every N files
    ):
        self.output_dir = Path(output_dir)
        self.html_dir = Path(html_dir)
        self.max_concurrent = max_concurrent
        self.parse_workers = parse_workers
        self.min_wait = min_wait
        self.max_wait = max_wait
        self.zip_interval = zip_interval
        
        self.html_counter = 0
        self.parsed_counter = 0
        self.failed_counter = 0
        self.batch_counter = 0
        
        self.results_file = self.output_dir / "articles.jsonl"
        self.failed_file = self.output_dir / "failed_urls.jsonl"
        
        # Queue for scraped HTML (URL, HTML, filepath)
        self.scrape_queue: asyncio.Queue = asyncio.Queue()
        
        # Create directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.html_dir.mkdir(parents=True, exist_ok=True)
    
    async def scrape_with_delay(self, url: str, semaphore: asyncio.Semaphore) -> Optional[tuple]:
        """Scrape a single URL with rate limiting via semaphore."""
        async with semaphore:
            try:
                delay = random.uniform(self.min_wait, self.max_wait)
                await asyncio.sleep(delay)
                
                loop = asyncio.get_event_loop()
                html = await loop.run_in_executor(None, scrape_article, url)
                
                # Save HTML immediately
                filename = f"{self.html_counter:06d}_article.html"
                filepath = self.html_dir / filename
                
                with open(filepath, 'w', encoding='utf-8') as f:
                    f.write(html)
                
                self.html_counter += 1
                print(f"✓ Scraped & saved: {url[:50]}... ({len(html)} chars)")
                
                # Zip every N files
                if self.html_counter % self.zip_interval == 0:
                    await self.zip_batch_async(self.batch_counter)
                    self.batch_counter += 1
                
                return (url, html, str(filepath))
                
            except Exception as e:
                print(f"✗ Scrape failed: {url[:50]}... {str(e)[:50]}")
                await self.log_failed_url(url, str(e))
                return None
    
    async def parse_worker(self):
        """Continuously parse HTML from queue and save JSONL."""
        while True:
            try:
                result = await asyncio.wait_for(self.scrape_queue.get(), timeout=5.0)
                
                if result is None:  # Sentinel value to stop
                    break
                
                url, html, filepath = result
                
                try:
                    soup = BeautifulSoup(html, 'html.parser')
                    article_data = parse_g1_article(soup)
                    article_data['url'] = url
                    article_data['html_path'] = filepath
                    article_data['scraped_at'] = datetime.now().isoformat()
                    
                    # Save to JSONL immediately
                    with open(self.results_file, 'a', encoding='utf-8') as f:
                        f.write(json.dumps(article_data, ensure_ascii=False) + '\n')
                    
                    self.parsed_counter += 1
                    print(f"📝 Parsed & saved: {url[:50]}... ({len(article_data['content'])} content blocks)")
                    
                except Exception as e:
                    print(f"✗ Parse failed: {url[:50]}... {str(e)[:50]}")
                    await self.log_failed_url(url, f"Parse error: {str(e)}")
                
                # Free memory
                del html
                gc.collect()
                
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                print(f"✗ Worker error: {str(e)}")
                break
    
    async def zip_batch_async(self, batch_num: int):
        """Async zip operation."""
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self._zip_batch_sync, batch_num)
    
    def _zip_batch_sync(self, batch_num: int):
        """Synchronous zip operation."""
        html_files = sorted(glob.glob(str(self.html_dir / "*.html")))
        
        if not html_files:
            return
        
        zip_filename = self.html_dir / f"batch_{batch_num:03d}.zip"
        with zipfile.ZipFile(zip_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for html_file in html_files:
                zipf.write(html_file, arcname=Path(html_file).name)
        
        # Delete original HTML files
        for html_file in html_files:
            os.remove(html_file)
        
        print(f"📦 Zipped {len(html_files)} files to {zip_filename.name}")
    
    async def log_failed_url(self, url: str, error: str):
        """Log failed URLs."""
        self.failed_counter += 1
        with open(self.failed_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps({
                'url': url,
                'error': error,
                'timestamp': datetime.now().isoformat()
            }, ensure_ascii=False) + '\n')
    
    async def run(self, urls: List[str]):
        """Main async runner with concurrent scraping & parsing."""
        print(f"🚀 Starting optimized scraper: {len(urls)} URLs")
        print(f"   Concurrent scrapers: {self.max_concurrent}")
        print(f"   Parse workers: {self.parse_workers}")
        print(f"   Output: {self.results_file}\n")
        
        # Create semaphore to limit concurrent scrapes
        semaphore = asyncio.Semaphore(self.max_concurrent)
        
        # Start parse workers
        parse_tasks = [
            asyncio.create_task(self.parse_worker())
            for _ in range(self.parse_workers)
        ]
        
        # Create scrape tasks
        scrape_tasks = [
            asyncio.create_task(self.scrape_with_delay(url, semaphore))
            for url in urls
        ]
        
        # Wait for all scrapes to complete
        scrape_results = await asyncio.gather(*scrape_tasks, return_exceptions=True)
        
        # Put results in parse queue
        for result in scrape_results:
            if result:
                await self.scrape_queue.put(result)
        
        # Signal workers to stop
        for _ in range(self.parse_workers):
            await self.scrape_queue.put(None)
        
        # Wait for all parsing to complete
        await asyncio.gather(*parse_tasks)
        
        # Final zip of remaining files
        remaining_html = glob.glob(str(self.html_dir / "*.html"))
        if remaining_html:
            await self.zip_batch_async(self.batch_counter)
        
        # Summary
        print(f"\n✅ Complete!")
        print(f"   ✓ Parsed: {self.parsed_counter}")
        print(f"   ✗ Failed: {self.failed_counter}")
        print(f"   📝 JSONL: {self.results_file}")
        print(f"   ❌ Failed URLs: {self.failed_file}")

if __name__ == "__main__":
    import pandas as pd
    import asyncio
    
    # Load URLs from CSV
    dff = pd.read_csv("result/dff.csv")

    # ============ USAGE ============
    urls = dff['URL'].tolist()

    scraper = G1ArticleScraperOptimized(
        max_concurrent=2,     # 2 concurrent scraping tasks
        parse_workers=2,      # 2 parsing workers
        min_wait=6.0,
        max_wait=7.0,
        # wait 6 or 7 seconds
    )
    
    asyncio.run(scraper.run(urls[:10]))