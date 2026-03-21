import os
import sys
import psutil
import asyncio
import requests
import hashlib
from xml.etree import ElementTree


__location__ = os.path.dirname(os.path.abspath(__file__))
__output__ = os.path.join(__location__, "output")

# Append parent directory to system path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)


from typing import List
from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig, CacheMode


async def crawl_terveyskirjasto_pages(urls: List[str], max_concurrent: int = 3):
    print("\n=== Parallel Crawling of Terveyskirjasto Pages with Browser Reuse + Memory Check ===")

    # Keep track of peak memory usage across all tasks
    peak_memory = 0
    process = psutil.Process(os.getpid())

    def log_memory(prefix: str = ""):
        nonlocal peak_memory
        current_mem = process.memory_info().rss  # in bytes
        if current_mem > peak_memory:
            peak_memory = current_mem
        print(f"{prefix} Current Memory: {current_mem // (1024 * 1024)} MB, Peak: {peak_memory // (1024 * 1024)} MB")

    # Minimal browser config
    browser_config = BrowserConfig(
        headless=True,
        verbose=False,
        extra_args=["--disable-gpu", "--disable-dev-shm-usage", "--no-sandbox"],
    )
    crawl_config = CrawlerRunConfig(cache_mode=CacheMode.BYPASS)

    # Create the crawler instance
    crawler = AsyncWebCrawler(config=browser_config)
    await crawler.start()

    try:
        # Chunk the URLs in batches of 'max_concurrent'
        success_count = 0
        fail_count = 0
        for i in range(0, len(urls), max_concurrent):
            batch = urls[i : i + max_concurrent]
            tasks = []

            for j, url in enumerate(batch):
                # Unique session_id per concurrent sub-task
                session_id = f"terveys_session_{i + j}"
                task = crawler.arun(url=url, config=crawl_config, session_id=session_id)
                tasks.append(task)

            # Check memory usage before and after launching tasks
            log_memory(prefix=f"Before batch {i//max_concurrent + 1}: ")
            results = await asyncio.gather(*tasks, return_exceptions=True)
            log_memory(prefix=f"After batch {i//max_concurrent + 1}: ")

            # Evaluate results
            for url, result in zip(batch, results):
                if isinstance(result, Exception):
                    print(f"Error crawling {url}: {result}")
                    fail_count += 1
                elif result.success:
                    save_markdown(url, result.markdown)
                    success_count += 1
                else:
                    fail_count += 1

        print(f"\nSummary:")
        print(f"  - Successfully crawled: {success_count}")
        print(f"  - Failed: {fail_count}")

    finally:
        print("\nClosing crawler...")
        await crawler.close()
        # Final memory log
        log_memory(prefix="Final: ")
        print(f"\nPeak memory usage (MB): {peak_memory // (1024 * 1024)}")


def get_terveyskirjasto_urls():
    """
    Retrieves article URLs from Terveyskirjasto's sitemap (https://www.terveyskirjasto.fi/sitemap.xml)

    Returns:
        List[str]: List of Terveyskirjasto article URLs
    """
    sitemap_url = "https://www.terveyskirjasto.fi/articles.xml"
    try:
        response = requests.get(sitemap_url)
        response.raise_for_status()
        
        # Parse the XML 
        root = ElementTree.fromstring(response.content)
        
        # Extract all URLs from the sitemap
        namespace = {'ns': 'http://www.sitemaps.org/schemas/sitemap/0.9'}
        urls = [loc.text for loc in root.findall('.//ns:loc', namespace)]
        
        return urls
    except Exception as e:
        print(f"Error fetching sitemap: {e}")
        return []
    

def save_markdown(url: str, markdown: str):
    os.makedirs("terveys_markdown", exist_ok=True)
    filename = hashlib.md5(url.encode()).hexdigest() + ".md"
    with open(os.path.join("terveys_markdown", filename), "w", encoding="utf-8") as f:
        f.write(markdown)


async def main():
    urls = get_terveyskirjasto_urls()
    if urls:
        print(f"Found {len(urls)} URLs to crawl")
        await crawl_terveyskirjasto_pages(urls, max_concurrent=100)
    else:
        print("No URLs found to crawl")


if __name__ == "__main__":
    asyncio.run(main())