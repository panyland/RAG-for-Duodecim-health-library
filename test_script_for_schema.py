import asyncio
from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig, CacheMode

async def inspect_extracted_content(url):
    browser_config = BrowserConfig(
        headless=True,
        verbose=False,
        extra_args=["--disable-gpu", "--disable-dev-shm-usage", "--no-sandbox"],
    )
    crawl_config = CrawlerRunConfig(cache_mode=CacheMode.BYPASS)

    crawler = AsyncWebCrawler(config=browser_config)
    await crawler.start()

    try:
        result = await crawler.arun(url=url, config=crawl_config, session_id="debug_session")
        if result.success:
            print("\n--- Extracted Content ---")
            print(result.markdown)  # Print only the first 1000 chars
        else:
            print("Crawling failed.")
    finally:
        await crawler.close()

# Replace with an actual Terveyskirjasto article URL
test_url = "https://www.terveyskirjasto.fi/dlk00086"
asyncio.run(inspect_extracted_content(test_url))
