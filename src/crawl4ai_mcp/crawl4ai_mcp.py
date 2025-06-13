"""
MCP server for web crawling with Crawl4AI.

This server provides tools to crawl websites using Crawl4AI, automatically detecting
the appropriate crawl method based on URL type (sitemap, txt file, or regular webpage).
"""

from typing import Optional

import asyncio
import os

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP, Context
from sentence_transformers import CrossEncoder
import requests
from supabase import Client


from crawl4ai import (
    AsyncWebCrawler,
    BrowserConfig,
    CrawlerRunConfig,
    CacheMode,
    MemoryAdaptiveDispatcher,
)

from .storage.supabase import get_supabase_client

# Load environment variables from the project root .env file
project_root = Path(__file__).resolve().parent.parent
dotenv_path = project_root / ".env"

# Force override of existing environment variables
load_dotenv(dotenv_path, override=True)


# Create a dataclass for our application context
@dataclass
class Crawl4AIContext:
    """Context for the Crawl4AI MCP server."""

    crawler: AsyncWebCrawler
    supabase_client: Client
    reranking_model: Optional[CrossEncoder] = None


@asynccontextmanager
async def crawl4ai_lifespan(server: FastMCP) -> AsyncIterator[Crawl4AIContext]:
    """
    Manages the Crawl4AI client lifecycle.

    Args:
        server: The FastMCP server instance

    Yields:
        Crawl4AIContext: The context containing the Crawl4AI crawler and Supabase client
    """
    # Create browser configuration
    browser_config = BrowserConfig(headless=True, verbose=False)

    # Initialize the crawler
    crawler = AsyncWebCrawler(config=browser_config)
    await crawler.__aenter__()

    # Initialize Supabase client
    supabase_client = get_supabase_client()

    # Initialize cross-encoder model for reranking if enabled
    reranking_model = None
    if os.getenv("USE_RERANKING", "false") == "true":
        try:
            reranking_model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
        except Exception as e:
            print(f"Failed to load reranking model: {e}")
            reranking_model = None

    try:
        yield Crawl4AIContext(
            crawler=crawler,
            supabase_client=supabase_client,
            reranking_model=reranking_model,
        )
    finally:
        # Clean up the crawler
        await crawler.__aexit__(None, None, None)


# Initialize FastMCP server
mcp = FastMCP(
    "mcp-crawl4ai-rag",
    description="MCP server for RAG and web crawling with Crawl4AI",
    lifespan=crawl4ai_lifespan,
    host=os.getenv("HOST", "0.0.0.0"),
    port=os.getenv("PORT", "8051"),
)


async def arun_server():
    transport = os.getenv("TRANSPORT", "sse")
    if transport == "sse":
        # Run the MCP server with sse transport
        await mcp.run_sse_async()
    else:
        # Run the MCP server with stdio transport
        await mcp.run_stdio_async()


def main():
    asyncio.run(arun_server())


if __name__ == "__main__":
    main()
