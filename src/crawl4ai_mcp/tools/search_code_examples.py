import json
import os

from mcp.server.fastmcp import Context

from . import apply_source_filter
from ..transformers import rerank_results
from ..crawl4ai_mcp import mcp


@mcp.tool()
async def search_code_examples(
    ctx: Context, query: str, source_id: str = None, match_count: int = 5
) -> str:
    """
    Search for code examples relevant to the query.

    This tool searches the vector database for code examples relevant to the query and returns
    the matching examples with their summaries. Optionally filter by source_id.
    Get the source_id by using the get_available_sources tool before calling this search!

    Use the get_available_sources tool first to see what sources are available for filtering.

    Args:
        ctx: The MCP server provided context
        query: The search query
        source_id: Optional source ID to filter results (e.g., 'example.com')
        match_count: Maximum number of results to return (default: 5)

    Returns:
        JSON string with the search results
    """
    # Check if code example extraction is enabled
    extract_code_examples_enabled = os.getenv("USE_AGENTIC_RAG", "false") == "true"
    if not extract_code_examples_enabled:
        return json.dumps(
            {
                "success": False,
                "error": "Code example extraction is disabled. Perform a normal RAG search.",
            },
            indent=2,
        )

    try:
        # Get the Supabase client from the context
        supabase_client = ctx.request_context.lifespan_context.supabase_client

        # Check if hybrid search is enabled
        use_hybrid_search = os.getenv("USE_HYBRID_SEARCH", "false") == "true"

        # Prepare filter if source is provided and not empty
        filter_metadata = None
        if source_id and source_id.strip():
            filter_metadata = {"source": source_id}

        if use_hybrid_search:
            # Hybrid search: combine vector and keyword search

            from ..storage.supabase import (
                search_code_examples as search_code_examples_impl,
            )

            # 1. Get vector search results (get more to account for filtering)
            vector_results = search_code_examples_impl(
                client=supabase_client,
                query=query,
                match_count=match_count * 2,  # Get double to have room for filtering
                filter_metadata=filter_metadata,
            )

            # 2. Get keyword search results using ILIKE on both content and summary
            keyword_query = (
                supabase_client.from_("code_examples")
                .select("id, url, chunk_number, content, summary, metadata, source_id")
                .or_(f"content.ilike.%{query}%,summary.ilike.%{query}%")
            )

            if source_id and source_id.strip():
                keyword_query = keyword_query.eq("source_id", source_id)

            results = apply_source_filter(match_count, keyword_query, vector_results)

        else:
            # Standard vector search only
            from ..storage.supabase import (
                search_code_examples as search_code_examples_impl,
            )

            results = search_code_examples_impl(
                client=supabase_client,
                query=query,
                match_count=match_count,
                filter_metadata=filter_metadata,
            )

        # Apply reranking if enabled
        use_reranking = os.getenv("USE_RERANKING", "false") == "true"
        if use_reranking and ctx.request_context.lifespan_context.reranking_model:
            results = rerank_results(
                ctx.request_context.lifespan_context.reranking_model,
                query,
                results,
                content_key="content",
            )

        # Format the results
        formatted_results = []
        for result in results:
            formatted_result = {
                "url": result.get("url"),
                "code": result.get("content"),
                "summary": result.get("summary"),
                "metadata": result.get("metadata"),
                "source_id": result.get("source_id"),
                "similarity": result.get("similarity"),
            }
            # Include rerank score if available
            if "rerank_score" in result:
                formatted_result["rerank_score"] = result["rerank_score"]
            formatted_results.append(formatted_result)

        return json.dumps(
            {
                "success": True,
                "query": query,
                "source_filter": source_id,
                "search_mode": "hybrid" if use_hybrid_search else "vector",
                "reranking_applied": use_reranking
                and ctx.request_context.lifespan_context.reranking_model is not None,
                "results": formatted_results,
                "count": len(formatted_results),
            },
            indent=2,
        )
    except Exception as e:
        return json.dumps({"success": False, "query": query, "error": str(e)}, indent=2)
