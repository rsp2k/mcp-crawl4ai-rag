import json
import os

from mcp.server.fastmcp import Context

from . import apply_source_filter
from ..crawl4ai_mcp import mcp
from ..storage.supabase import search_documents
from ..transformers import rerank_results


@mcp.tool()
async def perform_rag_query(
    ctx: Context, query: str, source: str = None, match_count: int = 5
) -> str:
    """
    Perform a RAG (Retrieval Augmented Generation) query on the stored content.

    This tool searches the vector database for content relevant to the query and returns
    the matching documents. Optionally filter by source domain.
    Get the source by using the get_available_sources tool before calling this search!

    Args:
        ctx: The MCP server provided context
        query: The search query
        source: Optional source domain to filter results (e.g., 'example.com')
        match_count: Maximum number of results to return (default: 5)

    Returns:
        JSON string with the search results
    """
    try:
        # Get the Supabase client from the context
        supabase_client = ctx.request_context.lifespan_context.supabase_client

        # Check if hybrid search is enabled
        use_hybrid_search = os.getenv("USE_HYBRID_SEARCH", "false") == "true"

        # Prepare filter if source is provided and not empty
        filter_metadata = None
        if source and source.strip():
            filter_metadata = {"source": source}

        if use_hybrid_search:
            # Hybrid search: combine vector and keyword search

            # 1. Get vector search results (get more to account for filtering)
            vector_results = search_documents(
                client=supabase_client,
                query=query,
                match_count=match_count * 2,  # Get double to have room for filtering
                filter_metadata=filter_metadata,
            )

            # 2. Get keyword search results using ILIKE
            keyword_query = (
                supabase_client.from_("crawled_pages")
                .select("id, url, chunk_number, content, metadata, source_id")
                .ilike("content", f"%{query}%")
            )
            if source and source.strip():

                keyword_query = keyword_query.eq("source_id", source)

            results = apply_source_filter(match_count, keyword_query, vector_results)

        else:
            # Standard vector search only
            results = search_documents(
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
                "content": result.get("content"),
                "metadata": result.get("metadata"),
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
                "source_filter": source,
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
