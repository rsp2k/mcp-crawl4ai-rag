import json

from mcp.server.fastmcp import Context

from ..crawl4ai_mcp import mcp


@mcp.tool()
async def get_available_sources(ctx: Context) -> str:
    """
    Get all available sources from the sources table.

    This tool returns a list of all unique sources (domains) that have been crawled and stored
    in the database, along with their summaries and statistics. This is useful for discovering
    what content is available for querying.

    Always use this tool before calling the RAG query or code example query tool
    with a specific source filter!

    Args:
        ctx: The MCP server provided context

    Returns:
        JSON string with the list of available sources and their details
    """
    try:
        # Get the Supabase client from the context
        supabase_client = ctx.request_context.lifespan_context.supabase_client

        # Query the sources table directly
        result = (
            supabase_client.from_("sources").select("*").order("source_id").execute()
        )

        # Format the sources with their details
        sources = []
        if result.data:
            for source in result.data:
                sources.append(
                    {
                        "source_id": source.get("source_id"),
                        "summary": source.get("summary"),
                        "total_words": source.get("total_words"),
                        "created_at": source.get("created_at"),
                        "updated_at": source.get("updated_at"),
                    }
                )

        return json.dumps(
            {"success": True, "sources": sources, "count": len(sources)}, indent=2
        )
    except Exception as e:
        return json.dumps({"success": False, "error": str(e)}, indent=2)
