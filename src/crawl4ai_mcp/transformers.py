from typing import List, Dict, Any

from sentence_transformers import CrossEncoder


def rerank_results(
    model: CrossEncoder,
    query: str,
    results: List[Dict[str, Any]],
    content_key: str = "content",
) -> List[Dict[str, Any]]:
    """
    Rerank search results using a cross-encoder model.

    Args:
        model: The cross-encoder model to use for reranking
        query: The search query
        results: List of search results
        content_key: The key in each result dict that contains the text content

    Returns:
        Reranked list of results
    """
    if not model or not results:
        return results

    try:
        # Extract content from results
        texts = [result.get(content_key, "") for result in results]

        # Create pairs of [query, document] for the cross-encoder
        pairs = [[query, text] for text in texts]

        # Get relevance scores from the cross-encoder
        scores = model.predict(pairs)

        # Add scores to results and sort by score (descending)
        for i, result in enumerate(results):
            result["rerank_score"] = float(scores[i])

        # Sort by rerank score
        reranked = sorted(results, key=lambda x: x.get("rerank_score", 0), reverse=True)

        return reranked
    except Exception as e:
        print(f"Error during reranking: {e}")
        return results
