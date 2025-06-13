def apply_source_filter(match_count, keyword_query, vector_results):
    # Execute keyword search
    keyword_response = keyword_query.limit(match_count * 2).execute()
    keyword_results = keyword_response.data if keyword_response.data else []

    # 3. Combine results with preference for items appearing in both
    seen_ids = set()
    combined_results = []

    # First, add items that appear in both searches (these are the best matches)
    vector_ids = {r.get("id") for r in vector_results if r.get("id")}
    for kr in keyword_results:
        if kr["id"] in vector_ids and kr["id"] not in seen_ids:
            # Find the vector result to get similarity score
            for vr in vector_results:
                if vr.get("id") == kr["id"]:
                    # Boost similarity score for items in both results
                    vr["similarity"] = min(1.0, vr.get("similarity", 0) * 1.2)
                    combined_results.append(vr)
                    seen_ids.add(kr["id"])
                    break

    # Then add remaining vector results (semantic matches without exact keyword)
    for vr in vector_results:
        if (
            vr.get("id")
            and vr["id"] not in seen_ids
            and len(combined_results) < match_count
        ):
            combined_results.append(vr)
            seen_ids.add(vr["id"])

    return combined_results[:match_count]
