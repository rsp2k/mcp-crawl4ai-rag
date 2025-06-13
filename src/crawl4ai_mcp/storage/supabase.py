import os
import concurrent.futures
import time
from typing import List, Dict, Any, Optional, Tuple

from urllib.parse import urlparse

from ..embeddings import (
    generate_contextual_embedding,
    create_embeddings_batch,
    create_embedding,
)

from supabase import create_client, Client


def get_supabase_client() -> Client:
    """
    Get a Supabase client with the URL and key from environment variables.

    Returns:
        Supabase client instance
    """
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_SERVICE_KEY")

    if not url or not key:
        raise ValueError(
            "SUPABASE_URL and SUPABASE_SERVICE_KEY must be set in environment variables"
        )

    return create_client(url, key)


def process_chunk_with_context(args):
    """
    Process a single chunk with contextual embedding.
    This function is designed to be used with concurrent.futures.

    Args:
        args: Tuple containing (url, content, full_document)

    Returns:
        Tuple containing:
        - The contextual text that situates the chunk within the document
        - Boolean indicating if contextual embedding was performed
    """
    url, content, full_document = args
    return generate_contextual_embedding(full_document, content)


def add_documents_to_supabase(
    client: Client,
    urls: List[str],
    chunk_numbers: List[int],
    contents: List[str],
    metadatas: List[Dict[str, Any]],
    url_to_full_document: Dict[str, str],
    batch_size: int = 20,
) -> None:
    """
    Add documents to the Supabase crawled_pages table in batches.
    Deletes existing records with the same URLs before insertion to prevent duplicates.

    Args:
        client: Supabase client
        urls: List of URLs
        chunk_numbers: List of chunk numbers
        contents: List of document contents
        metadatas: List of document metadata
        url_to_full_document: Dictionary mapping URLs to their full document content
        batch_size: Size of each batch for insertion
    """
    # Get unique URLs to delete existing records
    unique_urls = list(set(urls))

    # Delete existing records for these URLs in a single operation
    try:
        if unique_urls:
            # Use the .in_() filter to delete all records with matching URLs
            client.table("crawled_pages").delete().in_("url", unique_urls).execute()
    except Exception as e:
        print(f"Batch delete failed: {e}. Trying one-by-one deletion as fallback.")
        # Fallback: delete records one by one
        for url in unique_urls:
            try:
                client.table("crawled_pages").delete().eq("url", url).execute()
            except Exception as inner_e:
                print(f"Error deleting record for URL {url}: {inner_e}")
                # Continue with the next URL even if one fails

    # Check if MODEL_CHOICE is set for contextual embeddings
    use_contextual_embeddings = (
        os.getenv("USE_CONTEXTUAL_EMBEDDINGS", "false") == "true"
    )
    print(f"\n\nUse contextual embeddings: {use_contextual_embeddings}\n\n")

    # Process in batches to avoid memory issues
    for i in range(0, len(contents), batch_size):
        batch_end = min(i + batch_size, len(contents))

        # Get batch slices
        batch_urls = urls[i:batch_end]
        batch_chunk_numbers = chunk_numbers[i:batch_end]
        batch_contents = contents[i:batch_end]
        batch_metadatas = metadatas[i:batch_end]

        # Apply contextual embedding to each chunk if MODEL_CHOICE is set
        if use_contextual_embeddings:
            # Prepare arguments for parallel processing
            process_args = []
            for j, content in enumerate(batch_contents):
                url = batch_urls[j]
                full_document = url_to_full_document.get(url, "")
                process_args.append((url, content, full_document))

            # Process in parallel using ThreadPoolExecutor
            contextual_contents = []
            with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
                # Submit all tasks and collect results
                future_to_idx = {
                    executor.submit(process_chunk_with_context, arg): idx
                    for idx, arg in enumerate(process_args)
                }

                # Process results as they complete
                for future in concurrent.futures.as_completed(future_to_idx):
                    idx = future_to_idx[future]
                    try:
                        result, success = future.result()
                        contextual_contents.append(result)
                        if success:
                            batch_metadatas[idx]["contextual_embedding"] = True
                    except Exception as e:
                        print(f"Error processing chunk {idx}: {e}")
                        # Use original content as fallback
                        contextual_contents.append(batch_contents[idx])

            # Sort results back into original order if needed
            if len(contextual_contents) != len(batch_contents):
                print(
                    f"Warning: Expected {len(batch_contents)} results but got {len(contextual_contents)}"
                )
                # Use original contents as fallback
                contextual_contents = batch_contents
        else:
            # If not using contextual embeddings, use original contents
            contextual_contents = batch_contents

        # Create embeddings for the entire batch at once
        batch_embeddings = create_embeddings_batch(contextual_contents)

        batch_data = []
        for j in range(len(contextual_contents)):
            # Extract metadata fields
            chunk_size = len(contextual_contents[j])

            # Extract source_id from URL
            parsed_url = urlparse(batch_urls[j])
            source_id = parsed_url.netloc or parsed_url.path

            # Prepare data for insertion
            data = {
                "url": batch_urls[j],
                "chunk_number": batch_chunk_numbers[j],
                "content": contextual_contents[j],  # Store original content
                "metadata": {"chunk_size": chunk_size, **batch_metadatas[j]},
                "source_id": source_id,  # Add source_id field
                "embedding": batch_embeddings[
                    j
                ],  # Use embedding from contextual content
            }

            batch_data.append(data)

        # Insert batch into Supabase with retry logic
        max_retries = 3
        retry_delay = 1.0  # Start with 1 second delay

        for retry in range(max_retries):
            try:
                client.table("crawled_pages").insert(batch_data).execute()
                # Success - break out of retry loop
                break
            except Exception as e:
                if retry < max_retries - 1:
                    print(
                        f"Error inserting batch into Supabase (attempt {retry + 1}/{max_retries}): {e}"
                    )
                    print(f"Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff
                else:
                    # Final attempt failed
                    print(f"Failed to insert batch after {max_retries} attempts: {e}")
                    # Optionally, try inserting records one by one as a last resort
                    print("Attempting to insert records individually...")
                    successful_inserts = 0
                    for record in batch_data:
                        try:
                            client.table("crawled_pages").insert(record).execute()
                            successful_inserts += 1
                        except Exception as individual_error:
                            print(
                                f"Failed to insert individual record for URL {record['url']}: {individual_error}"
                            )

                    if successful_inserts > 0:
                        print(
                            f"Successfully inserted {successful_inserts}/{len(batch_data)} records individually"
                        )


def search_documents(
    client: Client,
    query: str,
    match_count: int = 10,
    filter_metadata: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """
    Search for documents in Supabase using vector similarity.

    Args:
        client: Supabase client
        query: Query text
        match_count: Maximum number of results to return
        filter_metadata: Optional metadata filter

    Returns:
        List of matching documents
    """
    # Create embedding for the query
    query_embedding = create_embedding(query)

    # Execute the search using the match_crawled_pages function
    try:
        # Only include filter parameter if filter_metadata is provided and not empty
        params = {"query_embedding": query_embedding, "match_count": match_count}

        # Only add the filter if it's actually provided and not empty
        if filter_metadata:
            params["filter"] = (
                filter_metadata  # Pass the dictionary directly, not JSON-encoded
            )

        result = client.rpc("match_crawled_pages", params).execute()

        return result.data
    except Exception as e:
        print(f"Error searching documents: {e}")
        return []


def add_code_examples_to_supabase(
    client: Client,
    urls: List[str],
    chunk_numbers: List[int],
    code_examples: List[str],
    summaries: List[str],
    metadatas: List[Dict[str, Any]],
    batch_size: int = 20,
):
    """
    Add code examples to the Supabase code_examples table in batches.

    Args:
        client: Supabase client
        urls: List of URLs
        chunk_numbers: List of chunk numbers
        code_examples: List of code example contents
        summaries: List of code example summaries
        metadatas: List of metadata dictionaries
        batch_size: Size of each batch for insertion
    """
    if not urls:
        return

    # Delete existing records for these URLs
    unique_urls = list(set(urls))
    for url in unique_urls:
        try:
            client.table("code_examples").delete().eq("url", url).execute()
        except Exception as e:
            print(f"Error deleting existing code examples for {url}: {e}")

    # Process in batches
    total_items = len(urls)
    for i in range(0, total_items, batch_size):
        batch_end = min(i + batch_size, total_items)
        batch_texts = []

        # Create combined texts for embedding (code + summary)
        for j in range(i, batch_end):
            combined_text = f"{code_examples[j]}\n\nSummary: {summaries[j]}"
            batch_texts.append(combined_text)

        # Create embeddings for the batch
        embeddings = create_embeddings_batch(batch_texts)

        # Check if embeddings are valid (not all zeros)
        valid_embeddings = []
        for embedding in embeddings:
            if embedding and not all(v == 0.0 for v in embedding):
                valid_embeddings.append(embedding)
            else:
                print(
                    f"Warning: Zero or invalid embedding detected, creating new one..."
                )
                # Try to create a single embedding as fallback
                single_embedding = create_embedding(batch_texts[len(valid_embeddings)])
                valid_embeddings.append(single_embedding)

        # Prepare batch data
        batch_data = []
        for j, embedding in enumerate(valid_embeddings):
            idx = i + j

            # Extract source_id from URL
            parsed_url = urlparse(urls[idx])
            source_id = parsed_url.netloc or parsed_url.path

            batch_data.append(
                {
                    "url": urls[idx],
                    "chunk_number": chunk_numbers[idx],
                    "content": code_examples[idx],
                    "summary": summaries[idx],
                    "metadata": metadatas[idx],  # Store as JSON object, not string
                    "source_id": source_id,
                    "embedding": embedding,
                }
            )

        # Insert batch into Supabase with retry logic
        max_retries = 3
        retry_delay = 1.0  # Start with 1 second delay

        for retry in range(max_retries):
            try:
                client.table("code_examples").insert(batch_data).execute()
                # Success - break out of retry loop
                break
            except Exception as e:
                if retry < max_retries - 1:
                    print(
                        f"Error inserting batch into Supabase (attempt {retry + 1}/{max_retries}): {e}"
                    )
                    print(f"Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff
                else:
                    # Final attempt failed
                    print(f"Failed to insert batch after {max_retries} attempts: {e}")
                    # Optionally, try inserting records one by one as a last resort
                    print("Attempting to insert records individually...")
                    successful_inserts = 0
                    for record in batch_data:
                        try:
                            client.table("code_examples").insert(record).execute()
                            successful_inserts += 1
                        except Exception as individual_error:
                            print(
                                f"Failed to insert individual record for URL {record['url']}: {individual_error}"
                            )

                    if successful_inserts > 0:
                        print(
                            f"Successfully inserted {successful_inserts}/{len(batch_data)} records individually"
                        )
        print(
            f"Inserted batch {i//batch_size + 1} of {(total_items + batch_size - 1)//batch_size} code examples"
        )


def update_source_info(client: Client, source_id: str, summary: str, word_count: int):
    """
    Update or insert source information in the sources table.

    Args:
        client: Supabase client
        source_id: The source ID (domain)
        summary: Summary of the source
        word_count: Total word count for the source
    """
    try:
        # Try to update existing source
        result = (
            client.table("sources")
            .update(
                {
                    "summary": summary,
                    "total_word_count": word_count,
                    "updated_at": "now()",
                }
            )
            .eq("source_id", source_id)
            .execute()
        )

        # If no rows were updated, insert new source
        if not result.data:
            client.table("sources").insert(
                {
                    "source_id": source_id,
                    "summary": summary,
                    "total_word_count": word_count,
                }
            ).execute()
            print(f"Created new source: {source_id}")
        else:
            print(f"Updated source: {source_id}")

    except Exception as e:
        print(f"Error updating source {source_id}: {e}")


def search_code_examples(
    client: Client,
    query: str,
    match_count: int = 10,
    filter_metadata: Optional[Dict[str, Any]] = None,
    source_id: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Search for code examples in Supabase using vector similarity.

    Args:
        client: Supabase client
        query: Query text
        match_count: Maximum number of results to return
        filter_metadata: Optional metadata filter
        source_id: Optional source ID to filter results

    Returns:
        List of matching code examples
    """
    # Create a more descriptive query for better embedding match
    # Since code examples are embedded with their summaries, we should make the query more descriptive
    enhanced_query = (
        f"Code example for {query}\n\nSummary: Example code showing {query}"
    )

    # Create embedding for the enhanced query
    query_embedding = create_embedding(enhanced_query)

    # Execute the search using the match_code_examples function
    try:
        # Only include filter parameter if filter_metadata is provided and not empty
        params = {"query_embedding": query_embedding, "match_count": match_count}

        # Only add the filter if it's actually provided and not empty
        if filter_metadata:
            params["filter"] = filter_metadata

        # Add source filter if provided
        if source_id:
            params["source_filter"] = source_id

        result = client.rpc("match_code_examples", params).execute()

        return result.data
    except Exception as e:
        print(f"Error searching code examples: {e}")
        return []
