import os
import time
from typing import List, Tuple, Any

import openai

# Load OpenAI API key for embeddings
openai.api_key = os.getenv("OPENAI_API_KEY")
openai_model = os.getenv("OPENAI_MODEL", "text-embedding-3-small")


def create_embeddings_batch(texts: List[str]) -> list[Any] | None:
    """
    Create embeddings for multiple texts in a single API call.

    Args:
        texts: List of texts to create embeddings for

    Returns:
        List of embeddings (each embedding is a list of floats)
    """
    if not texts:
        return []

    max_retries = 3
    retry_delay = 1.0  # Start with 1 second delay

    for retry in range(max_retries):
        try:
            response = openai.embeddings.create(model=openai_model, input=texts)
            return [item.embedding for item in response.data]
        except Exception as e:
            if retry < max_retries - 1:
                print(
                    f"Error creating batch embeddings (attempt {retry + 1}/{max_retries}): {e}"
                )
                print(f"Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
                retry_delay *= 2  # Exponential backoff
            else:
                print(
                    f"Failed to create batch embeddings after {max_retries} attempts: {e}"
                )
                # Try creating embeddings one by one as fallback
                print("Attempting to create embeddings individually...")
                embeddings = []
                successful_count = 0

                for i, text in enumerate(texts):
                    try:
                        individual_response = openai.embeddings.create(
                            model="text-embedding-3-small", input=[text]
                        )
                        embeddings.append(individual_response.data[0].embedding)
                        successful_count += 1
                    except Exception as individual_error:
                        print(
                            f"Failed to create embedding for text {i}: {individual_error}"
                        )
                        # Add zero embedding as fallback
                        embeddings.append([0.0] * 1536)

                print(
                    f"Successfully created {successful_count}/{len(texts)} embeddings individually"
                )
                return embeddings


def create_embedding(text: str) -> List[float]:
    """
    Create an embedding for a single text using OpenAI's API.

    Args:
        text: Text to create an embedding for

    Returns:
        List of floats representing the embedding
    """
    try:
        embeddings = create_embeddings_batch([text])
        return embeddings[0] if embeddings else [0.0] * 1536
    except Exception as e:
        print(f"Error creating embedding: {e}")
        # Return empty embedding if there's an error
        return [0.0] * 1536


def generate_contextual_embedding(full_document: str, chunk: str) -> Tuple[str, bool]:
    """
    Generate contextual information for a chunk within a document to improve retrieval.

    Args:
        full_document: The complete document text
        chunk: The specific chunk of text to generate context for

    Returns:
        Tuple containing:
        - The contextual text that situates the chunk within the document
        - Boolean indicating if contextual embedding was performed
    """
    model_choice = os.getenv("MODEL_CHOICE")

    try:
        # Create the prompt for generating contextual information
        prompt = f"""<document> 
{full_document[:25000]} 
</document>
Here is the chunk we want to situate within the whole document 
<chunk> 
{chunk}
</chunk> 
Please give a short succinct context to situate this chunk within the overall document for the purposes of improving search retrieval of the chunk. Answer only with the succinct context and nothing else."""

        # Call the OpenAI API to generate contextual information
        response = openai.chat.completions.create(
            model=model_choice,
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant that provides concise contextual information.",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.3,
            max_tokens=200,
        )

        # Extract the generated context
        context = response.choices[0].message.content.strip()

        # Combine the context with the original chunk
        contextual_text = f"{context}\n---\n{chunk}"

        return contextual_text, True

    except Exception as e:
        print(
            f"Error generating contextual embedding: {e}. Using original chunk instead."
        )
        return chunk, False


def generate_code_example_summary(
    code: str, context_before: str, context_after: str
) -> str:
    """
    Generate a summary for a code example using its surrounding context.

    Args:
        code: The code example
        context_before: Context before the code
        context_after: Context after the code

    Returns:
        A summary of what the code example demonstrates
    """
    model_choice = os.getenv("MODEL_CHOICE")

    # Create the prompt
    prompt = f"""<context_before>
{context_before[-500:] if len(context_before) > 500 else context_before}
</context_before>

<code_example>
{code[:1500] if len(code) > 1500 else code}
</code_example>

<context_after>
{context_after[:500] if len(context_after) > 500 else context_after}
</context_after>

Based on the code example and its surrounding context, provide a concise summary (2-3 sentences) that describes what this code example demonstrates and its purpose. Focus on the practical application and key concepts illustrated.
"""

    try:
        response = openai.chat.completions.create(
            model=model_choice,
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant that provides concise code example summaries.",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.3,
            max_tokens=100,
        )

        return response.choices[0].message.content.strip()

    except Exception as e:
        print(f"Error generating code example summary: {e}")
        return "Code example for demonstration purposes."


def extract_source_summary(source_id: str, content: str, max_length: int = 500) -> str:
    """
    Extract a summary for a source from its content using an LLM.

    This function uses the OpenAI API to generate a concise summary of the source content.

    Args:
        source_id: The source ID (domain)
        content: The content to extract a summary from
        max_length: Maximum length of the summary

    Returns:
        A summary string
    """
    # Default summary if we can't extract anything meaningful
    default_summary = f"Content from {source_id}"

    if not content or len(content.strip()) == 0:
        return default_summary

    # Get the model choice from environment variables
    model_choice = os.getenv("MODEL_CHOICE")

    # Limit content length to avoid token limits
    truncated_content = content[:25000] if len(content) > 25000 else content

    # Create the prompt for generating the summary
    prompt = f"""<source_content>
{truncated_content}
</source_content>

The above content is from the documentation for '{source_id}'. Please provide a concise summary (3-5 sentences) that describes what this library/tool/framework is about. The summary should help understand what the library/tool/framework accomplishes and the purpose.
"""

    try:
        # Call the OpenAI API to generate the summary
        response = openai.chat.completions.create(
            model=model_choice,
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant that provides concise library/tool/framework summaries.",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.3,
            max_tokens=150,
        )

        # Extract the generated summary
        summary = response.choices[0].message.content.strip()

        # Ensure the summary is not too long
        if len(summary) > max_length:
            summary = summary[:max_length] + "..."

        return summary

    except Exception as e:
        print(
            f"Error generating summary with LLM for {source_id}: {e}. Using default summary."
        )
        return default_summary
