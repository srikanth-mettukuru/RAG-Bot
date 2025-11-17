from unstructured.chunking.title import chunk_by_title

def chunk_text(elements):
    """
    Chunks the preprocessed text elements based on titles using Unstructured.
    Returns a list of chunked elements.
    """
    chunked_elements = chunk_by_title(elements, combine_text_under_n_chars=50, max_chars=200)
    return chunked_elements


def fallback_chunk(text: str, max_chars=200):
    """
    Simple fixed-size chunker.
    Only used when title-based chunking returns nothing.
    """
    for i in range(0, len(text), max_chars):
        yield text[i:i + max_chars]
    