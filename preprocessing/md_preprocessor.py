from unstructured.partition.md import partition_md

def preprocess_md(path: str):
    """
    Reads and preprocesses a .md file using Unstructured.
    Returns a list of elements where each element is a dictionary with metadata and text.
    """
    elements = partition_md(filename=path)
    return elements