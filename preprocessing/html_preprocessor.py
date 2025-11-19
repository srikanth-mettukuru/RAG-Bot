from unstructured.partition.html import partition_html

def preprocess_html(path: str):
    """
    Reads and preprocesses a .html file using Unstructured.
    Returns a list of elements where each element is a dictionary with metadata and text.
    """
    elements = partition_html(filename=path)
    return elements