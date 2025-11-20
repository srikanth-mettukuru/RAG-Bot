from unstructured.partition.text import partition_text

def preprocess_txt(path: str):
    """
    Reads and preprocesses a .txt file using Unstructured.
    Returns a list of elements where each element is a dictionary with metadata and text.
    """
    elements = partition_text(filename=path)
    return elements