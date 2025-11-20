from unstructured.partition.pptx import partition_pptx

def preprocess_pptx(path: str):
    """
    Reads and preprocesses a .pptx file using Unstructured.
    Returns a list of elements where each element is a dictionary with metadata and text.
    """
    elements = partition_pptx(filename=path)
    return elements