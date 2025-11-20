from unstructured.partition.docx import partition_docx

def preprocess_docx(path: str):
    """
    Reads and preprocesses a .docx file using Unstructured.
    Returns a list of elements where each element is a dictionary with metadata and text.
    """
    elements = partition_docx(filename=path)
    return elements