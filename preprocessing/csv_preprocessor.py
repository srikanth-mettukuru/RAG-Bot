from unstructured.partition.csv import partition_csv

def preprocess_csv(path: str):
    """
    Reads and preprocesses a .csv file using Unstructured.
    Returns a list of elements where each element is a dictionary with metadata and text.
    """
    elements = partition_csv(filename=path)
    return elements