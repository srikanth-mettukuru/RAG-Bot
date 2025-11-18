from unstructured.chunking.title import chunk_by_title
from langchain_community.vectorstores import Chroma

def chunk_text(elements):
    """
    Chunks the preprocessed text elements based on titles using Unstructured.
    Returns a list of chunked elements.
    """
    chunked_elements = chunk_by_title(elements, combine_text_under_n_chars=50, max_characters=500,overlap=50)
    return chunked_elements


def fallback_chunk(text: str, max_chars=500):
    """
    Simple fixed-size chunker.
    Only used when title-based chunking returns nothing.
    """
    for i in range(0, len(text), max_chars):
        yield text[i:i + max_chars]


class RAGPipeline:

    def __init__(self, embedding_model, persist_directory=None):        
        self.embedding_model = embedding_model
        self.persist_directory = persist_directory
        self.vector_store = None

    def ingest(self, elements):
        """
        Ingests preprocessed elements into the vector store - chunk, embed, persist(in Vector DB).
        """
        chunked_elements = chunk_text(elements)
        
        # Fallback to simple chunking if no chunks were created
        if not chunked_elements:
            full_text = " ".join([el['text'] for el in elements])
            chunked_elements = [{'text': chunk} for chunk in fallback_chunk(full_text)]
            print("Used fallback simple chunking to create chunks")

        #Extract texts from chunked elements
        text_chunks = []
        for el in chunked_elements:
            if hasattr(el, 'text'):
                text_chunks.append(el.text)
            elif isinstance(el, dict) and 'text' in el:
                text_chunks.append(el['text'])
            else:
                text_chunks.append(str(el))

        # Create or load the vector store
        self.vector_store = Chroma.from_texts(
            texts=text_chunks,
            embedding=self.embedding_model
        )
        
        print(f"Created in-memory vector store with {len(text_chunks)} chunks")


    # provide retriever functionality
    def retriever(self, top_k=5):
        """
        Returns a retriever that fetches top_k similar documents from the vector store.
        """
        if self.vector_store is None:
            raise ValueError("Vector store not initialized. Call ingest() first.")
        
        return self.vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": top_k}
       )