from unstructured.chunking.title import chunk_by_title
from langchain_community.vectorstores import Chroma
from langchain.docstore.document import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from typing import List, Any
from pydantic import Field


def chunk_text(elements):
    """
    Chunks the preprocessed text elements based on titles using Unstructured.
    Returns a list of chunked elements.
    """
    chunked_elements = chunk_by_title(elements, combine_text_under_n_chars=100, max_characters=800, overlap=100, overlap_all=True)
    return chunked_elements


def fallback_chunk(text: str, max_chars=800):
    """
    Simple fixed-size chunker.
    Only used when title-based chunking returns nothing.
    """
    overlap = 100
    for i in range(0, len(text), max_chars - overlap):
        chunk = text[i:i + max_chars]
        if chunk.strip():  # Only yield non-empty chunks
            yield chunk

class SequentialRetriever(BaseRetriever):
    """Retriever that fetches adjacent chunks to ensure topic completeness"""

    vector_store: Any = Field(description="The vector store to retrieve from")
    top_k: int = Field(default=5, description="Number of top documents to retrieve")
    adjacent_chunks: int = Field(default=3, description="Number of adjacent chunks to fetch on each side")
    all_chunks: List[str] = Field(default_factory=list, description="All chunks for sequential access")
    
    class Config:
        arbitrary_types_allowed = True
    
    def __init__(self, vector_store, top_k=5, adjacent_chunks=3, **kwargs):
        super().__init__(
            vector_store=vector_store,
            top_k=top_k,
            adjacent_chunks=adjacent_chunks,
            all_chunks=[],
            **kwargs
        )
        self._index_chunks()
    
    def _index_chunks(self):
        """Index all chunks for sequential access"""
        try:
            # Search with generic terms to get all chunks
            all_docs = self.vector_store.similarity_search("", k=1000)
            self.all_chunks = [doc.page_content for doc in all_docs]
            print(f"Indexed {len(self.all_chunks)} chunks for sequential retrieval")
        except Exception as e:
            print(f"Warning: Could not index chunks: {e}")
            self.all_chunks = []

    def _get_relevant_documents(
        self, 
        query: str, 
        *, 
        run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        """Get relevant documents plus their adjacent chunks"""
        
        # Step 1: Get initial relevant chunks with similarity scores
        initial_docs_with_scores = self.vector_store.similarity_search_with_score(query, k=self.top_k)
        
        if not initial_docs_with_scores or not self.all_chunks:
            # Fallback to regular search if scoring fails
            return self.vector_store.similarity_search(query, k=self.top_k)
        
        # Step 2: Sort by relevance (lower score = more relevant for most vector stores)
        initial_docs_with_scores.sort(key=lambda x: x[1])
        
        # Step 3: Take the MOST RELEVANT chunk for adjacent expansion
        most_relevant_doc = initial_docs_with_scores[0][0]
        most_relevant_chunk = most_relevant_doc.page_content
        
        print(f"Most relevant chunk for '{query}': {most_relevant_chunk[:100]}...")
        
        enhanced_docs = []
        added_texts = set()
        
        # Step 4: Add the most relevant chunk first
        enhanced_docs.append(most_relevant_doc)
        added_texts.add(most_relevant_chunk)
    
        # Step 5: Find and add adjacent chunks for the most relevant chunk only
        try:
            chunk_index = self.all_chunks.index(most_relevant_chunk)
            print(f"Found most relevant chunk at index: {chunk_index}")
            
            # Add adjacent chunks (both before and after)
            for offset in range(-self.adjacent_chunks, self.adjacent_chunks + 1):
                adjacent_index = chunk_index + offset
                
                # Skip the current chunk (offset = 0) as we already added it
                if offset == 0:
                    continue
                
                # Check if the adjacent index is valid
                if 0 <= adjacent_index < len(self.all_chunks):
                    adjacent_text = self.all_chunks[adjacent_index]
                    
                    # Only add if we haven't seen this chunk before
                    if adjacent_text not in added_texts:
                        adjacent_doc = Document(
                            page_content=adjacent_text,
                            metadata={
                                "type": "adjacent", 
                                "offset": offset,
                                "source_chunk_index": chunk_index
                            }
                        )
                        enhanced_docs.append(adjacent_doc)
                        added_texts.add(adjacent_text)
                        print(f"Added adjacent chunk at offset {offset}")
        
        except ValueError:
            print("Most relevant chunk not found in indexed chunks")

        # Step 6: Add remaining relevant chunks if there's space
        remaining_slots = max(0, self.top_k - len(enhanced_docs))
        for doc, score in initial_docs_with_scores[1:]:  # Skip the first one (already added)
            if remaining_slots <= 0:
                break
            if doc.page_content not in added_texts:
                enhanced_docs.append(doc)
                added_texts.add(doc.page_content)
                remaining_slots -= 1
        
        print(f"Final retrieval: 1 primary + {len(enhanced_docs)-1} adjacent/additional chunks = {len(enhanced_docs)} total")
        return enhanced_docs    
    
    
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

    #def retriever(self, top_k=5):
    #    """
    #    Returns a retriever that fetches top_k similar documents from the vector store.
    #    """
    #    if self.vector_store is None:
    #        raise ValueError("Vector store not initialized. Call ingest() first.")
    #
    #    return self.vector_store.as_retriever(
    #        search_type="similarity",
    #        search_kwargs={"k": top_k}
    #   )

    def retriever(self, top_k=5):
        """
        Returns a sequential retriever that includes adjacent chunks for better topic coverage.
        """
        if self.vector_store is None:
            raise ValueError("Vector store not initialized. Call ingest() first.")
        
        return SequentialRetriever(
            vector_store=self.vector_store,
            top_k=top_k,
            adjacent_chunks=2  # Fetch 2 chunks before and after each relevant chunk
        )