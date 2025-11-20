from unstructured.chunking.title import chunk_by_title
from langchain_community.vectorstores import Chroma
from langchain.docstore.document import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from typing import List, Any
from pydantic import Field
import re


def chunk_text(elements):
    """
    Chunks the preprocessed text elements based on titles using Unstructured.
    Returns a list of chunked elements.
    """
    chunked_elements = chunk_by_title(elements, combine_text_under_n_chars=100, max_characters=800, overlap=100)
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
    all_chunks_ordered: List[str] = Field(default_factory=list, description="All chunks in original document order")
    
    class Config:
        arbitrary_types_allowed = True
    
    def __init__(self, vector_store, all_chunks_ordered, top_k=5, adjacent_chunks=2, **kwargs):
        super().__init__(
            vector_store=vector_store,
            top_k=top_k,
            adjacent_chunks=adjacent_chunks,
            all_chunks_ordered=all_chunks_ordered,
            **kwargs
        ) 

        self.debug_chunk_indexing()   

    def debug_chunk_indexing(self):
        """Debug method to show all indexed chunks in their original order"""
        print(f"\n🔧 DEBUG: Ordered chunks ({len(self.all_chunks_ordered)} total)")  # CHANGE THIS LINE
        print("-" * 80)
        for i, chunk in enumerate(self.all_chunks_ordered):  # CHANGE THIS LINE
            preview = ' '.join(chunk.split()[:20])  # First 20 words
            print(f"[{i:2d}] '{preview}...' (length: {len(chunk)} chars)")
        print("-" * 80)

    def _extract_key_terms(self, query: str) -> List[str]:
        """Extract meaningful terms from query, removing stop words"""
        # Common stop words
        stop_words = {
            'what', 'is', 'are', 'how', 'does', 'do', 'did', 'will', 'would', 'could', 'should',
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with',
            'by', 'from', 'up', 'about', 'into', 'through', 'during', 'before', 'after',
            'above', 'below', 'can', 'has', 'have', 'had', 'be', 'been', 'being', 'was', 'were'
        }
        
        # Extract words and clean them
        words = re.findall(r'\b\w+\b', query.lower())
        key_terms = [word for word in words if word not in stop_words and len(word) > 2]
        
        return key_terms
    
    def _calculate_generic_relevance_score(self, chunk_text: str, query_terms: List[str]) -> float:
        """Calculate a generic relevance score based on term frequency and positioning"""
        chunk_lower = chunk_text.lower()
        score = 0.0
        
        # Split chunk into lines for positional analysis
        lines = chunk_text.split('\n')
        first_line = lines[0].lower().strip() if lines else ""
        first_50_chars = chunk_lower[:50] if len(chunk_lower) > 50 else chunk_lower
        
        for term in query_terms:
            term_lower = term.lower()
            
            # Count occurrences of the term
            occurrences = chunk_lower.count(term_lower)
            
            if occurrences > 0:
                # Base score for term presence
                score += occurrences * 1.0
                
                # Bonus for term appearing in first line (likely a header/title)
                if term_lower in first_line:
                    score += 3.0
                
                # Bonus for term appearing early in chunk (first 50 characters)
                if term_lower in first_50_chars:
                    score += 2.0
                
                # Bonus for exact term matches (not partial)
                if f" {term_lower} " in f" {chunk_lower} ":
                    score += 1.0
        
        # Bonus for chunks that contain multiple query terms
        terms_found = sum(1 for term in query_terms if term.lower() in chunk_lower)
        if terms_found > 1:
            score += terms_found * 0.5
        
        # Bonus for term density (terms per 100 characters)
        if len(chunk_text) > 0:
            density_bonus = (terms_found / len(chunk_text)) * 100
            score += min(density_bonus, 2.0)  # Cap the density bonus
        
        return score
    
    def _hybrid_search(self, query: str, k: int) -> List[tuple]:
        """Perform generic hybrid search combining vector similarity and term relevance"""
        
        # Extract key terms from query
        query_terms = self._extract_key_terms(query)
        print(f"Key terms extracted: {query_terms}")
        
        # Get vector similarity results
        try:
            vector_results = self.vector_store.similarity_search_with_score(query, k=k*2)
        except:
            # Fallback if similarity search with score fails
            vector_docs = self.vector_store.similarity_search(query, k=k*2)
            vector_results = [(doc, 0.5) for doc in vector_docs]  # Assign neutral score
        
        # Calculate combined scores
        enhanced_results = []
        
        for doc, vector_score in vector_results:
            # Calculate term-based relevance score
            term_score = self._calculate_generic_relevance_score(doc.page_content, query_terms)
            
            # Normalize vector score (lower is better for distance-based similarity)
            # Convert to 0-10 scale where higher is better
            normalized_vector_score = max(0, 10 - (vector_score * 2))
            
            # Combine scores with balanced weights
            combined_score = (normalized_vector_score * 0.6) + (term_score * 0.4)
            
            enhanced_results.append((doc, vector_score, term_score, combined_score))
            
            # Debug output
            preview = doc.page_content[:60].replace('\n', ' ')
            print(f"Chunk: '{preview}...' | Vector: {vector_score:.2f} | Norm. Vector: {normalized_vector_score:.2f} | Terms: {term_score:.2f} | Combined: {combined_score:.2f}")
        
        # Sort by combined score (higher is better)
        enhanced_results.sort(key=lambda x: x[3], reverse=True)
        
        return [(doc, combined_score) for doc, _, _, combined_score in enhanced_results[:k]]

    def _get_relevant_documents(
    self, 
    query: str, 
    *, 
    run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        """Get relevant documents plus their adjacent chunks using generic approach"""
    
        print(f"\n🔍 Generic search for: '{query}'")
        print("-" * 60)
        
        # Use hybrid search for better ranking
        try:
            hybrid_results = self._hybrid_search(query, self.top_k)
        except Exception as e:
            print(f"Hybrid search failed, using fallback: {e}")
            # Fallback to simple vector search
            docs = self.vector_store.similarity_search(query, k=self.top_k)
            return docs[:self.top_k]
        
        if not hybrid_results or not self.all_chunks_ordered:
            return self.vector_store.similarity_search(query, k=self.top_k)
        
        # Take the most relevant chunk
        most_relevant_doc = hybrid_results[0][0]
        most_relevant_score = hybrid_results[0][1]
        most_relevant_chunk = most_relevant_doc.page_content
        
        print(f"✅ Best match (score: {most_relevant_score:.2f}): '{most_relevant_chunk[:100].replace(chr(10), ' ')}...'")
        
        enhanced_docs = []
        added_texts = set()
        
        # Add the most relevant chunk
        enhanced_docs.append(most_relevant_doc)
        added_texts.add(most_relevant_chunk)
        print(f"📝 Added primary chunk (length: {len(most_relevant_chunk)} chars)")   
        
        # Find and add adjacent chunks
        try:
            chunk_index = self.all_chunks_ordered.index(most_relevant_chunk)
            print(f"🎯 Found primary chunk at index: {chunk_index} out of {len(self.all_chunks_ordered)} total chunks")
            print(f"📍 Will search for adjacent chunks in range: {max(0, chunk_index - self.adjacent_chunks)} to {min(len(self.all_chunks_ordered) - 1, chunk_index + self.adjacent_chunks)}")
            
            adjacent_added_count = 0
            for offset in range(-self.adjacent_chunks, self.adjacent_chunks + 1):
                adjacent_index = chunk_index + offset
                
                if offset == 0:
                    continue  # Skip the current chunk
                
                if 0 <= adjacent_index < len(self.all_chunks_ordered):
                    adjacent_text = self.all_chunks_ordered[adjacent_index]

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
                        adjacent_added_count += 1
                        
                        # Debug: Show first few words of each adjacent chunk
                        preview = ' '.join(adjacent_text.split()[:15])  # First 15 words
                        direction = "before" if offset < 0 else "after"
                        print(f"➕ Added adjacent chunk {direction} (offset {offset:+d}): '{preview}...'")
                    else:
                        print(f"⚠️  Skipped duplicate adjacent chunk at offset {offset}")
                else:
                    boundary = "start" if adjacent_index < 0 else "end"
                    print(f"📍 Adjacent index {adjacent_index} is beyond {boundary} of document (offset {offset:+d})")
            
            print(f"📊 Added {adjacent_added_count} adjacent chunks")
        
        except ValueError:
            print("❌ Primary chunk not found in ordered chunks - cannot find adjacent chunks")
            print(f"🔍 Searching for chunk starting with: '{most_relevant_chunk[:50]}...'")
            print(f"🔍 In {len(self.all_chunks_ordered)} indexed chunks")

        # Add remaining relevant chunks if there's space
        remaining_slots = max(0, self.top_k - len(enhanced_docs))
        print(f"📊 Remaining slots for additional chunks: {remaining_slots}")
        
        additional_added = 0
        for doc, score in hybrid_results[1:]:  # Skip the first one (already added)
            if remaining_slots <= 0:
                break
            if doc.page_content not in added_texts:
                enhanced_docs.append(doc)
                added_texts.add(doc.page_content)
                remaining_slots -= 1
                additional_added += 1
                
                preview = ' '.join(doc.page_content.split()[:10])  # First 10 words
                print(f"➕ Added additional relevant chunk: '{preview}...'")
        
        print(f"📊 Final summary: 1 primary + {len(enhanced_docs)-1-additional_added} adjacent + {additional_added} additional = {len(enhanced_docs)} total chunks")
        print("-" * 60)
        return enhanced_docs
    

class RAGPipeline:

    def __init__(self, embedding_model, persist_directory=None):        
        self.embedding_model = embedding_model
        self.persist_directory = persist_directory
        self.vector_store = None
        self.ordered_chunks = []

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

        # Store chunks in their original order
        self.ordered_chunks = text_chunks.copy()
        print(f"Stored {len(self.ordered_chunks)} chunks in original document order")

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
            all_chunks_ordered=self.ordered_chunks,
            top_k=top_k,
            adjacent_chunks=2  # Fetch 2 chunks before and after each relevant chunk
        )   