import os
import sys
from dotenv import load_dotenv
from preprocessing.txt_preprocessor import preprocess_txt
from preprocessing.md_preprocessor import preprocess_md
from preprocessing.html_preprocessor import preprocess_html
from preprocessing.pptx_preprocessor import preprocess_pptx
from preprocessing.csv_preprocessor import preprocess_csv
from preprocessing.docx_preprocessor import preprocess_docx
from rag_pipeline import RAGPipeline
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import ConversationalRetrievalChain, LLMChain
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate

def process_document_for_QnA(document_path):
    """
    Process a document and create a Q&A chain for interactive querying.
    
    Args:
        document_path (str): Path to the document to be processed
        
    Returns:
        ConversationalRetrievalChain: The configured QA chain for querying
    """
    # Load environment variables from .env file
    load_dotenv()
    
    # Get and verify API key
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        print("Error: OPENAI_API_KEY not found in environment variables.")
        print("Please check your .env file.")
        return None
    
    # Verify document exists
    if not os.path.exists(document_path):
        print(f"Error: Document '{document_path}' not found.")
        return None

    print(f"Processing document: {document_path}")

    # Step 1: Preprocess → Elements
    elements = preprocess_document(document_path)

    # Step 2: Embedding model
    embedding_model = OpenAIEmbeddings(
        model="text-embedding-3-small",
        api_key=openai_api_key
    )

    # Step 3: RAG pipeline : chunk + embed + persist
    rag = RAGPipeline(
        embedding_model=embedding_model
    )   
    rag.ingest(elements)
    print("Document processed and stored in memory.")

    # Step 4: Create retriever
    retriever = rag.retriever(top_k=5)

    # Step 5: LLM for answering
    llm = ChatOpenAI(model="gpt-4o", temperature=0, api_key=openai_api_key)


    # Step 6: Create chains for conversational Q & A chat

    #Create custom prompt for document-only responses for user questions
    qa_prompt = PromptTemplate(
        input_variables=["context", "question"],
        template="""You are a helpful assistant that answers questions ONLY based on the provided context from a document.

STRICT RULES:
1. Answer ONLY using information explicitly stated in the context below
2. If the question cannot be answered using the provided context, respond with: "I cannot answer this question based on the provided document content."
3. Do NOT use your general knowledge or training data
4. Do NOT make assumptions beyond what's explicitly stated in the context
5. If information is partially available, state what you know from the context and what's missing

Context from document:
{context}

Question: {question}

Answer based ONLY on the context above:"""
)
    # Create document chain with custom prompt. this chain will be used to answer the user question
    doc_chain = load_qa_chain(llm, chain_type="stuff", prompt=qa_prompt)

    # Create question generator chain to rephrase user questions based on chat history. 
    # First, create the prompt to rephrase questions
    rephrase_prompt = PromptTemplate(
        input_variables=["chat_history", "question"],
        template="""Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question.
        
IMPORTANT: Keep the rephrased question as close as possible to the original question's intent and keywords.
Only add context from chat history if absolutely necessary for understanding.

Chat History:
{chat_history}

Follow Up Input: {question}

Instructions: 
- If the follow up question is already clear and standalone, return it with minimal changes
- Only add context if the question contains pronouns (it, this, that) or is unclear without history
- Preserve the exact keywords and phrases from the original question when possible

Standalone question:"""
    )

    # Create question generator chain
    question_generator_chain = LLMChain(llm=llm, prompt=rephrase_prompt)
    
    qa_chain = ConversationalRetrievalChain(   # provides conversational Q&A functionality
        retriever=retriever,  # provide retriever functionality
        question_generator=question_generator_chain, # rephrase user questions based on chat history and current question
        combine_docs_chain=doc_chain, # answer user questions based on document content
        return_source_documents=True  # return source chunks in the response
    )

    return qa_chain

def get_file_extension(file_path):
    """Get the file extension in lowercase"""
    return os.path.splitext(file_path)[1].lower()

def preprocess_document(document_path):
    """
    Preprocess document based on file type.
    Currently supports .txt files, easily expandable for other types.
    
    Args:
        document_path (str): Path to the document
        
    Returns:
        list: Preprocessed elements or None if unsupported file type
    """
    file_ext = get_file_extension(document_path)
    
    if file_ext == '.txt':
        return preprocess_txt(document_path)
    elif file_ext == '.md':        
        return preprocess_md(document_path)
    elif file_ext == '.html' or file_ext == '.htm':
        return preprocess_html(document_path)
    elif file_ext == '.pptx':        
        return preprocess_pptx(document_path)
    elif file_ext == '.csv':
        return preprocess_csv(document_path)
    elif file_ext == '.docx':
        return preprocess_docx(document_path)
    else:
        print(f"Error: Unsupported file type '{file_ext}'. Currently supported: .txt, .md")
        return None


def run_interactive_session(qa_chain):
    """
    Run an interactive command-line session with the QA chain.
    
    Args:
        qa_chain: The configured ConversationalRetrievalChain
    """
    if not qa_chain:
        print("Failed to initialize QA chain.")
        return
        
    print("\n=== RAG Bot Started ===")
    print("Type 'exit' to quit\n")
    
    chat_history = []
    while True:
        query = input("Enter your query: ")
        if query.lower() in ['exit', 'quit']:
            print("Goodbye!")
            break
        
        try:
            result = qa_chain.invoke({"question": query, "chat_history": chat_history})
            print(f"\nAnswer: {result['answer']}")            
            print("-" * 50)

            # Only show source chunks if the question was actually answered (not rejected)
            answer_text = result['answer'].lower()
            if ("cannot answer this question based on the provided document content" not in answer_text and
                'source_documents' in result and result['source_documents']):
                print("\n📄 Source chunks used to answer this question:")
                print("=" * 60)
                for i, doc in enumerate(result['source_documents'], 1):
                    chunk_text = doc.page_content.strip()
                    # Truncate very long chunks for readability
                    if len(chunk_text) > 300:
                        chunk_text = chunk_text[:300] + "..."
                    print(f"\n[Source {i}]:")
                    print(f"{chunk_text}")
                    print("-" * 40)
            else:
                print("\n📄 No source documents retrieved.")
            
            # Update chat history
            chat_history.append((query, result['answer']))
            
        except Exception as e:
            print(f"Error: {e}")
            print("Please try again.")


def get_document_path():
    """
    Get document path from user input with validation.
    
    Returns:
        str: Valid document path or None if user cancels
    """
    while True:
        document_path = input("Enter the path to your document (or 'quit' to exit): ").strip()
        
        if document_path.lower() in ['quit', 'exit']:
            print("Goodbye!")
            return None
        
        # Remove surrounding quotes if present
        document_path = document_path.strip('"').strip("'")
        
        # Normalize path separators
        document_path = os.path.normpath(document_path)
        
        if not document_path:
            print("Please enter a valid file path.")
            continue
            
        if os.path.exists(document_path):
            return document_path
        else:
            print(f"Error: File '{document_path}' not found. Please try again.")


if __name__ == "__main__":

    # Handle command line arguments
    if len(sys.argv) > 1:
        document_path = sys.argv[1]
    else:
         document_path = get_document_path()
        
         if not document_path:
            sys.exit(0)
    
   
    # Process document and create QA chain
    qa_chain = process_document_for_QnA(document_path)
    
    
    # Run interactive session
    run_interactive_session(qa_chain)