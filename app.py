import streamlit as st
import tempfile
import os
from document_processor import process_document_for_QnA

st.set_page_config(page_title="Document Q&A Bot", page_icon="📄")

# Supported file types - easily expandable
SUPPORTED_FILES = ['txt', 'md', 'html', 'htm', 'pptx', 'csv', 'docx']

st.title("📄 Document Q&A Bot")

# Initialize session state
if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "document_processed" not in st.session_state:
    st.session_state.document_processed = False
if "current_document" not in st.session_state:
    st.session_state.current_document = None

# File upload
uploaded_file = st.file_uploader("Upload a document", type=SUPPORTED_FILES)

# Reset state when a new file is uploaded (different from current)
if uploaded_file and uploaded_file.name != st.session_state.current_document:
    st.session_state.qa_chain = None
    st.session_state.chat_history = []
    st.session_state.document_processed = False
    st.session_state.current_document = None

# Process Document button - disabled if already processed
if uploaded_file and st.button("Process Document", disabled=st.session_state.document_processed):
    with st.spinner("Processing..."):
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=f'.{uploaded_file.name.split(".")[-1]}') as tmp:
            tmp.write(uploaded_file.getvalue())
            tmp_path = tmp.name
        
        # Process the document
        qa_chain = process_document_for_QnA(tmp_path)
        
        if qa_chain:
            st.session_state.qa_chain = qa_chain
            st.session_state.chat_history = []
            st.session_state.document_processed = True
            st.session_state.current_document = uploaded_file.name
            st.success(f"✅ Processed: {uploaded_file.name}")
            st.rerun()  # This forces the UI to update immediately
        else:
            st.error("❌ Failed to process document")
        
        # Clean up
        os.unlink(tmp_path)

# Show currently processed document
if st.session_state.document_processed and st.session_state.current_document:
    st.info(f"📄 Currently loaded: **{st.session_state.current_document}**")

# Chat interface
if st.session_state.qa_chain:
    if question := st.chat_input("Ask about your document..."):
        # Get answer
        result = st.session_state.qa_chain.invoke({
            "question": question, 
            "chat_history": st.session_state.chat_history
        })
        
        # Display
        st.write(f"**Q:** {question}")
        st.write(f"**A:** {result['answer']}")
        
        # Only show sources if the question was actually answered (not rejected)
        answer_text = result['answer'].lower()
        if ("cannot answer this question based on the provided document content" not in answer_text and 
            result.get('source_documents')):
            with st.expander("📄 Sources"):
                for i, doc in enumerate(result['source_documents'], 1):
                    st.text(f"[{i}] {doc.page_content.strip()}")
        
        # Update history
        st.session_state.chat_history.append((question, result['answer']))
        st.divider()

else:
    st.info("👆 Upload a document to get started!")