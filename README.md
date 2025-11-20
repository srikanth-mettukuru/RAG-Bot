# 📄 Document Q&A Bot

A smart document question-answering application built with **RAG (Retrieval-Augmented Generation)** that allows you to upload documents and ask questions about their content.

## 🚀 Features

- **Multi-format Support**: Upload txt, md, html, pptx, csv, docx files
- **Intelligent Q&A**: Ask questions in natural language about your documents
- **Source References**: See exactly which parts of your document were used to answer questions
- **Chat Interface**: Conversational experience with chat history
- **Real-time Processing**: Fast document processing and question answering

## 🛠️ Technology Stack

- **🔧 Unstructured** - Document parsing and preprocessing
- **🗄️ Chroma Vector DB** - Vector storage and similarity search
- **⛓️ LangChain** - LLM application framework
- **🤖 OpenAI** - Embedding model and LLM for generation
- **🎨 Streamlit** - Web application interface

## 📋 Prerequisites

- Python 3.11
- OpenAI API key


## 🎯 How to Use

1. **Upload** a supported document using the file uploader
2. **Process** the document by clicking "Process Document"
3. **Ask questions** about your document in the chat interface
4. **View sources** to see which document sections were referenced

## 📁 Project Structure

```
RAG-Bot/
├── app.py                 # Main Streamlit application
├── document_processor.py  # Document processing and RAG chain setup
├── rag_pipeline.py        # RAG pipeline implementation
├── preprocessing/         # Document preprocessing modules
│   ├
│   └── (preprocessing utilities that use the Unstructured libraries)
├── requirements.txt       # Python dependencies
└── README.md             # Project documentation
```