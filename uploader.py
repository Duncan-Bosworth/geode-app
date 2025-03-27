import os
import shutil
from langchain_community.document_loaders import PyPDFLoader, UnstructuredExcelLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

def clear_folder(folder_path):
    """Ensure the temporary upload folder is clean before writing new files."""
    if os.path.exists(folder_path):
        shutil.rmtree(folder_path)
    os.makedirs(folder_path)

def load_documents_from_upload(uploaded_files, temp_dir="temp_uploads"):
    """Save uploaded files locally and load them into LangChain Document format."""
    clear_folder(temp_dir)
    docs = []
    for file in uploaded_files:
        file_path = os.path.join(temp_dir, file.name)
        with open(file_path, "wb") as f:
            f.write(file.getbuffer())

        if file.name.endswith(".pdf"):
            loader = PyPDFLoader(file_path)
        elif file.name.endswith(".xlsx"):
            loader = UnstructuredExcelLoader(file_path)
        else:
            continue  # Skip unsupported file types

        docs.extend(loader.load())

    return docs

def build_vectorstore(docs, openai_api_key, output_dir="vector_index"):
    """Split, embed, and store document chunks using FAISS."""
    # Split text into chunks for better retrieval
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    chunks = text_splitter.split_documents(docs)

    # Embed and build FAISS vectorstore
    embeddings = OpenAIEmbeddings(api_key=openai_api_key)
    vectorstore = FAISS.from_documents(chunks, embeddings)

    # Ensure output path exists, then save
    os.makedirs(output_dir, exist_ok=True)
    vectorstore.save_local(output_dir)
