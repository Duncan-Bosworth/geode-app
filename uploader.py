import os
import shutil
from langchain_community.document_loaders import PyPDFLoader, UnstructuredExcelLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

def clear_folder(folder_path):
    if os.path.exists(folder_path):
        shutil.rmtree(folder_path)
    os.makedirs(folder_path)

def load_documents_from_upload(uploaded_files, temp_dir="temp_uploads"):
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
            continue
        docs.extend(loader.load())
    return docs

def build_vectorstore(docs, openai_api_key, output_dir="vector_index"):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    chunks = text_splitter.split_documents(docs)
    embeddings = OpenAIEmbeddings(api_key=openai_api_key)
    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local(output_dir)
