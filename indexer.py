# indexer.py
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from loader import load_documents, chunk_documents
import os
from dotenv import load_dotenv

load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

def build_vector_index():
    docs = load_documents()
    chunks = chunk_documents(docs)

    texts = [c["content"] for c in chunks]
    metadatas = [{"source": c["source"]} for c in chunks]

    embedder = OpenAIEmbeddings(openai_api_key=openai_api_key)
    vectorstore = FAISS.from_texts(texts, embedder, metadatas=metadatas)
    vectorstore.save_local("vector_index")

if __name__ == "__main__":
    build_vector_index()
