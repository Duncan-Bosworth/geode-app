# chatbot.py
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
import os
from dotenv import load_dotenv

load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

def ask_question(query):
    embedder = OpenAIEmbeddings(openai_api_key=openai_api_key)
    vectorstore = FAISS.load_local("vector_index", embedder, allow_dangerous_deserialization=True)
    retriever = vectorstore.as_retriever(search_type="similarity", k=5)

    qa = RetrievalQA.from_chain_type(
        llm=ChatOpenAI(openai_api_key=openai_api_key, model_name="gpt-4"),
        chain_type="stuff",
        retriever=retriever
    )

    result = qa.run(query)
    print(f"\n🧠 Answer:\n{result}")

if __name__ == "__main__":
    while True:
        question = input("Ask Geode something: ")
        ask_question(question)
