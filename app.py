import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from uploader import load_documents_from_upload, build_vectorstore
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

# Streamlit UI setup
st.set_page_config(page_title="Geode Assistant", layout="wide")
st.title("🧠 Geode: Streetworks Assistant")
st.sidebar.markdown("Ask a question based on your internal docs or streetworks guidance.")
st.sidebar.markdown("### 📤 Upload and Reindex")
uploaded_files = st.sidebar.file_uploader("Upload files", type=["pdf", "xlsx"], accept_multiple_files=True)

if st.sidebar.button("Reindex Documents"):
    if uploaded_files:
        with st.spinner("Building vector index..."):
            docs = load_documents_from_upload(uploaded_files)
            build_vectorstore(docs, openai_api_key)
        st.sidebar.success("✅ Reindex complete. You can now ask questions.")
    else:
        st.sidebar.warning("⚠️ Please upload at least one PDF or Excel file.")


if "history" not in st.session_state:
    st.session_state.history = []

# Build the QA chain
@st.cache_resource
def get_qa():
    embedder = OpenAIEmbeddings(api_key=openai_api_key)
    vectorstore = FAISS.load_local("vector_index", embedder, allow_dangerous_deserialization=True)
    retriever = vectorstore.as_retriever(search_type="similarity", k=5)

    llm = ChatOpenAI(
        openai_api_key=openai_api_key,
        model_name="gpt-4",
        temperature=0
    )

    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template=(
            "You are Geode, a calm and precise assistant specialising in UK streetworks legislation and best practice. "
            "Always refer to official guidance. Never speculate.\n\n"
            "Context:\n{context}\n\n"
            "Question: {question}"
        )
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt}
    )

    return qa_chain

qa = get_qa()

# User input
question = st.text_input("Ask Geode something...", placeholder="e.g. What are the FPN charges?")

if question:
    result = qa({"query": question})
    answer = result["result"]
    sources = result.get("source_documents", [])
    st.session_state.history.append((question, answer, sources))

# Display chat history
for q, a, sources in reversed(st.session_state.history):
    st.markdown(f"**You:** {q}")
    st.markdown(f"**Geode:** {a}")
    if sources:
        st.markdown("**Sources:**")
        for doc in sources:
            st.markdown(f"- *{doc.metadata.get('source', 'Unknown source')}*")
        with st.expander("Show matched content"):
            for doc in sources:
                st.markdown(f"**From {doc.metadata.get('source', 'Unknown')}**")
                st.code(doc.page_content.strip()[:1000])
    st.markdown("---")
