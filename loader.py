# loader.py
import fitz
import pandas as pd
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os

def extract_pdf_text(path):
    doc = fitz.open(path)
    return "\n".join([page.get_text() for page in doc])

def extract_excel_text(path):
    all_text = []
    excel_file = pd.ExcelFile(path)
    for sheet in excel_file.sheet_names:
        df = pd.read_excel(path, sheet_name=sheet)
        text = df.to_markdown(index=False)
        all_text.append(f"### Sheet: {sheet}\n{text}")
    return "\n\n".join(all_text)

def load_documents(folder="docs"):
    data = []
    for filename in os.listdir(folder):
        filepath = os.path.join(folder, filename)
        if filename.endswith(".pdf"):
            text = extract_pdf_text(filepath)
        elif filename.endswith(".xlsx"):
            text = extract_excel_text(filepath)
        else:
            continue
        data.append({"source": filename, "content": text})
    return data

def chunk_documents(docs):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = []
    for doc in docs:
        for chunk in splitter.split_text(doc["content"]):
            chunks.append({
                "content": chunk,
                "source": doc["source"]
            })
    return chunks
