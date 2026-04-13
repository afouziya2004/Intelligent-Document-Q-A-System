import streamlit as st
import sys, os

sys.path.append(os.path.abspath("../src"))

from loader import load_document, split_text
from embedder import create_embeddings
from retriever import retrieve
from answer_generator import generate_answer

st.title("📄 AI Document Q&A System")

text = load_document("../data/sample.txt")
chunks = split_text(text)
vectorizer, vectors = create_embeddings(chunks)

query = st.text_input("Ask a question:")

if query:
    result, score = retrieve(query, vectorizer, vectors, chunks)
    answer = generate_answer(query, result)

    st.write("### 💡 Answer")
    st.write(answer)

    st.write(f"🔎 Similarity Score: {score:.4f}")
