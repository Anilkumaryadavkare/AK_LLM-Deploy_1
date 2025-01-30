import streamlit as st
import re
import os
import numpy as np
from app_r1 import load_pdf, initialize_text_splitter, initialize_embeddings, initialize_qa_chain
from langchain_community.vectorstores import FAISS

# Initialize session state
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None

st.title("📄 Enhanced Document Q&A with RAG")

uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

# Function to clean up the answer
def clean_answer(answer):
    answer = re.sub(r"\bo9 Internal use Only\b", "", answer, flags=re.IGNORECASE)
    answer = re.sub(r"(Leaf Level Query\s*)+", "Leaf Level Query ", answer, flags=re.IGNORECASE)
    answer = re.sub(r"\s+", " ", answer).strip()
    return answer

# **New:** Semantic filtering of passages using cosine similarity
def filter_passages(query, passages, embeddings, threshold=0.5):
    query_embedding = np.array(embeddings.embed_query(query))  # Convert query to vector
    filtered_passages = []
    
    for passage in passages:
        passage_embedding = np.array(embeddings.embed_query(passage.page_content))
        similarity = np.dot(query_embedding, passage_embedding) / (
            np.linalg.norm(query_embedding) * np.linalg.norm(passage_embedding)
        )  # Cosine similarity
        
        if similarity >= threshold:  # Only keep relevant passages
            filtered_passages.append(passage)

    return filtered_passages

FAISS_INDEX_PATH = "faiss_index"

def load_or_create_vector_store(chunks, embeddings):
    if os.path.exists(FAISS_INDEX_PATH):
        print("Loading FAISS index from disk...")
        return FAISS.load_local(FAISS_INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
    else:
        print("Creating FAISS index from scratch...")
        vector_store = FAISS.from_texts(chunks, embeddings)
        vector_store.save_local(FAISS_INDEX_PATH)
        return vector_store

if uploaded_file:
    with st.spinner("Processing document..."):
        try:
            pdf_text = load_pdf(uploaded_file)
            text_splitter = initialize_text_splitter()
            chunks = text_splitter.split_text(pdf_text)

            embeddings = initialize_embeddings()
            st.session_state.vector_store = load_or_create_vector_store(chunks, embeddings)

            st.session_state.qa_chain = initialize_qa_chain(st.session_state.vector_store)

            st.success(f"Processed {len(chunks)} text chunks!")
        except Exception as e:
            st.error(f"Error occurred while processing the PDF: {e}")

if st.session_state.vector_store and st.session_state.qa_chain:
    query = st.text_input("Ask anything about the document:")

    if query:
        with st.spinner("Searching for answers..."):
            try:
                result = st.session_state.qa_chain.invoke({"query": query})
                cleaned_answer = clean_answer(result["result"])

                st.subheader("Answer:")
                st.write(cleaned_answer)

                st.subheader("Relevant Passages:")
                docs = st.session_state.vector_store.similarity_search(query, k=8)  # Retrieve more passages
                filtered_docs = filter_passages(query, docs, st.session_state.qa_chain.retriever.vectorstore.embeddings, threshold=0.6)

                for i, doc in enumerate(filtered_docs):
                    st.markdown(f"**Passage {i + 1}:**")
                    st.write(doc.page_content)
            except Exception as e:
                st.error(f"Error occurred while fetching the answer: {e}")
else:
    st.warning("No vector store found. Please upload a PDF to process the document.")
