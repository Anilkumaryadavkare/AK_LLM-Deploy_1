import streamlit as st
from app_r1 import load_pdf, initialize_text_splitter, initialize_embeddings, initialize_qa_chain
from langchain_community.vectorstores import FAISS

# Initialize session state
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None

# Title
st.title("📄 Document Q&A with RAG")

# File upload
uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

if uploaded_file:
    with st.spinner("Processing document..."):
        try:
            # Load PDF text
            pdf_text = load_pdf(uploaded_file)

            # Initialize text splitter and split text
            text_splitter = initialize_text_splitter()
            chunks = text_splitter.split_text(pdf_text)

            # Initialize embeddings and create vector store
            embeddings = initialize_embeddings()
            st.session_state.vector_store = FAISS.from_texts(chunks, embeddings)

            # Initialize QA chain
            st.session_state.qa_chain = initialize_qa_chain(st.session_state.vector_store)

            st.success(f"Processed {len(chunks)} text chunks!")
        except Exception as e:
            st.error(f"Error occurred while processing the PDF: {e}")

# Query section
if st.session_state.vector_store and st.session_state.qa_chain:
    query = st.text_input("Ask anything about the document:")

    if query:
        with st.spinner("Searching for answers..."):
            try:
                response = st.session_state.qa_chain.invoke({"query": query})["result"]
                st.subheader("Answer:")
                st.write(response)
            except Exception as e:
                st.error(f"Error occurred while fetching the answer: {e}")
else:
    st.warning("No vector store found. Please upload a PDF to process the document.")
