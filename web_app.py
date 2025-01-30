import streamlit as st
import re  # For text cleaning
import os  # For file path operations
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

# Function to clean up the answer
def clean_answer(answer):
    # Remove internal-use-only phrases
    answer = re.sub(r"\bo9 Internal use Only\b", "", answer, flags=re.IGNORECASE)
    # Remove repetitive phrases like "Leaf Level Query"
    answer = re.sub(r"(Leaf Level Query\s*)+", "Leaf Level Query ", answer, flags=re.IGNORECASE)
    # Remove extra whitespace
    answer = re.sub(r"\s+", " ", answer).strip()
    return answer

# Function to filter irrelevant passages
def filter_passages(passages, query):
    filtered_passages = []
    irrelevant_keywords = ["o9 Internal use Only", "GraphCube", "IBPL"]  # Add keywords to exclude
    for passage in passages:
        if not any(keyword.lower() in passage.page_content.lower() for keyword in irrelevant_keywords):
            filtered_passages.append(passage)
    return filtered_passages

# Load or create vector store (caching embeddings)
FAISS_INDEX_PATH = "faiss_index"

def load_or_create_vector_store(chunks, embeddings):
    if os.path.exists(FAISS_INDEX_PATH):  # Check if FAISS index exists
        print("Loading FAISS index from disk...")
        return FAISS.load_local(FAISS_INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
    else:
        print("Creating FAISS index from scratch...")
        vector_store = FAISS.from_texts(chunks, embeddings)
        vector_store.save_local(FAISS_INDEX_PATH)
        return vector_store

# Process the uploaded file
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
            st.session_state.vector_store = load_or_create_vector_store(chunks, embeddings)

            # Initialize QA chain
            st.session_state.qa_chain = initialize_qa_chain(st.session_state.vector_store)

            st.success(f"Processed {len(chunks)} text chunks!")
        except Exception as e:
            st.error(f"Error occurred while processing the PDF: {e}")

# Query section
query = None  # Initialize query to avoid NameError
if st.session_state.vector_store and st.session_state.qa_chain:
    query = st.text_input("Ask anything about the document:")

    if query:  # Check if query is not empty
        with st.spinner("Searching for answers..."):
            try:
                # Get response using the QA chain
                result = st.session_state.qa_chain.invoke({"query": query})

                # Post-process the answer to make it more concise and readable
                cleaned_answer = clean_answer(result["result"])

                # Display the answer
                st.subheader("Answer:")
                st.write(cleaned_answer)

                # Show relevant passages in chronological order
                st.subheader("Relevant Passages:")
                docs = st.session_state.vector_store.similarity_search(query, k=5)  # Retrieve top 5 relevant chunks
                filtered_docs = filter_passages(docs, query)  # Filter irrelevant passages
                for i, doc in enumerate(filtered_docs):
                    st.markdown(f"**Passage {i + 1}:**")
                    st.write(doc.page_content)
            except Exception as e:
                st.error(f"Error occurred while fetching the answer: {e}")
else:
    st.warning("No vector store found. Please upload a PDF to process the document.")
