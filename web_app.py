import streamlit as st
from app_r1 import load_pdf, initialize_text_splitter, initialize_embeddings, initialize_qa_chain
from langchain_community.vectorstores import FAISS

# Title
st.title("📄 Document Q&A with RAG")

# Initialize session state for vector store and QA chain
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None

# File upload
uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

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
            st.session_state.vector_store = FAISS.from_texts(chunks, embeddings)

            # Initialize QA chain
            st.session_state.qa_chain = initialize_qa_chain(st.session_state.vector_store)

            st.success(f"Processed {len(chunks)} text chunks!")
        except Exception as e:
            st.error(f"Error occurred while processing the PDF: {e}")

# Check if the vector store and QA chain are initialized
if st.session_state.vector_store and st.session_state.qa_chain:
    query = st.text_input("Ask anything about the document:")

    if query:
        with st.spinner("Searching for answers..."):
            try:
                # Get response using the QA chain
                response = st.session_state.qa_chain.invoke({"query": query})["result"]

                # Display results
                st.subheader("Answer:")
                st.write(response)

                # Show context
                with st.expander("See relevant passages"):
                    docs = st.session_state.vector_store.similarity_search(query, k=2)
                    for i, doc in enumerate(docs):
                        st.markdown(f"**Passage {i+1}:**")
                        st.write(doc.page_content)
            except Exception as e:
                st.error(f"Error occurred while fetching the answer: {e}")
else:
    st.warning("No vector store found. Please upload a PDF to process the document.")