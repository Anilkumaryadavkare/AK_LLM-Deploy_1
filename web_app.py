import streamlit as st
from app_r1 import load_pdf, text_splitter, embeddings
from langchain_community.vectorstores import FAISS  # Add this import
from app_r1 import qa  # Import your QA chain

st.title("📄 Document Q&A with RAG")

# File upload
uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

# Initialize session state for vector store
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
    st.write("Vector store initialized.")

# Process the uploaded file
if uploaded_file:
    with st.spinner("Processing document..."):
        try:
            # Process PDF text
            text = load_pdf(uploaded_file)
            chunks = text_splitter.split_text(text)

            # Clear previous vector store (if any) and create a new one
            st.session_state.vector_store = None
            st.session_state.vector_store = FAISS.from_texts(
                chunks, embeddings
            )
            
            st.success(f"Processed {len(chunks)} text chunks!")
        except Exception as e:
            st.error(f"Error occurred while processing the PDF: {e}")

# Check if the vector store is initialized
if st.session_state.vector_store:
    query = st.text_input("Ask anything about the document:")
    
    if query:
        with st.spinner("Searching for answers..."):
            try:
                # Get response using the QA chain
                response = qa.invoke({"query": query})["result"]
                
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
