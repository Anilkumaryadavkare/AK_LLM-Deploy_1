import os
import re
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from transformers import pipeline
from langchain.chains import RetrievalQA
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.llms import HuggingFacePipeline

# Load PDF and extract text
def load_pdf(file):
    reader = PdfReader(file)
    text = " ".join(page.extract_text() for page in reader.pages if page.extract_text())
    text = re.sub(r"\s+", " ", text).strip()  # Clean up text
    return text

# Initialize text splitter
def initialize_text_splitter():
    return RecursiveCharacterTextSplitter(
        chunk_size=500,  # Keep chunks moderately large for context
        chunk_overlap=150,  # Slightly increase overlap for better retrieval
        length_function=len
    )

# Initialize embeddings
def initialize_embeddings():
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    return HuggingFaceEmbeddings(model_name=model_name)

# Initialize QA chain
def initialize_qa_chain(vector_store):
    # Use a more precise generation pipeline
    flan_pipeline = pipeline(
        task="text2text-generation",
        model="google/flan-t5-base",
        max_new_tokens=100,  # Limit verbosity
        temperature=0.1,  # Make responses more deterministic
        do_sample=False,  # Disable sampling for consistency
        top_k=20  # Reduce randomness further
    )
    
    hf_pipeline = HuggingFacePipeline(pipeline=flan_pipeline)

    qa = RetrievalQA.from_chain_type(
        llm=hf_pipeline,
        chain_type="stuff",
        retriever=vector_store.as_retriever(search_kwargs={"k": 8}),  # Retrieve more relevant passages
        return_source_documents=True
    )
    return qa
