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
        chunk_size=700,  # Larger chunk size to maintain context
        chunk_overlap=200,  # More overlap for context retention
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
        max_new_tokens=200,  # Increased token limit for complete answers
        temperature=0.1,  # More deterministic output
        do_sample=False,  # Reduce randomness
        top_k=10  # Retrieve a diverse set of relevant results
    )
    
    hf_pipeline = HuggingFacePipeline(pipeline=flan_pipeline)

    qa = RetrievalQA.from_chain_type(
        llm=hf_pipeline,
        chain_type="stuff",
        retriever=vector_store.as_retriever(search_kwargs={"k": 10}),  # Retrieve more chunks
        return_source_documents=True
    )
    return qa
