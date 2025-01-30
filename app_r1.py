import os
import re
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from transformers import pipeline
from langchain.chains import RetrievalQA
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.llms import HuggingFacePipeline  # Import HuggingFacePipeline

# Load PDF and extract text
def load_pdf(file):
    reader = PdfReader(file)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    # Clean up text (e.g., remove extra spaces, newlines)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# Initialize text splitter
def initialize_text_splitter():
    return RecursiveCharacterTextSplitter(
        chunk_size=500,  # Increase chunk size for faster processing
        chunk_overlap=100,  # Reduce overlap to minimize redundancy
        length_function=len
    )

# Initialize embeddings
def initialize_embeddings():
    model_name = "sentence-transformers/all-MiniLM-L6-v2"  # Use a lightweight embedding model
    return HuggingFaceEmbeddings(model_name=model_name)

# Initialize QA chain
def initialize_qa_chain(vector_store):
    # Create a Hugging Face pipeline
    flan_pipeline = pipeline(
        task="text2text-generation",
        model="google/flan-t5-base",  # Use a lightweight language model
        max_new_tokens=150,  # Limit token length for concise answers
        temperature=0.3,     # Lower temperature for more deterministic outputs
        do_sample=True,
        top_k=30             # Reduce top_k for more focused responses
    )
    
    # Wrap the pipeline with HuggingFacePipeline
    hf_pipeline = HuggingFacePipeline(pipeline=flan_pipeline)
    
    qa = RetrievalQA.from_chain_type(
        llm=hf_pipeline,  # Use the wrapped pipeline
        chain_type="stuff",
        retriever=vector_store.as_retriever(search_kwargs={"k": 3}),  # Retrieve fewer but more relevant chunks
        return_source_documents=True
    )
    return qa
