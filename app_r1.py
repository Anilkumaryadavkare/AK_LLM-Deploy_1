import time
import os
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from transformers import AutoTokenizer, pipeline
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_community.llms import HuggingFacePipeline

# Initialize tokenizer
tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")

def load_pdf(file):
    """Load text from a PDF file."""
    reader = PdfReader(file)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

def initialize_text_splitter():
    """Initialize the text splitter."""
    return RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=50,
        length_function=lambda text: len(tokenizer.encode(text))
    )

def initialize_embeddings():
    """Initialize the embedding model."""
    return SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

def initialize_qa_chain(vector_store):
    """Initialize the QA chain."""
    flan_pipeline = pipeline(
        task="text2text-generation",
        model="google/flan-t5-base",
        max_new_tokens=512,
        temperature=0.7,
        do_sample=True,
        top_k=50
    )
    qa = RetrievalQA.from_chain_type(
        llm=HuggingFacePipeline(pipeline=flan_pipeline),
        chain_type="stuff",
        retriever=vector_store.as_retriever(search_kwargs={"k": 3}),
        return_source_documents=True
    )
    return qa