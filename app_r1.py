import time
import os
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from transformers import AutoTokenizer, pipeline
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_community.llms import HuggingFacePipeline
from langchain_huggingface import HuggingFaceEmbeddings

# Start timing model initialization
start_time = time.time()
tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
print(f"Tokenizer loaded in {time.time() - start_time:.2f} seconds")

def load_pdf(file_path):
    """Load and extract text from a PDF."""
    start_time = time.time()
    reader = PdfReader(file_path)
    text = "".join(page.extract_text() for page in reader.pages if page.extract_text())
    print(f"PDF loaded in {time.time() - start_time:.2f} seconds")
    return text

# Load PDF
pdf_text = load_pdf("C:/Users/algol/rag_project/docs/graphcube-ibpl+server+user+guide.pdf")
print(f"Extracted text length: {len(pdf_text)} characters")

# Split text
start_time = time.time()
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=50,
    length_function=lambda text: len(tokenizer.encode(text))
)
chunks = text_splitter.split_text(pdf_text)
print(f"Text split into {len(chunks)} chunks in {time.time() - start_time:.2f} seconds")

# Initialize embeddings
start_time = time.time()
embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

FAISS_INDEX_PATH = "faiss_index"

if os.path.exists(FAISS_INDEX_PATH):
    print("Loading FAISS index from disk...")
    vector_store = FAISS.from_texts(chunks, embeddings)
    vector_store = FAISS.load_local(FAISS_INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
    
else:
    print("Creating FAISS index from scratch...")
    vector_store = FAISS.from_texts(chunks, embeddings)
    vector_store.save_local(FAISS_INDEX_PATH)

print(f"Vector store initialized in {time.time() - start_time:.2f} seconds")

# Create pipeline
start_time = time.time()
flan_pipeline = pipeline(
    task="text2text-generation",
    model="google/flan-t5-base",
    max_new_tokens=512,
    temperature=0.7,
    do_sample=True,
    top_k=50
)
print(f"LLM pipeline loaded in {time.time() - start_time:.2f} seconds")

# Create QA chain
start_time = time.time()
qa = RetrievalQA.from_chain_type(
    llm=HuggingFacePipeline(pipeline=flan_pipeline),
    chain_type="stuff",
    retriever=vector_store.as_retriever(search_kwargs={"k": 3}),
    return_source_documents=True
)
print(f"QA Chain initialized in {time.time() - start_time:.2f} seconds")

# Test query
query = "Answer in full sentences. What are the main chronological sections mentioned in this document?"
start_time = time.time()
result = qa.invoke({"query": query})
print(f"Query processed in {time.time() - start_time:.2f} seconds")

print("Final Answer:", result["result"])
print("\nSources Used:")
for doc in result["source_documents"]:
    print(doc.page_content[:100] + "...")
