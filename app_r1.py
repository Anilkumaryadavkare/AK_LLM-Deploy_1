from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from transformers import AutoTokenizer, pipeline
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.llms import HuggingFacePipeline

# Initialize tokenizer first
tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")

def load_pdf(file_path):
    reader = PdfReader(file_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

# Load PDF
pdf_text = load_pdf("C:/Users/algol/rag_project/docs/graphcube-ibpl+server+user+guide.pdf")
print(f"Extracted text length: {len(pdf_text)} characters")

# Split text
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=50,
    length_function=lambda text: len(tokenizer.encode(text))
)
chunks = text_splitter.split_text(pdf_text)

# Initialize embeddings
embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
vector_store = FAISS.from_texts(chunks, embeddings)

# Create pipeline
flan_pipeline = pipeline(
    task="text2text-generation",
    model="google/flan-t5-base",
    max_new_tokens=512,
    temperature=0.7,
    do_sample=True,
    top_k=50
)

# Create QA chain
qa = RetrievalQA.from_chain_type(
    llm=HuggingFacePipeline(pipeline=flan_pipeline),
    chain_type="stuff",
    retriever=vector_store.as_retriever(search_kwargs={"k": 3}),
    return_source_documents=True
)

# Test query
query = "Answer in full sentences. What are the main chronological sections mentioned in this document?"
result = qa.invoke({"query": query})

print("Final Answer:", result["result"])
print("\nSources Used:")
for doc in result["source_documents"]:
    print(doc.page_content[:100] + "...")