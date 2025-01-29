
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter

def load_pdf(file_path):  # ✅ Correct parameter definition
    reader = PdfReader(file_path)  # ✅ Use the parameter variable
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text


# Test PDF loading
pdf_text = load_pdf("C:/Users/algol/rag_project/docs/graphcube-ibpl+server+user+guide.pdf")
print(f"Extracted text length: {len(pdf_text)} characters")



from transformers import AutoTokenizer  # Add this import

# Initialize tokenizer BEFORE creating the text splitter
tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")

# Now create the text splitter with the tokenizer
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=50,
    length_function=lambda text: len(tokenizer.encode(text))
)

# Then split the text
chunks = text_splitter.split_text(pdf_text)

from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.vectorstores import FAISS

# Create embeddings
embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

# Create vector store
vector_store = FAISS.from_texts(chunks, embeddings)
vector_store.save_local("faiss_index")

from langchain.chains import RetrievalQA
from langchain.llms import HuggingFaceHub

from transformers import pipeline
from langchain.llms import HuggingFacePipeline


flan_pipeline = pipeline(
    task="text2text-generation",
    model="google/flan-t5-base",
    max_new_tokens=512,
    temperature=0.7,  # Increased for more creative responses
    do_sample=True,
    top_k=50
)



# 2. Create LLM wrapper
llm = HuggingFacePipeline(pipeline=flan_pipeline)

# 3. Use in QA chain
# Updated imports
from langchain_huggingface import HuggingFacePipeline  # After installing langchain-huggingface

# ... (previous PDF loading and text splitting code)

# Improved QA chain configuration
qa = RetrievalQA.from_chain_type(
    llm=HuggingFacePipeline(pipeline=flan_pipeline),
    chain_type="stuff",
    retriever=vector_store.as_retriever(search_kwargs={"k": 3}),
    return_source_documents=True  # Add this to debug context
)

# Modified query execution


query = "Answer in full sentences. What are the main chronological sections mentioned in this document?"
result = qa.invoke({"query": query})

print("Final Answer:", result["result"])
print("\nSources Used:")
for doc in result["source_documents"]:
    print(doc.page_content[:100] + "...")
