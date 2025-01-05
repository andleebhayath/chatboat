import os
import streamlit as st
import pinecone
from dotenv import load_dotenv
from langchain.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
from langchain.schema import Document
from langchain.chains.question_answering import load_qa_chain

from langchain.chat_models import ChatOpenAI
load_dotenv()
st.title("📚 AI Chatbot with PDF Knowledge Base")
st.sidebar.header("Upload PDF Documents")

# File Upload Section
uploaded_files = st.sidebar.file_uploader("Upload PDFs", type=["pdf"], accept_multiple_files=True)
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT")

pc = pinecone.Pinecone(
    api_key=PINECONE_API_KEY,
    environment=PINECONE_ENVIRONMENT
)
index_name="lanchainvector"
if index_name not in pc.list_indexes().names():  # Use the client instance to list indexes
    pc.create_index(index_name, dimension=1536, metric="cosine")  # Adjust dimension based on embedding model
 # Adjust dimension based on embedding model

index = pc.Index(index_name)
# OpenAI Embeddings
embeddings = OpenAIEmbeddings(api_key=os.getenv('OPENAI_API_KEY'))
def process_documents(files):
    doc_texts = []
    for file in files:
        with open(f"temp_{file.name}", "wb") as f:
            f.write(file.getbuffer())

        loader = PyPDFDirectoryLoader("./")
        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=50)
        doc_chunks = text_splitter.split_documents(documents)
        
        doc_texts.extend(doc_chunks)
        os.remove(f"temp_{file.name}")  # Cleanup

    return doc_texts

if uploaded_files:
    st.sidebar.success("Processing PDFs...")
    docs = process_documents(uploaded_files)
    st.sidebar.success("PDFs Processed Successfully!")

    # Upsert into Pinecone
    vectors = []
    for i, doc in enumerate(docs):
        embedding = embeddings.embed_query(doc.page_content)
        vectors.append((f"doc_{i}", embedding, {"text": doc.page_content}))

    index.upsert(vectors=vectors)
    st.sidebar.success("Documents uploaded to Pinecone!")

# Load LLM
llm = ChatOpenAI(api_key=os.getenv("OPENAI_API_KEY"), model_name="gpt-3.5-turbo")
qa_chain = load_qa_chain(llm, chain_type="stuff")

# Function to retrieve answers
def retrieve_answer(query, top_k=3):
    query_embedding = embeddings.embed_query(query)
    
    results = index.query(
        vector=query_embedding,
        top_k=top_k,
        include_metadata=True
    )

    retrieved_docs = [match["metadata"]["text"] for match in results.get("matches", [])]
    return retrieved_docs

# Chat Section
query = st.text_input("Ask a Question from the Documents:")
if query:
    retrieved_docs = retrieve_answer(query)
    if retrieved_docs:
        answer = qa_chain.run(input_documents=[Document(page_content=doc) for doc in retrieved_docs], question=query)
        st.write("**Answer:**", answer)
    else:
        st.write("No relevant information found in the uploaded documents.")
