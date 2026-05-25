import os  
import warnings

#Silence all conversational and path alignment logging warnings from the terminal screen
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message="Accessing __path__")

import streamlit as st
from dotenv import load_dotenv 
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings  
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser


#AUTOMATED ENVIRONMENT CONFIGURATION

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = os.getenv("INDEX_NAME")

#Map explicit environment variables to satisfy internal LangChain system expectations
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY

st.set_page_config(page_title="LitIntel Engine", page_icon="📝", layout="wide")

#STREAMLIT PERFORMANCE CACHE LAYER

@st.cache_resource
def get_embedding_model():
    """Loads a powerful, completely free embedding model that runs locally."""
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

@st.cache_resource
def init_vector_db():
    """Establishes connection and provisions the Cloud Serverless Database Index."""
    if not PINECONE_API_KEY:
        st.error("❌ Setup Error: PINECONE_API_KEY missing from your hidden .env file.")
        return None
        
    pc = Pinecone(api_key=PINECONE_API_KEY)
    existing_indexes = [idx.name for idx in pc.list_indexes()]
    
    if INDEX_NAME not in existing_indexes:
        pc.create_index(
            name=INDEX_NAME,
            dimension=384,  # Matches sentence-transformers dimensionality output layout
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )
    return pc


#BACKGROUND DIRECTORY INGESTION ENGINE

def process_workspace_dataset():
    """Loads all PDFs out of local /dataset folder and pushes vector structures to cloud."""
    if not os.path.exists("dataset") or len(os.listdir("dataset")) == 0:
        st.error("The folder '/dataset' is empty or missing. Place your PDFs inside it.")
        return 0
        
    loader = DirectoryLoader("dataset", glob="**/*.pdf", loader_cls=PyPDFLoader)
    documents = loader.load()
    
    # Chunking configuration strategy creating overlapping contextual bridges [cite: 22, 26]
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(documents)
    
    embeddings = get_embedding_model()
    PineconeVectorStore.from_documents(chunks, embeddings, index_name=INDEX_NAME)
    return len(chunks)

def get_rag_chain():
    """Constructs the explicit zero-hallucination LCEL processing pipeline."""
    embeddings = get_embedding_model()
    vectorstore = PineconeVectorStore(index_name=INDEX_NAME, embedding=embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    
    prompt_template = """You are a domain-specific research assistant. Answer the user's question using ONLY the provided background context.
    If the context does not explicitly contain the details needed to answer, reply strictly with: 
    "I cannot find a grounded answer within the provided knowledge base." Do not synthesize or assume information outside these bounds.

    Context:
    {context}

    Question: 
    {question}

    Answer:"""
    
    prompt = ChatPromptTemplate.from_template(prompt_template)
    
    llm = ChatGroq(
        groq_api_key=GROQ_API_KEY,
        model_name="llama-3.3-70b-versatile",
        temperature=0.0  # Force deterministic factual precision responses
    )
    
    # Explicit LangChain Expression Language layout stream 
    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain


#AUTOMATED LIFECYCLE MANAGEMENT

db_client = init_vector_db()

if "dataset_indexed" not in st.session_state:
    st.session_state["dataset_indexed"] = False

# Smart check to see if Pinecone cloud already has vectors to prevent redundant uploads
if db_client is not None and not st.session_state["dataset_indexed"]:
    try:
        index_description = db_client.Index(INDEX_NAME).describe_index_stats()
        total_vectors = index_description.get("total_vector_count", 0)
        
        if total_vectors > 0:
            # Vectors exist in cloud, skip local document processing completely
            st.session_state["dataset_indexed"] = True
            st.toast(f"Connected to Cloud! Found {total_vectors} existing vectors.", icon="🛰️")
        else:
            # Index is empty, proceed with automated onboarding
            with st.spinner("🚀 Boot Sequence: Automatically processing, chunking, and cloud-indexing your PDFs... Please hold."):
                total_chunks = process_workspace_dataset()
                if total_chunks > 0:
                    st.session_state["dataset_indexed"] = True
                    st.toast(f"System Optimized! Auto-indexed {total_chunks} chunks.", icon="🛰️")
    except Exception as e:
        st.error(f"Automatic Index Exception Pipeline: {e}")


#REFINED FRONTEND USER INTERFACE

st.title("🔬 LitIntel: Advanced Research RAG Dashboard")
st.subheader("Search and Analyze Your Literature PDFs in Real-Time with Groq AI")
st.markdown("---")

# Dynamic status notification bar
if st.session_state["dataset_indexed"]:
    st.success("System Status: Active. Knowledge graph successfully synced with Pinecone cloud layer.")
else:
    st.warning("System Status: Establishing communication matrix with cloud indexes...")

user_query = st.text_input(
    "Enter your literature query here:", 
    placeholder="Type query and press Enter... (e.g., 'What metrics are used in InterviewBot?')"
)

if user_query:
    with st.spinner("Generating insights..."):
        try:
            chain = get_rag_chain()
            response = chain.invoke(user_query)
            st.markdown("### Response:")
            st.info(response)
        except Exception as e:
            st.error(f"Execution Error Breakdown: {e}")