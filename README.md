# LitIntel Engine: Domain-Specific Technical RAG Dashboard

An advanced, zero-hallucination Retrieval-Augmented Generation (RAG) platform built to securely extract factual insights from a specialized literature corpus without data bleeding.

##  Features
* **Zero-Hallucination Guardrails:** Locked down using Llama 3.3 70B via Groq API at a strict temperature ($\tau = 0.0$).
* **Smart Lifecycle Boot Gate:** Pings Pinecone index statistics on startup to safely bypass redundant document reprocessing and cloud upload cycles.
* **Semantic Vector Sync:** Parses text into 1,000-character chunks (200-character overlap) using a local HuggingFace embedding model (`all-MiniLM-L6-v2`) and interfaces with a serverless Pinecone cloud database.

##  Tech Stack
* **Frontend:** Streamlit
* **Orchestration:** LangChain Expression Language (LCEL)
* **Vector DB:** Pinecone Serverless (AWS us-east-1)
* **Inference Accelerators:** Groq Cloud API Engine


## 🛠️ Technical Architecture & Pipeline Strategy

The platform relies on a completely decoupled data engineering execution model built via **LangChain Expression Language (LCEL)**, managing asynchronous communication between the client-facing front-end and the persistent cloud database layer.

### 1. Persistent Asynchronous Ingestion Pipeline
* **Text Chunking Optimization:** Employs a `RecursiveCharacterTextSplitter` configured to an optimized target window of `1000` characters with a `200` character overlap. This strategy dynamically maps word spaces, newlines, and paragraphs to safeguard the structural integrity of semantic data while preserving unbroken contextual bridges across split boundaries.
* **Localized Vector Embeddings:** Text chunks are transformed locally into a `384`-dimensional mathematical vector space via the `sentence-transformers/all-MiniLM-L6-v2` stack, eliminating runtime API pricing overhead and removing external network processing latencies.
* **Cloud Database Persistence:** Dense vector layouts are streamed directly into a cloud-hosted serverless **Pinecone** index running on top of AWS (`us-east-1` infrastructure), relying on **Cosine Similarity** metrics for precise geometric alignment.

### 2. Context-Constrained Inference Pipeline
* **Automated Lifecycle Boot Gate:** To systematically mitigate the script-refresh overhead native to Streamlit runtimes, an intelligent boot sequence calls Pinecone's `describe_index_stats()` API on startup. If vectors already exist in the cloud ecosystem, the application safely bypasses local directory scanning, parsing, and uploading altogether—saving system compute.
* **Deterministic Inference Layer:** User prompts trigger a vector store retriever to pull the top $k=3$ most contextually relative blocks. The query and context are encapsulated inside a strict, zero-hallucination system prompt, routed to the `llama-3.3-70b-versatile` framework via the **Groq Cloud API Acceleration Layer**. Anchoring the generation hyperparameters to an absolute temperature setting of `0.0` ensures deterministic, highly factual data extraction.

---

## 🚀 Minimum Deliverables Checklist

- [x] **Functional RAG Chatbot:** Accepting technical user queries and returning strictly grounded text outputs through a streamlined web UI dashboard.
- [x] **LCEL Retrieval Chain:** Implemented completely via modular LangChain Expression Language components.
- [x] **Cloud Vector DB Integration:** Decoupling multi-page PDF matrix state memory to an AWS serverless cloud-hosted Pinecone index.
- [x] **Zero-Hallucination Guardrail Verification:** Rigorously audited against both target-grounded technical questions and out-of-distribution unanswerable inquiries.

---

## 💻 Local Installation & Setup

Follow these steps to launch the LitIntel Engine research assistant platform inside your local environment:
    ### 1. Clone the Repository
   ```bash
    git clone [https://github.com/Hridyanmohan/RAG_Gen-AI_Chatbot.git](https://github.com/Hridyanmohan/RAG_Gen-AI_Chatbot.git)
    cd RAG_Gen-AI_Chatbot

##  Local Setup
2. Set Up a Virtual Environment & Dependencies
   ```bash
   # Create environment
    python -m venv venv

   # Activate environment (Windows)
   venv\Scripts\activate

   # Activate environment (Mac/Linux)
   source venv/bin/activate

    # Install required pipeline libraries
    pip install -r requirements.txt
