import os
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_openai import OpenAIEmbeddings

# Global vector store instance
VECTOR_STORE = None
CHROMA_PATH = "chroma_db"

def get_embeddings():
    """Select the embedding backend based on available API keys."""
    # if os.getenv("GOOGLE_API_KEY"):
    #     return GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    if os.getenv("OPENROUTER_API_KEY"):
        # Use OpenRouter-compatible endpoint via the OpenAI client.
        return OpenAIEmbeddings(
            model=os.getenv("OPENROUTER_EMBED_MODEL", "text-embedding-3-small"),
            api_key=os.getenv("OPENROUTER_API_KEY"),
            base_url=os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
        )

    raise ValueError("No embedding API key found. Set GOOGLE_API_KEY or OPENROUTER_API_KEY.")

def initialize_vector_store():
    global VECTOR_STORE
    embeddings = get_embeddings()
    VECTOR_STORE = Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=embeddings
    )

def ingest_file(file_path: str):
    global VECTOR_STORE
    if VECTOR_STORE is None:
        initialize_vector_store()
    
    if file_path.endswith(".pdf"):
        loader = PyPDFLoader(file_path)
    else:
        loader = TextLoader(file_path)
        
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)
    
    VECTOR_STORE.add_documents(splits)
    return len(splits)

def get_retriever():
    global VECTOR_STORE
    if VECTOR_STORE is None:
        initialize_vector_store()
    return VECTOR_STORE.as_retriever(search_kwargs={"k": 3})
