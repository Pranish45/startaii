import os
import requests
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.memory.buffer import ConversationBufferMemory

# Load environment variables
load_dotenv()

print("🚀 Starting StartAI Advisory Chatbot Backend...")

# -------------------- CONFIGURATION -------------------- #
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

# Validate environment variables
if not OPENROUTER_API_KEY:
    raise ValueError("❌ OPENROUTER_API_KEY is required")
if not QDRANT_URL:
    raise ValueError("❌ QDRANT_URL is required")
if not QDRANT_API_KEY:
    raise ValueError("❌ QDRANT_API_KEY is required")

print("✅ Environment variables loaded")

# Initialize Qdrant
try:
    qdrant = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
    print("✅ Qdrant client initialized")
except Exception as e:
    print(f"❌ Qdrant connection failed: {e}")
    raise

# LAZY LOAD EMBEDDINGS: Don't load here, load on first use
_emb_model = None

def get_embeddings():
    """Lazy load embeddings model on first use."""
    global _emb_model
    if _emb_model is None:
        print("📥 Loading HuggingFace embeddings model (first time only)...")
        _emb_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        print("✅ Embeddings model loaded")
    return _emb_model

# FastAPI App
app = FastAPI(title="StartAI Advisory API")

# CORS configuration - Allow Netlify frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://startaii.netlify.app",
        "http://localhost:3000",
        "http://localhost:5500"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

print("✅ FastAPI app initialized with CORS")

# Session me
