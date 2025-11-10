# --- Secure app.py (backend) ---
import os
import time
import logging
import re
from collections import defaultdict
from typing import Dict, Optional
from datetime import datetime

import requests
from fastapi import FastAPI, Request, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator, ValidationError
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.memory.buffer import ConversationBufferMemory

# -------------------- LOGGING CONFIGURATION -------------------- #
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()
logger.info("🚀 Starting StartAI Advisory Chatbot Backend...")

# -------------------- CONFIGURATION -------------------- #
# Load secrets from environment variables only (never hardcode)
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

# Validate all required environment variables
if not OPENROUTER_API_KEY:
    logger.error("❌ OPENROUTER_API_KEY is missing")
    raise ValueError("❌ OPENROUTER_API_KEY is required")
if not QDRANT_URL:
    logger.error("❌ QDRANT_URL is missing")
    raise ValueError("❌ QDRANT_URL is required")
if not QDRANT_API_KEY:
    logger.error("❌ QDRANT_API_KEY is missing")
    raise ValueError("❌ QDRANT_API_KEY is required")

logger.info("✅ Environment variables loaded securely")

# -------------------- QDRANT INITIALIZATION -------------------- #
try:
    qdrant = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
    logger.info("✅ Qdrant client initialized")
except Exception as e:
    logger.error(f"❌ Qdrant connection failed: {e}")
    raise

# -------------------- LAZY LOAD EMBEDDINGS -------------------- #
_emb_model = None

def get_embeddings():
    """Lazy load embeddings model on first use to improve startup time."""
    global _emb_model
    if _emb_model is None:
        logger.info("📥 Loading HuggingFace embeddings model (first time only)...")
        _emb_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        logger.info("✅ Embeddings model loaded")
    return _emb_model

# -------------------- FASTAPI APP INITIALIZATION -------------------- #
app = FastAPI(
    title="StartAI Advisory API",
    description="Secure persona-based advisory chatbot API",
    version="1.0.0",
    docs_url=None,  # Disable docs in production for security
    redoc_url=None
)

# -------------------- CORS CONFIGURATION -------------------- #
app.add_middleware(
    CORSMiddleware,
    allow_origins=['https://startaii.netlify.app'],  # Strict origin control
    allow_credentials=True,
    allow_methods=["GET", "POST"],  # Only allow necessary methods
    allow_headers=["Content-Type", "Authorization"],  # Restrict headers
)

logger.info("✅ FastAPI app initialized with restricted CORS")

# -------------------- RATE LIMITING -------------------- #
# Simple in-memory rate limiter (per session)
rate_limit_store: Dict[str, list] = defaultdict(list)
MAX_REQUESTS_PER_MINUTE = 10
RATE_LIMIT_WINDOW = 60  # seconds

def check_rate_limit(session_id: str) -> bool:
    """
    Check if the session has exceeded rate limit.
    Returns True if allowed, False if rate limit exceeded.
    """
    current_time = time.time()

    # Clean up old entries
    rate_limit_store[session_id] = [
        timestamp for timestamp in rate_limit_store[session_id]
        if current_time - timestamp < RATE_LIMIT_WINDOW
    ]

    # Check if limit exceeded
    if len(rate_limit_store[session_id]) >= MAX_REQUESTS_PER_MINUTE:
        return False

    # Add current request timestamp
    rate_limit_store[session_id].append(current_time)
    return True

# -------------------- SECURITY HEADERS MIDDLEWARE -------------------- #
@app.middleware("http")
async def add_security_headers(request: Request, call_next):
    """Add security headers to all responses."""
    response = await call_next(request)

    # HSTS: Force HTTPS for 2 years
    response.headers["Strict-Transport-Security"] = "max-age=63072000; includeSubDomains"

    # Prevent MIME type sniffing
    response.headers["X-Content-Type-Options"] = "nosniff"

    # Prevent clickjacking
    response.headers["X-Frame-Options"] = "DENY"

    # Enable XSS protection
    response.headers["X-XSS-Protection"] = "1; mode=block"

    # Content Security Policy
    response.headers["Content-Security-Policy"] = "default-src 'self'"

    return response

# -------------------- REQUEST LOGGING MIDDLEWARE -------------------- #
@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log all incoming requests (without sensitive data)."""
    start_time = time.time()

    # Log request details (excluding body content for security)
    logger.info(f"Request: {request.method} {request.url.path} from {request.client.host}")

    response = await call_next(request)

    # Log response time
    process_time = time.time() - start_time
    logger.info(f"Response: {request.url.path} completed in {process_time:.2f}s with status {response.status_code}")

    return response

# -------------------- PYDANTIC MODELS FOR VALIDATION -------------------- #
class ChatRequest(BaseModel):
    """Validated request model for /chat endpoint."""
    persona: str = Field(..., min_length=1, max_length=50, description="AI persona name")
    query: str = Field(..., min_length=1, max_length=500, description="User query")
    session_id: str = Field(..., min_length=1, max_length=100, description="Session identifier")

    @validator('persona')
    def validate_persona(cls, v):
        """Ensure persona contains only safe characters."""
        if not re.match(r'^[a-zA-Z0-9\s\-]+$', v):
            raise ValueError("Persona must contain only alphanumeric characters, spaces, and hyphens")
        return v.strip()

    @validator('query')
    def validate_query(cls, v):
        """Sanitize query input."""
        # Remove any potential HTML/script tags
        cleaned = re.sub(r'<[^>]+>', '', v)
        return cleaned.strip()

    @validator('session_id')
    def validate_session_id(cls, v):
        """Ensure session_id is alphanumeric with underscores."""
        if not re.match(r'^[a-zA-Z0-9_\-]+$', v):
            raise ValueError("Session ID must be alphanumeric with underscores or hyphens")
        return v.strip()

    class Config:
        json_schema_extra = {
            "example": {
                "persona": "Elon Musk",
                "query": "How can I build a sustainable rocket company?",
                "session_id": "user_c3i1jp93ank"
            }
        }

class ChatResponse(BaseModel):
    """Structured response model for /chat endpoint."""
    persona: str
    response: str

class HealthResponse(BaseModel):
    """Response model for /health endpoint."""
    status: str
    message: str

class ErrorResponse(BaseModel):
    """Standard error response format."""
    detail: str

# -------------------- SESSION MEMORY MANAGEMENT -------------------- #
session_memories: Dict[str, ConversationBufferMemory] = {}

def get_session_memory(session_id: str) -> ConversationBufferMemory:
    """Get or create session memory."""
    if session_id not in session_memories:
        session_memories[session_id] = ConversationBufferMemory()
        logger.info(f"Created new session memory for: {session_id}")
    return session_memories[session_id]

# -------------------- GLOBAL EXCEPTION HANDLER -------------------- #
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """
    Catch all unhandled exceptions and return sanitized error responses.
    Never expose internal stack traces to clients.
    """
    # Log the full error internally
    logger.error(f"Unhandled exception: {type(exc).__name__}: {str(exc)}", exc_info=True)

    # Return generic error to client
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"error": "Internal server error. Please try again later."}
    )

@app.exception_handler(ValidationError)
async def validation_exception_handler(request: Request, exc: ValidationError):
    """Handle Pydantic validation errors."""
    logger.warning(f"Validation error: {exc}")
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={"detail": "Invalid input: persona and query fields are required."}
    )

# -------------------- API ENDPOINTS -------------------- #

@app.get("/", include_in_schema=False)
async def root():
    """Root endpoint redirect to health check."""
    return {"message": "StartAI Advisory API - Use /health for status"}

@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """
    Health check endpoint to verify API is running.
    Returns minimal information for security.
    """
    logger.info("Health check requested")
    return HealthResponse(
        status="ok",
        message="StartAI backend running securely."
    )

@app.post("/chat", response_model=ChatResponse, tags=["Chat"], responses={
    200: {"description": "Successful response with AI advice"},
    422: {"model": ErrorResponse, "description": "Invalid input data"},
    429: {"model": ErrorResponse, "description": "Rate limit exceeded"},
    500: {"model": ErrorResponse, "description": "Internal server error"}
})
async def chat_endpoint(request: ChatRequest):
    """
    Main chat endpoint for persona-based advisory.

    Security features:
    - Input validation via Pydantic
    - Rate limiting per session
    - Sanitized error responses
    - No sensitive data logging
    """
    try:
        # Rate limiting check
        if not check_rate_limit(request.session_id):
            logger.warning(f"Rate limit exceeded for session: {request.session_id}")
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Rate limit exceeded. Try again later."
            )

        logger.info(f"Processing chat request for persona: {request.persona}")

        # Get session memory
        memory = get_session_memory(request.session_id)

        # Get embeddings (lazy loaded)
        embeddings = get_embeddings()

        # Generate query embedding
        query_vector = embeddings.embed_query(request.query)

        # Search Qdrant for relevant context
        search_results = qdrant.search(
            collection_name="persona_knowledge",
            query_vector=query_vector,
            limit=3
        )

        # Extract context from search results
        context = "\n".join([
            result.payload.get("text", "") 
            for result in search_results 
            if hasattr(result, 'payload')
        ])

        # Get conversation history
        history = memory.load_memory_variables({})
        conversation_context = history.get("history", "")

        # Build prompt for LLM
        system_prompt = f"""You are {request.persona}, providing advice based on your expertise and philosophy.

Context from knowledge base:
{context}

Previous conversation:
{conversation_context}

Provide thoughtful, actionable advice in the voice and style of {request.persona}."""

        # Call OpenRouter API
        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                "Content-Type": "application/json"
            },
            json={
                "model": "meta-llama/llama-3.1-8b-instruct:free",
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": request.query}
                ]
            },
            timeout=30  # 30 second timeout
        )

        response.raise_for_status()
        ai_response = response.json()["choices"][0]["message"]["content"]

        # Save to memory
        memory.save_context(
            {"input": request.query},
            {"output": ai_response}
        )

        logger.info(f"Successfully generated response for session: {request.session_id}")

        return ChatResponse(
            persona=request.persona,
            response=ai_response
        )

    except requests.exceptions.Timeout:
        logger.error("OpenRouter API timeout")
        raise HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            detail="Request timeout. Please try again."
        )
    except requests.exceptions.RequestException as e:
        logger.error(f"OpenRouter API error: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="AI service temporarily unavailable."
        )
    except Exception as e:
        # This will be caught by global exception handler
        logger.error(f"Unexpected error in chat endpoint: {e}")
        raise

# -------------------- STARTUP EVENT -------------------- #
@app.on_event("startup")
async def startup_event():
    """Log startup information."""
    logger.info("=" * 50)
    logger.info("StartAI Advisory API Started Successfully")
    logger.info(f"Allowed CORS Origins: 'https://startaii.netlify.app'")
    logger.info(f"Rate Limit: {MAX_REQUESTS_PER_MINUTE} requests per {RATE_LIMIT_WINDOW} seconds")
    logger.info("=" * 50)

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    logger.info("Shutting down StartAI Advisory API")
    session_memories.clear()
