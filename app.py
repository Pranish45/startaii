import os
import requests
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.memory.buffer import ConversationBufferMemory

# Load environment variables
load_dotenv()

print("🚀 Starting StartAI Advisory Chatbot...")

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
app = FastAPI(title="StartAI Advisory Chatbot")

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://startaii.netlify.app/ai_advisory_page", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

print("✅ FastAPI app initialized")

# Session memory
user_sessions = {}


# -------------------- MODELS -------------------- #
class QueryIn(BaseModel):
    persona: str
    query: str
    session_id: str


# -------------------- HELPER FUNCTIONS -------------------- #
def openrouter_chat(model: str, prompt: str):
    """Generic OpenRouter API call."""
    try:
        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.7,
                "max_tokens": 500,
            },
            timeout=30,
        )
        data = response.json()
        return data["choices"][0]["message"]["content"]
    except Exception as e:
        print(f"OpenRouter API error ({model}): {e}")
        return ""


def retrieve_context(query):
    """Retrieve relevant context from Qdrant and supplement with DeepSeek."""
    try:
        emb = get_embeddings()  # Lazy load here
        vector = emb.embed_query(query)
        results = qdrant.search(collection_name="personas", query_vector=vector, limit=3)
        context = "\n\n".join([r.payload.get("content", "") for r in results])

        if len(context.strip()) < 200:
            print("Context short — fetching supplemental info from DeepSeek...")
            supplemental = openrouter_chat("deepseek/deepseek-chat-v3.1:free", query)
            context += "\n\nRecent Insights (via DeepSeek):\n" + supplemental

        return context

    except Exception as e:
        print(f"Context retrieval error: {e}")
        return ""


def generate_reply(persona, query, session_id):
    """Generate persona-based AI response using Gemma through OpenRouter."""
    context = retrieve_context(query)

    if session_id not in user_sessions:
        user_sessions[session_id] = ConversationBufferMemory(memory_key="history", return_messages=True)

    memory = user_sessions[session_id]
    previous_conversation = "\n".join(
        [f"{m.type}: {m.content}" for m in memory.chat_memory.messages[-6:]]
    )

    persona_styles = {
        "Elon Musk": "You are Elon Musk. Think from first principles, challenge assumptions, focus on engineering and long-term impact. Be bold and analytical.",
        "Steve Jobs": "You are Steve Jobs. Focus on simplicity, perfection, and design excellence. Be visionary and passionate about innovation.",
        "Ratan Tata": "You are Ratan Tata. Emphasize ethics, compassion, and national development. Stay calm, humble, and purpose-driven."
    }

    system_prompt = persona_styles.get(persona, f"You are {persona}, an influential entrepreneur.")

    prompt = f"""{system_prompt}

Previous conversation:
{previous_conversation}

Relevant knowledge and context:
{context}

User question: {query}

Respond as {persona} would — authentic, insightful, and focused.
"""

    try:
        reply = openrouter_chat("google/gemma-3n-e2b-it:free", prompt)
        memory.chat_memory.add_user_message(query)
        memory.chat_memory.add_ai_message(reply)
        return reply if reply else "I could not generate a full response at this time."

    except Exception as e:
        print(f"Gemma generation error: {e}")
        return "I am temporarily unavailable. Please try again later."


# -------------------- ROUTES -------------------- #
@app.post("/chat")
def chat(payload: QueryIn):
    reply = generate_reply(payload.persona, payload.query, payload.session_id)
    return {"persona": payload.persona, "response": reply}


@app.get("/health")
def health():
    return {"status": "ok", "message": "StartAI Advisory is running"}


@app.get("/")
def read_root():
    try:
        return FileResponse("ai_advisory_page.html")
    except Exception as e:
        return {"error": f"HTML file not found: {e}"}


# Startup event
@app.on_event("startup")
async def startup_event():
    print("=" * 50)
    print("✅ FastAPI server started successfully!")
    print(f"✅ Listening on port {os.environ.get('PORT', '10000')}")
    print("=" * 50)


print("✅ App initialization complete. Ready to start uvicorn.")
