import os
import requests
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.memory.buffer import ConversationBufferMemory

# Load environment variables
load_dotenv()

# -------------------- CONFIGURATION -------------------- #
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

# Initialize Qdrant and embeddings
qdrant = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
emb = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# FastAPI App
app = FastAPI(title="StartAI Advisory Chatbot")

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://startaii.netlify.app"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static files
app.mount("/static", StaticFiles(directory="static"), name="static")

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
        vector = emb.embed_query(query)
        results = qdrant.search(collection_name="personas", query_vector=vector, limit=3)
        context = "\n\n".join([r.payload.get("content", "") for r in results])

        # If context is too short, supplement with DeepSeek (via OpenRouter)
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

    # Persona styles
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

        # Save in memory
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
    return FileResponse("ai_advisory_page.html")


import os
import uvicorn

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
