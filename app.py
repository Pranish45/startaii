import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.memory.buffer import ConversationBufferMemory
import google.generativeai as genai
from perplexityai import Perplexity

load_dotenv()

# Configure APIs
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
perplexity = Perplexity(api_key=os.getenv("PERPLEXITY_API_KEY"))
qdrant = QdrantClient(url=os.getenv("QDRANT_URL"), api_key=os.getenv("QDRANT_API_KEY"))
emb = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

app = FastAPI(title="StartAI Advisory Chatbot")

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://startaii.netlify.app/ai_advisory_page"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static files
app.mount("/static", StaticFiles(directory="static"), name="static")

user_sessions = {}

class QueryIn(BaseModel):
    persona: str
    query: str
    session_id: str

def retrieve_context(query):
    """Retrieve relevant context from Qdrant vector database"""
    try:
        vector = emb.embed_query(query)
        results = qdrant.search(collection_name="personas", query_vector=vector, limit=3)
        context = "\n\n".join([r.payload.get("content", "") for r in results])

        # If context is too short, supplement with Perplexity web search
        if len(context.strip()) < 200:
            try:
                resp = perplexity.chat.completions.create(
                    model="sonar",
                    messages=[{"role": "user", "content": query}]
                )
                context += "\n\nRecent Information:\n" + resp.choices[0].message.content
            except Exception as e:
                print(f"Perplexity error: {e}")

        return context
    except Exception as e:
        print(f"Context retrieval error: {e}")
        return ""

def generate_reply(persona, query, session_id):
    """Generate AI response using RAG + Gemini Pro"""
    # Retrieve relevant context from knowledge base
    context = retrieve_context(query)

    # Initialize or get conversation memory
    if session_id not in user_sessions:
        user_sessions[session_id] = ConversationBufferMemory(memory_key="history", return_messages=True)

    memory = user_sessions[session_id]
    previous_conversation = "\n".join(
        [f"{m.type}: {m.content}" for m in memory.chat_memory.messages[-6:]]
    )

    # Persona-specific system prompts
    persona_styles = {
        "Elon Musk": "You are Elon Musk. Think from first principles, challenge assumptions, focus on physics and engineering. Be ambitious and direct. Use phrases like 'first principles thinking', 'physics-based approach', and 'make life multiplanetary'.",
        "Steve Jobs": "You are Steve Jobs. Focus on simplicity, design excellence, and creating insanely great products. Be passionate about user experience. Use phrases like 'insanely great', 'think different', and 'stay hungry, stay foolish'.",
        "Ratan Tata": "You are Ratan Tata. Emphasize ethical business practices, social responsibility, compassion, and long-term value creation. Be humble yet visionary. Focus on nation-building and creating value for society."
    }

    system_prompt = persona_styles.get(persona, f"You are {persona}, an influential entrepreneur.")

    prompt = f"""{system_prompt}

Previous conversation:
{previous_conversation}

Relevant knowledge from your life and work:
{context}

User question: {query}

Respond as {persona} would, using insights from the context above. Be authentic to your persona's style and philosophy. Keep responses focused, actionable, and inspirational.

{persona}:"""

    try:
        model = genai.GenerativeModel('gemini-pro')
        response = model.generate_content(prompt)
        reply_text = response.text

        # Store in conversation memory
        memory.chat_memory.add_user_message(query)
        memory.chat_memory.add_ai_message(reply_text)

        return reply_text
    except Exception as e:
        print(f"Generation error: {e}")
        return f"I apologize, but I'm having trouble processing your request right now. Please try again."

@app.post("/chat")
def chat(payload: QueryIn):
    """Main chat endpoint"""
    reply = generate_reply(payload.persona, payload.query, payload.session_id)
    return {"persona": payload.persona, "response": reply}

@app.get("/health")
def health():
    """Health check endpoint"""
    return {"status": "ok", "message": "StartAI Advisory is running"}

@app.get("/")
def read_root():
    """Serve the main HTML page"""
    return FileResponse("ai_advisory_page.html")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
