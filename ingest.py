import os
import markdown
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.http import models as rest
from langchain_community.embeddings import HuggingFaceEmbeddings
from qdrant_client.http import models as rest

# Use open-source embedding model (no Google key required)

load_dotenv()

qdrant = QdrantClient(url=os.getenv("QDRANT_URL"), api_key=os.getenv("QDRANT_API_KEY"))
collection = "personas"
emb = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

if qdrant.collection_exists("personas"):
    qdrant.delete_collection("personas")

qdrant.create_collection(
    collection_name="personas",
    vectors_config=rest.VectorParams(size=768, distance=rest.Distance.COSINE),
)

for file in os.listdir("data"):
    if file.endswith(".md"):
        with open(f"data/{file}", "r", encoding="utf-8") as f:
            md_text = f.read()
            html_content = markdown.markdown(md_text)
            vector = emb.embed_query(html_content)
            qdrant.upsert(
                collection_name=collection,
                points=[
                    rest.PointStruct(
                        id=file,
                        vector=vector,
                        payload={
                            "name": file.replace(".md", ""),
                            "content": html_content
                        },
                    )
                ],
            )
print("✅ Persona data successfully stored in Qdrant.")
