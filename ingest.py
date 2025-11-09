import os
import markdown
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.http import models as rest
from langchain_google_genai import GoogleGenerativeAIEmbeddings

load_dotenv()

qdrant = QdrantClient(url=os.getenv("QDRANT_URL"), api_key=os.getenv("QDRANT_API_KEY"))
collection = "personas"
emb = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

qdrant.recreate_collection(
    collection_name=collection,
    vectors_config={"size": 768, "distance": "Cosine"}
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
