import os
from dotenv import load_dotenv

load_dotenv()

HOST = os.getenv("HOST", "127.0.0.1")
PORT = int(os.getenv("PORT", 8000))

UPLOAD_DIR = os.getenv("UPLOAD_DIR", "./documents")
CHROMA_PATH = os.getenv("CHROMA_PATH", "./chroma_db")

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "nomic-embed-text")
LLM_MODEL = os.getenv("LLM_MODEL", "qwen2.5:7b")

ALLOWED_ORIGINS = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "*"  # Zezwala na połączenia z lokalnych plików .html
]
os.makedirs(UPLOAD_DIR, exist_ok=True)