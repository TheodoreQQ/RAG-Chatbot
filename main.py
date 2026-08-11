import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import HOST, PORT, ALLOWED_ORIGINS
from app.routers import chat, documents

app = FastAPI(
  title="RAG Knowledge Base API",
  description="Backend API do obsługi systemu RAG",
  version="1.0.0"
)

app.add_middleware(
  CORSMiddleware,
  allow_origins=ALLOWED_ORIGINS if ALLOWED_ORIGINS else ["*"],
  allow_credentials=True,
  allow_methods=["*"],
  allow_headers=["*"],
)

app.include_router(chat.router)
app.include_router(documents.router)

if __name__ == "__main__":
  uvicorn.run("main:app", host=HOST, port=PORT, reload=True)