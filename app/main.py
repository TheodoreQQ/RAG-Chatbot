from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Importujemy routery z folderu app/routers
from app.routers import chat, documents
from app.config import ALLOWED_ORIGINS

# Tworzymy główną instancję aplikacji FastAPI
app = FastAPI(
    title="RAG Chatbot API",
    description="Backend dla czatbota RAG z obsługą dokumentów PDF i Re-rankingiem",
    version="1.0.0"
)

# Konfiguracja CORS (umożliwia komunikację z frontendem w React)
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,  # W środowisku produkcyjnym podaj konkretny URL np. "http://localhost:5173"
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Wpinamy routery podłączające endpointy /api/chat oraz /api/documents
app.include_router(chat.router)
app.include_router(documents.router)

# Healthcheck / prosty endpoint sprawdzający stan serwera
@app.get("/")
def read_root():
    return {
        "status": "online",
        "message": "Serwer RAG działa poprawnie!"
    }

# Uruchomienie lokalne za pomocą: uvicorn app.main:app --reload
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="127.0.0.1", port=8000, reload=True)