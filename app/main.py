from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded


from app.routers import chat, documents
from app.config import ALLOWED_ORIGINS
from app.core.limiter import limiter

# główna instancję aplikacji FastAPI
app = FastAPI(
    title="RAG Chatbot API",
    description="Backend dla czatbota RAG z obsługą dokumentów PDF i Re-rankingiem",
    version="1.0.0"
)
# inicjalizacja limitera
app.state.limiter = limiter 
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Konfiguracja CORS 
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,  
    allow_credentials=True,
    allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization", "Accept"],
)

# routery podłączające endpointy
app.include_router(chat.router)
app.include_router(documents.router)

# healthcheck 
@app.get("/")
def read_root():
    return {
        "status": "online",
        "message": "Serwer RAG działa poprawnie!"
    }

 
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000)