from fastapi import APIRouter
from app.models import ChatRequest, ChatResponse
from app.rag_engine import rag_engine

router = APIRouter(prefix="/chat", tags=["Chat"])

@router.post("", response_model=ChatResponse)
async def chat(request: ChatRequest):
  reponse_text = rag_engine.generate_response(request.message, request.history)
  return ChatResponse(response=reponse_text)