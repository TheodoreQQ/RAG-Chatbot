from fastapi import APIRouter
from app.models import ChatRequest
from app.services.rag_service import rag_service
from fastapi.responses import StreamingResponse

router = APIRouter(prefix="/chat", tags=["Chat"])

@router.post("")
async def chat_stream(request: ChatRequest):
    return StreamingResponse(
        rag_service.generate_streaming_response(request.message, request.history),
        media_type="text/event-stream"
    )