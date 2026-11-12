from fastapi import APIRouter, Request
from app.models import ChatRequest
from app.services.rag_service import rag_service
from fastapi.responses import StreamingResponse
from app.core.limiter import limiter

router = APIRouter(prefix="/chat", tags=["Chat"])

@router.post("")
@limiter.limit("5/minute")
async def chat_stream(
    request: Request,
    payload: ChatRequest
):
    return StreamingResponse(
        rag_service.generate_streaming_response(payload.message, payload.history),
        media_type="text/event-stream"
    )