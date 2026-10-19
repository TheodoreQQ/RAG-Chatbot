import io
import pytest
from httpx import AsyncClient, ASGITransport
from unittest.mock import patch
from app.main import app

@pytest.mark.asyncio
async def test_chat_endpoint_streaming():
    """Testuje, czy endpoint /chat przyjmuje zapytanie i zwraca status 200 ze strumieniem."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        payload = {
            "message": "Cześć",
            "history": []
        }
        
        # Mockujemy generator, by nie odpytywać faktycznej Ollamy
        with patch("app.services.rag_service.rag_service.generate_streaming_response") as mock_stream:
            mock_stream.return_value = iter(["Witaj, ", "w czym mogę ", "pomóc?"])
            
            response = await client.post("/chat", json=payload)
            
            assert response.status_code == 200
            assert "Witaj, w czym mogę pomóc?" in response.text


@pytest.mark.asyncio
async def test_upload_non_text_pdf_returns_400():
    """Testuje, czy endpoint uploadu wyłapuje ValueError i zwraca kod HTTP 400 Bad Request."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        
        fake_pdf_content = b"%PDF-1.4 fake content"
        files = {"file": ("skan.pdf", io.BytesIO(fake_pdf_content), "application/pdf")}
        
        # Symulujemy rzucenie ValueError z logiki RAG
        with patch("app.services.rag_service.rag_service.add_document", side_effect=ValueError("Plik nie zawiera tekstu")):
            response = await client.post("/documents", files=files)
            
            assert response.status_code == 400
            assert "nie zawiera tekstu" in response.json()["detail"]