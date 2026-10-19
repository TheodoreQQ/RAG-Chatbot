import pytest
from unittest.mock import MagicMock, patch
from langchain_core.messages import AIMessage
from app.services.rag_service import RAGService

@pytest.fixture
def mock_rag_service():
    """Tworzy instancję RAGService z zamockowaną bazą ChromaDB, Rerankerem i modelem LLM."""
    with patch("app.services.rag_service.VectorStoreManager"), \
         patch("app.services.rag_service.Reranker"), \
         patch("app.services.rag_service.ChatOllama"), \
         patch("app.services.rag_service.DocumentLoader"):
        
        service = RAGService()
        
        # Domyślne zachowanie mocków
        service.vector_store.search = MagicMock(return_value=[])
        service.reranker.rerank = MagicMock(return_value=[])
        
        # Mock LLM zwracający pustą odpowiedź strumieniową
        service.llm.stream = MagicMock(return_value=[AIMessage(content="Odpowiedź testowa.")])
        service.llm.invoke = MagicMock(return_value=AIMessage(content="Samodzielne pytanie"))
        
        return service