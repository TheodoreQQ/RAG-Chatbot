import pytest
from unittest.mock import MagicMock
from langchain_core.documents import Document
from langchain_core.messages import AIMessage
from app.models import ChatMessage
from app.services.rag_service import RAGService
from unittest.mock import patch

def test_empty_retrieval_returns_fallback_refusal(mock_rag_service):
    """Gdy baza nie znajdzie fragmentów (context_text = ""), serwis zwraca zdefiniowaną formułkę odmowy."""
    mock_rag_service.vector_store.search.return_value = []
    mock_rag_service.reranker.rerank.return_value = []

    generator = mock_rag_service.generate_streaming_response(
        message="O co chodzi w art 5?",
        history=[]
    )
    result = "".join(list(generator))

    assert "Nie jestem w stanie udzielić odpowiedzi na zadane pytanie." in result
    # upewnienie się, że LLM w ogóle nie był wywoływany przy braku kontekstu
    mock_rag_service.llm.stream.assert_not_called()


def test_sources_and_page_deduplication(mock_rag_service):
    """Sprawdza, czy backend poprawnie grupuje numery stron (indeksowane od 1) i pliki w stopce Źródła."""
    mock_docs = [
        Document(page_content="Tekst o PL/SQL 1", metadata={"source": "Notatki.pdf", "page": 2}),
        Document(page_content="Tekst o PL/SQL 2", metadata={"source": "Notatki.pdf", "page": 5}),
        Document(page_content="Tekst o PL/SQL 3", metadata={"source": "Notatki.pdf", "page": 2}), # duplikat strony
        Document(page_content="Inny dokument", metadata={"source": "Regulamin.pdf", "page": 0}),
    ]

    mock_rag_service.vector_store.search.return_value = mock_docs
    mock_rag_service.reranker.rerank.return_value = mock_docs
    
    # symulacja merytorycznej odpowiedzi LLM
    mock_rag_service.llm.stream.return_value = [
        AIMessage(content="CLOB to typ danych large object.")
    ]

    generator = mock_rag_service.generate_streaming_response(
        message="Czym jest CLOB?",
        history=[]
    )
    result = "".join(list(generator))

    # weryfikacja treści
    assert "CLOB to typ danych large object." in result
    # weryfikacja formatowania źródeł (strony +1 oraz deduplikacja)
    assert "Źródła:" in result
    assert "Notatki.pdf (strona 3, 6)" in result
    assert "Regulamin.pdf (strona 1)" in result


def test_non_text_document_upload_raises_value_error(mock_rag_service):
    """Sprawdza, czy próba dodania pliku bez warstwy tekstowej rzuca ValueError."""
    # symulacja chunkera, który nie wyciągnął żadnego tekstu
    empty_chunk = Document(page_content="   ", metadata={})
    mock_rag_service.loader.load_and_split = MagicMock(return_value=[empty_chunk])

    with pytest.raises(ValueError) as exc_info:
        mock_rag_service.add_document("/fake/path.pdf", "pusty_skan.pdf")

    assert "nie zawiera tekstu" in str(exc_info.value)
    
@patch("app.services.rag_service.requests.get")
def test_fallback_to_gemini_when_ollama_fails(mock_requests_get, mock_rag_service):
    """Sprawdza, czy po błędzie Ollamy (np. timeout), system przełącza się na Gemini."""
    mock_requests_get.side_effect = Exception("Timeout")
    
    # mock bazy i rerankera
    mock_rag_service.vector_store.search.return_value = [Document(page_content="Testowy tekst", metadata={})]
    mock_rag_service.reranker.rerank.return_value = [Document(page_content="Testowy tekst", metadata={})]
    
    # mock gemini
    mock_rag_service.gemini_llm = MagicMock()
    mock_rag_service.gemini_llm.stream.return_value = [AIMessage(content="Odpowiedź z gemini.")]
    
    generator = mock_rag_service.generate_streaming_response("Pytanie?", [])
    result = "".join(list(generator))
    
    # test czy wywołało gemini
    mock_rag_service.gemini_llm.stream.assert_called_once()
    
@patch("app.services.rag_service.requests.get")
def test_graceful_degradation_on_total_failure(mock_requests_get, mock_rag_service):
    """Sprawdza, czy przy awarii OBU modeli system się nie wywala, lecz daje czysty komunikat."""
    
    mock_requests_get.side_effect = Exception("Ollama padła")
    mock_rag_service.vector_store.search.return_value = [Document(page_content="Tekst", metadata={})]
    mock_rag_service.reranker.rerank.return_value = [Document(page_content="Tekst", metadata={})]
    
    # symulacja gemini rzucającego bład
    mock_rag_service.gemini_llm = MagicMock()
    mock_rag_service.gemini_llm.stream.side_effect = Exception("Brak tokenów")
    
    generator = mock_rag_service.generate_streaming_response("Pytanie?", [])
    result = "".join(list(generator))
    
    assert "Systemy AI są obecnie niedostępne" in result 
    assert "Źródła:" not in result # test czy nie wypluje przypadkiem źródła 