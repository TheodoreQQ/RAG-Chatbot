import pytest
from unittest.mock import MagicMock
from langchain_core.documents import Document
from langchain_core.messages import AIMessage
from app.models import ChatMessage
from app.services.rag_service import RAGService

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
    # Upewniamy się, że LLM w ogóle nie był wywoływany przy braku kontekstu
    mock_rag_service.llm.stream.assert_not_called()


def test_sources_and_page_deduplication(mock_rag_service):
    """Sprawdza, czy backend poprawnie grupuje numery stron (indeksowane od 1) i pliki w stopce Źródła."""
    mock_docs = [
        Document(page_content="Tekst o PL/SQL 1", metadata={"source": "Notatki.pdf", "page": 2}),
        Document(page_content="Tekst o PL/SQL 2", metadata={"source": "Notatki.pdf", "page": 5}),
        Document(page_content="Tekst o PL/SQL 3", metadata={"source": "Notatki.pdf", "page": 2}), # Duplikat strony
        Document(page_content="Inny dokument", metadata={"source": "Regulamin.pdf", "page": 0}),
    ]

    mock_rag_service.vector_store.search.return_value = mock_docs
    mock_rag_service.reranker.rerank.return_value = mock_docs
    
    # Symulujemy merytoryczną odpowiedź LLM
    mock_rag_service.llm.stream.return_value = [
        AIMessage(content="CLOB to typ danych large object.")
    ]

    generator = mock_rag_service.generate_streaming_response(
        message="Czym jest CLOB?",
        history=[]
    )
    result = "".join(list(generator))

    # Weryfikacja treści
    assert "CLOB to typ danych large object." in result
    # Weryfikacja formatowania źródeł (strony +1 oraz deduplikacja)
    assert "Źródła:" in result
    assert "Notatki.pdf (strona 3, 6)" in result
    assert "Regulamin.pdf (strona 1)" in result


def test_non_text_document_upload_raises_value_error(mock_rag_service):
    """Sprawdza, czy próba dodania pliku bez warstwy tekstowej rzuca ValueError."""
    # Symulujemy chunker, który nie wyciągnął żadnego tekstu
    empty_chunk = Document(page_content="   ", metadata={})
    mock_rag_service.loader.load_and_split = MagicMock(return_value=[empty_chunk])

    with pytest.raises(ValueError) as exc_info:
        mock_rag_service.add_document("/fake/path.pdf", "pusty_skan.pdf")

    assert "nie zawiera tekstu" in str(exc_info.value)