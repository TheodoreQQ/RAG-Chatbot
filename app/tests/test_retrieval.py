import pytest
from langchain_core.documents import Document
from app.core.reranker import Reranker

def test_reranker_sorting():
    reranker = Reranker()
    
    query = "Ile wynosi dofinansowanie do pracy zdalnej?"
    docs = [
        Document(page_content="Kawa w kuchni jest darmowa dla każdego.", metadata={"source": "kawa.pdf"}),
        Document(page_content="Dofinansowanie do pracy zdalnej wynosi 150 PLN miesięcznie.", metadata={"source": "regulamin.pdf"}),
        Document(page_content="Godziny otwarcia biura to 8:00 - 16:00.", metadata={"source": "biuro.pdf"}),
    ]
    
    results = reranker.rerank(query, docs, top_n=1)
    
    # Oczekujemy, że TOP 1 będzie dokumentem o pracy zdalnej!
    assert len(results) == 1
    assert "150 PLN" in results[0].page_content
    assert results[0].metadata["source"] == "regulamin.pdf"