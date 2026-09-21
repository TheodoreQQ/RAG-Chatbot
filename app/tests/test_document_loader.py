import os
import pytest
from app.core.document_loader import DocumentLoader

def test_document_loader_chunking(tmp_path):
    # Tworzymy atrapę prostego tekstu (lub możesz wskazać ścieżkę do prawdziwego PDF-a)
    loader = DocumentLoader(chunk_size=100, chunk_overlap=20)
    
    # Testujemy, czy loader ma poprawne parametry
    assert loader.text_splitter._chunk_size == 100
    assert loader.text_splitter._chunk_overlap == 20