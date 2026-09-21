import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_root_endpoint():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["status"] == "online"

def test_list_documents_endpoint():
    response = client.get("/documents")  # lub /api/documents w zależności od main.py
    assert response.status_code == 200
    assert "documents" in response.json()

def test_invalid_file_upload():
    # Próba wysłania pliku .txt zamiast .pdf
    files = {"file": ("test.txt", b"to jest plik tekstowy", "text/plain")}
    response = client.post("/documents", files=files)
    
    # Oczekujemy błędu 400 (Walidacja rozszerzenia)
    assert response.status_code == 400
    assert "Tylko obsługa plików PDF" in response.json()["detail"]