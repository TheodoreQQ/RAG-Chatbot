import os
from typing import List
import requests
import chromadb
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.documents import Document
from app.config import CHROMA_PATH, EMBEDDING_MODEL

class VectorStoreManager:
    def __init__(self):
        self.ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://host.docker.internal:11434")
        
        # 1. Embeddings dla Ollamy
        self.emb_ollama = OllamaEmbeddings(
            model=EMBEDDING_MODEL,
            base_url=self.ollama_base_url
        )
        
        # 2. Embeddings dla Gemini
        google_api_key = os.getenv("GOOGLE_API_KEY")
        self.emb_gemini = GoogleGenerativeAIEmbeddings(
            model="gemini-embedding-001",
            google_api_key=google_api_key
        )

        chroma_host = os.getenv("CHROMA_HOST", None)
        chroma_port = int(os.getenv("CHROMA_PORT", 8000))

        if chroma_host:
            client = chromadb.HttpClient(host=chroma_host, port=chroma_port)
        else:
            client = chromadb.PersistentClient(path=CHROMA_PATH)

        # stworzenie dwóch osobnych kolekcji
        self.store_ollama = Chroma(
            client=client,
            collection_name="docs_ollama",
            embedding_function=self.emb_ollama
        )
        
        self.store_gemini = Chroma(
            client=client,
            collection_name="docs_gemini",
            embedding_function=self.emb_gemini
        )

    def add_documents(self, documents: List[Document]):
        # zapis do obu kolekcji
        try:
            requests.get(self.ollama_base_url, timeout=2)
        except Exception: 
            raise Exception("System działa w trybie awaryjnym (Tylko do odczytu). "
                "Główny silnik (Ollama) jest wyłączony. Wgrywanie nowych dokumentów zostało "
                "zablokowane, aby zapobiec rozspójnieniu baz danych. Nadal możesz korzystać z czatu!")
        try:
            self.store_ollama.add_documents(documents)
            print("[INFO] Wektory zapisane w lokalnej kolekcji (Ollama)")
        except Exception as e:
            raise Exception(f"Błąd wektoryzacji lokalnej {e}")

        try:
            self.store_gemini.add_documents(documents)
            print("[INFO] Wektory zapisane w chmurowej kolekcji (Gemini)")
        except Exception as e:
            # nie blokuję dodawania dokumentów w przypadku, gdy gemini jest niedostępne
            # Ollama to główny model, więc tylko Warning zostanie rzucony
            print(f"[WARNING] Błąd zapisywania wektorów w Gemini: {e}")

    def search(self, query: str, k: int = 20) -> List[Document]:
        # dynamiczny fallback, sprawdzenie przy każdym pytaniu, czy Ollama żyje
        try:
            import requests
            requests.get(self.ollama_base_url, timeout=1.5)
            # Jeśli nie ma błędu -> używamy kolekcji Ollamy
            print("[INFO] Wyszukiwanie w wektorach: Ollama")
            retriever = self.store_ollama.as_retriever(search_kwargs={"k": k})
            return retriever.invoke(query)
        except Exception as e:
            # Ollama leży -> wyciąganie danych z kolekcji Gemini
            print("[WARNING] Ollama wyłączona! Wyszukiwanie w wektorach: Gemini")
            retriever = self.store_gemini.as_retriever(search_kwargs={"k": k})
            return retriever.invoke(query)

    def delete_document(self, file_name: str):
        clean_filename = os.path.basename(file_name)
        self.store_ollama._collection.delete(where={"source": clean_filename})
        self.store_gemini._collection.delete(where={"source": clean_filename})