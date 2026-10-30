import os
from typing import List
import chromadb
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_core.documents import Document
from app.config import CHROMA_PATH, EMBEDDING_MODEL

class VectorStoreManager:
    def __init__(self):
        # Konfiguracja adresu Ollama (w kontenerze lub lokalnie)
        ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://host.docker.internal:11434")
        self.embeddings = OllamaEmbeddings(
            model=EMBEDDING_MODEL,
            base_url=ollama_base_url
        )

        chroma_host = os.getenv("CHROMA_HOST", None)
        chroma_port = int(os.getenv("CHROMA_PORT", 8000))

        if chroma_host:
            # W DOCKERZE: Łączenie z osobnym kontenerem ChromaDB przez HTTP
            client = chromadb.HttpClient(host=chroma_host, port=chroma_port)
            self.vector_store = Chroma(
                client=client,
                collection_name="documents",
                embedding_function=self.embeddings
            )
        else:
            # LOKALNIE: Dotychczasowy zapis w folderze na dysku
            self.vector_store = Chroma(
                persist_directory=CHROMA_PATH,
                embedding_function=self.embeddings
            )

    def add_documents(self, documents: List[Document]):
        self.vector_store.add_documents(documents)

    def search(self, query: str, k: int = 20) -> List[Document]:
        """Pobiera wstępne 'k' kandydatów do Re-rankingu."""
        retriever = self.vector_store.as_retriever(search_kwargs={"k": k})
        return retriever.invoke(query)

    def delete_document(self, file_name: str):
        """Precyzyjne, trwałe usuwanie z bazy po nazwie pliku."""
        clean_filename = os.path.basename(file_name)
        collection = self.vector_store._collection
        collection.delete(where={"source": clean_filename})