import os
from typing import List
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_core.documents import Document
from app.config import CHROMA_PATH, EMBEDDING_MODEL

class VectorStoreManager:
  def __init__(self):
    self.embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
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
    # Pobieramy natywną kolekcję ChromaDB pod spodem LangChaina
    collection = self.vector_store._collection
    collection.delete(where={"source": clean_filename})